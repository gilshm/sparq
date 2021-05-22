import torch
import torch.nn as nn
import cu_gemm_2x48
import Config as cfg


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UnfoldConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(UnfoldConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                                           bias=bias, padding_mode=padding_mode)

        # Registering buffers to be saved when calling torch.save
        self.register_buffer('tracked_n', torch.zeros(1))
        self.register_buffer('max_mean', torch.zeros(1))
        self.register_buffer('min_mean', torch.zeros(1))

        # Even if in training mode, the user can disable gathering the tensor min-max values
        self._disable_min_max_update = False

        # If user set unfold to True then he is probably want the custom CUDA kernel as well
        self._unfold = False

        # Quantization variables
        self._quantize = False
        self._x_bits = 8
        self._w_bits = 8

        # Custom kernel variables
        self._is_round = None
        self._shift_opt = None
        self._bit_group = None
        self._is_stc = False

    def _reset_stats(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._reset_stats(v)
            else:
                d[k] = 0

    def reset_stats(self):
        self._reset_stats(self.stats)

    def forward(self, x):
        # Prepare activations, weights, and bias
        if self._quantize:

            # Gather statistics during training
            if self.training and not self._disable_min_max_update:
                tracked_n_old = self.tracked_n.clone()
                self.tracked_n += x.size(0)

                max_sum = x.detach().max(dim=3).values.max(dim=2).values.max(dim=1).values.sum()
                min_sum = x.detach().min(dim=3).values.min(dim=2).values.min(dim=1).values.sum()

                self.max_mean = ((self.max_mean * tracked_n_old) + max_sum) / self.tracked_n
                self.min_mean = ((self.min_mean * tracked_n_old) + min_sum) / self.tracked_n

            # These statistics are mandatory for quantization
            assert (self.max_mean != 0 or self.min_mean != 0)

            # Activations quantization
            # Only supports unsigned uniform quantization
            if torch.min(x) == 0:
                x_q, x_q_delta = self._uniform_quantization(x, self.max_mean, self._x_bits)
                x_q = x_q.int().float()         # Just in case
                assert (x_q.max() <= ((2 ** self._x_bits) - 1) and x_q.min() >= 0)
            else:
                cfg.LOG.write('Error: not supporting signed activation quantization')
                raise NotImplementedError

            # Weights quantization
            weight_q, weight_q_delta = \
                self._uniform_symmetric_quantization_per_filter(self.weight,
                                                                self.weight.data.min(dim=3)[0].min(dim=2)[0].min(dim=1)[0],
                                                                self.weight.data.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0],
                                                                self._w_bits)

            weight_q = weight_q.int().float()   # Just in case
            assert (weight_q.max() <= ((2 ** self._w_bits) / 2 - 1) and weight_q.min() >= (-2 ** self._w_bits) / 2)

            # Bias quantization
            if self.bias is None:
                bias_fp = None
            else:
                bias_q, bias_q_delta = self._uniform_symmetric_quantization(self.bias,
                                                                            torch.min(self.bias.data),
                                                                            torch.max(self.bias.data), self._w_bits)

                assert (bias_q.max() <= ((2 ** self._w_bits) / 2 - 1) and bias_q.min() >= (-2 ** self._w_bits) / 2)

                bias_fp = bias_q * bias_q_delta

        else:
            # The single scalar movement to CUDA may be bad for performance
            x_q, x_q_delta = x, torch.Tensor([1]).cuda()
            weight_q, weight_q_delta = self.weight, torch.Tensor([1]).cuda()
            bias_fp = self.bias

        if not self._unfold:
            out = nn.functional.conv2d(x_q * x_q_delta,
                                       weight_q * weight_q_delta[:, None, None, None].expand_as(weight_q),
                                       bias=bias_fp,
                                       stride=(self.stride[0], self.stride[1]),
                                       padding=(self.padding[0], self.padding[1]), groups=self.groups)
        else:
            # At the moment, unfold and quantization must go together
            assert (self._quantize is True)
            assert (self._is_round is not None)
            assert (self._shift_opt is not None)
            assert (self._bit_group is not None)
            assert (self._is_stc is not None)

            # Im2col
            x_unf = nn.functional.unfold(x_q,
                                         kernel_size=(self.kernel_size[0], self.kernel_size[1]),
                                         padding=(self.padding[0], self.padding[1]),
                                         stride=(self.stride[0], self.stride[1])).transpose(1, 2)
            w_unf = weight_q.view(self.weight.size(0), -1).t()

            ofmap_height = \
                int((x.size(2) + 2 * self.padding[0] - self.kernel_size[0] + self.stride[0]) / self.stride[0])
            ofmap_width = \
                int((x.size(3) + 2 * self.padding[1] - self.kernel_size[1] + self.stride[1]) / self.stride[1])

            # C-W-H ordering
            x_unf_r = x_unf.reshape(x_unf.size(0) * x_unf.size(1), x_unf.size(2))
            _x_unf = x_unf_r.reshape((x_unf_r.size(0),
                                      int(x_unf_r.size(1) / (self.weight.size(2) * self.weight.size(3))),
                                      self.weight.size(2) * self.weight.size(3)))
            _x_unf = _x_unf.permute((0, 2, 1))
            _x_unf = _x_unf.reshape_as(x_unf_r)

            _w_unf = w_unf.t().reshape((self.weight.size(0),
                                        int(w_unf.size(0) / (self.weight.size(2) * self.weight.size(3))),
                                        self.weight.size(2) * self.weight.size(3)))
            _w_unf = _w_unf.permute(0, 2, 1)
            _w_unf = _w_unf.reshape_as(w_unf.t())

            # Custom CUDA kernel
            data_tensor = cu_gemm_2x48.run(_x_unf.contiguous(), _w_unf.contiguous(),
                                           self._is_round, self._shift_opt, self._bit_group, self._is_stc)

            data_tensor = data_tensor[0]
            data_tensor = data_tensor.reshape(x_unf.size(0),
                                              int(data_tensor.size(0) / x_unf.size(0)), data_tensor.size(1))

            out_unf = data_tensor.transpose(1, 2)
            out = nn.functional.fold(out_unf, (ofmap_height, ofmap_width), (1, 1))
            out = out * x_q_delta * weight_q_delta[None, :, None, None].expand_as(out) + (0 if bias_fp is None else bias_fp[None, :, None, None])

        return out

    def get_status_arr(self):
        key, val = [], []

        key.extend(['quant', 'x_b', 'w_b'])
        val.extend([self._quantize, self._x_bits, self._w_bits])

        key.append('unfold')
        val.append(self._unfold)

        key.extend(['is_round', 'shift_opt', 'bit_group', 'stc'])
        if self._unfold:
            val.extend([self._is_round, self._shift_opt, self._bit_group, self._is_stc])
        else:
            val.extend(['-', '-', '-', '-'])

        return key, val

    @staticmethod
    def _uniform_quantization(x, x_max, bits):
        N = 2 ** bits
        delta = x_max / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, 0, N - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization_per_filter(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = torch.where(x_min.abs() > x_max.abs(), x_min.abs(), x_max.abs()) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta[:, None, None, None].expand_as(x))
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta

    @staticmethod
    def _uniform_symmetric_quantization(x, x_min, x_max, bits):
        N = 2 ** bits
        delta = max(abs(x_min), abs(x_max)) * 2 / (N - 1)
        x_int = RoundSTE.apply(x / delta)
        x_q = torch.clamp(x_int, -N / 2, N / 2 - 1)
        return x_q, delta
