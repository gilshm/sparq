import torch
import torch.nn as nn
from tabulate import tabulate
from QuantConv2d import UnfoldConv2d
import Config as cfg


class SimModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold_list = []   # All UnfoldConv2d layers

    def forward(self, x):
        raise NotImplementedError

    def update_unfold_list(self):
        self.apply(self._apply_unfold_list)

    def _apply_unfold_list(self, m):
        if type(m) == UnfoldConv2d:
            self.unfold_list.append(m)

    def set_unfold(self, v):
        for l in self.unfold_list:
            l._unfold = v

    def set_quantize(self, v):
        for l in self.unfold_list:
            l._quantize = v

    def set_quantization_bits(self, x_bits, w_bits):
        for l in self.unfold_list:
            l._x_bits = x_bits
            l._w_bits = w_bits

    def set_min_max_update(self, v):
        for l in self.unfold_list:
            l._disable_min_max_update = not v

    def set_round(self, v):
        for l in self.unfold_list:
            l._is_round = v

    def set_shift_opt(self, v):
        for l in self.unfold_list:
            l._shift_opt = v

    def set_bit_group(self, v):
        for l in self.unfold_list:
            l._bit_group = v

    def set_stc(self, v):
        for l in self.unfold_list:
            l._is_stc = v

    def print_config(self):
        headers = None
        table = []

        for l in self.unfold_list:
            headers, config = l.get_status_arr()
            table.append(config)

        headers.insert(0, '#')
        cfg.LOG.write(tabulate(table, headers=headers, showindex=True), date=False)
