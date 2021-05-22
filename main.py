import argparse
import os
import numpy as np
import torch
import sys
import Config as cfg
from NeuralNet import NeuralNet
from Datasets import Datasets


parser = argparse.ArgumentParser(description='Gil Shomron, gilsho@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = list(cfg.MODELS.keys())

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=True,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('--action', choices=['QUANTIZE', 'INFERENCE'], required=True,
                    help='QUANTIZE: symmetric min-max uniform quantization\n'
                         'INFERENCE: either regular inference or hardware simulated inference')
parser.add_argument('--desc',
                    help='additional string to the test')
parser.add_argument('--chkp', default=None, metavar='PATH',
                    help='model checkpoint')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--x_bits', default=None, type=int, metavar='N',
                    help='activations quantization bits')
parser.add_argument('--w_bits', default=None, type=int, metavar='N',
                    help='weights quantization bits')
parser.add_argument('--skip_bn_recal', action='store_true',
                    help='skip BatchNorm recalibration (relevant only to the INFERENCE action)')
parser.add_argument('--eval', action='store_true',
                    help='evaluate our method')
parser.add_argument('--round_mode', choices=['ROUND', 'RAW'], default='ROUND',
                    help='rounding (i.e., nearbyint, default) or raw')
parser.add_argument('--shift_opt', choices=[5, 3, 2], default=5, type=int,
                    help='choose the number of window placement options (default: 5)')
parser.add_argument('--bit_group', choices=[4, 3, 2], default=4, type=int,
                    help='window size (default: 4)')
parser.add_argument('--stc', action='store_true',
                    help='correlate the weights to the activation preprocessing')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('-v', '--verbosity', default=0, type=int,
                    help='verbosity level (0,1,2) (default: 0)')


def quantize_network(arch, dataset, train_gen, test_gen, model_chkp=None,
                     x_bits=8, w_bits=8, desc=None):
    # Initialize log file
    name_str = '{}-{}_quantize_x-{}_w-{}'.format(arch, dataset, x_bits, w_bits)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_seed-{}-{}'.format(cfg.SEED_TORCH, cfg.SEED_NP)

    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}, x_bits={}, w_bits={}'.format(desc, x_bits, w_bits))

    # Initialize model
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    # Set configurations
    nn.model.set_quantize(True)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_min_max_update(True)
    nn.model.set_unfold(False)

    nn.best_top1_acc = 0
    nn.next_train_epoch = 0

    # Learning rate is set to zero
    nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)

    cfg.LOG.close_log()
    return


def inference(arch, dataset, train_gen, test_gen, model_chkp, x_bits=None, w_bits=None,
              unfold=True, is_round=None, shift_opt=None, bit_group=None, stc=False,
              skip_bn_recal=False, desc=None):

    # Get test string ready
    name_str = '{}-{}_inference'.format(arch, dataset)
    name_str = name_str + '_x-{}_w-{}'.format(x_bits, w_bits) if x_bits is not None and w_bits is not None else name_str
    name_str = name_str + '_unfold' if unfold else name_str
    name_str = name_str + '_round' if unfold and is_round else name_str + '_raw'
    name_str = name_str + '_sOpt-{}'.format(shift_opt) if unfold else name_str
    name_str = name_str + '_bGrp-{}'.format(bit_group) if unfold else name_str
    name_str = name_str + '_stc' if unfold and stc else name_str
    name_str = name_str + '_noBN' if skip_bn_recal else name_str
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_seed-{}-{}'.format(cfg.SEED_TORCH, cfg.SEED_NP)

    # Start log
    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}, x_bits={}, w_bits={}, unfold={}, is_round={}, stc={}, skip_bn_recal={}'
                  .format(desc, x_bits, w_bits, unfold, is_round, stc, skip_bn_recal))

    # Init neural net
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    # Configuration
    nn.model.set_quantize(x_bits is not None and w_bits is not None)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_unfold(unfold)
    nn.model.set_round(is_round)
    nn.model.set_shift_opt(shift_opt)
    nn.model.set_bit_group(bit_group)
    nn.model.set_stc(stc)
    nn.model.set_min_max_update(False)

    # Print configuration
    cfg.LOG.write_title('Configurations')
    nn.model.print_config()

    # Start test!
    cfg.LOG.write_title('Start Test')

    if skip_bn_recal is None or skip_bn_recal is False:
        cfg.LOG.write('Conducting BN recalibration')
        nn.next_train_epoch = 0
        nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)
    else:
        cfg.LOG.write('Skipping BN recalibration')
        nn.test(test_gen)

    return


def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    arch = args.arch.split('_')[0]
    dataset = args.arch.split('_')[1]

    cfg.BATCH_SIZE = args.batch_size
    cfg.VERBOSITY = args.verbosity
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.INCEPTION = (arch == 'inception')

    dataset_ = Datasets.get(dataset)

    # Deterministic random numbers
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED_TORCH)
    torch.cuda.manual_seed(cfg.SEED_TORCH)
    np.random.seed(cfg.SEED_NP)

    test_gen, _ = dataset_.testset(batch_size=args.batch_size)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=args.batch_size, max_samples=None)

    model_chkp = None if args.chkp is None else cfg.RESULTS_DIR + '/' + args.chkp

    if args.action == 'QUANTIZE':
        quantize_network(arch, dataset, train_gen, test_gen,
                         model_chkp=model_chkp,
                         x_bits=args.x_bits, w_bits=args.w_bits, desc=args.desc)

    elif args.action == 'INFERENCE':
        inference(arch, dataset, train_gen, test_gen,
                  model_chkp=model_chkp,
                  x_bits=args.x_bits, w_bits=args.w_bits,
                  unfold=args.eval, is_round=(args.round_mode == 'ROUND'),
                  shift_opt=args.shift_opt, bit_group=args.bit_group, stc=args.stc,
                  skip_bn_recal=args.skip_bn_recal, desc=args.desc)

    return


if __name__ == '__main__':
    main()
