from __future__ import print_function, division

from argparse import ArgumentParser
from misc_functions import str2bool

def add_default_arguments(parser: ArgumentParser):
    parser.add_argument("--asr-model-file", type=str, default=None)
    parser.add_argument("--asr-batch-max-seconds", type=int, default=200)
    parser.add_argument("--asr-accumulation-steps", type=int, default=4)
    parser.add_argument("--asr-use-pretrained-encoder", type=str2bool, default=False)
    parser.add_argument("--asr-use-seq2seq-decoder", type=str2bool, default=False)
    parser.add_argument("--asr-use-syllabe-gpt", type=str2bool, default=False)
    parser.add_argument("--asr-use-char-vocab", type=str2bool, default=False)
    parser.add_argument("--asr-num-train-epochs", type=int, default=40)

    parser.add_argument("--peak-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--stop-grad-norm", type=float, default=1.0)
    parser.add_argument("--stop-loss", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--num-iters", type=int, default=200000)
    parser.add_argument("--warmup-iter", type=int, default=2000)

def py_trainer_args(list_args=None, silent=False):
    parser = ArgumentParser(description="PyTorch Trainer")
    add_default_arguments(parser)
    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()
    args.world_size = args.gpus
    if not silent:
        print('Call arguments:\n', args)
    return args
