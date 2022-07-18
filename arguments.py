import argparse
from math import inf

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str,
                       help='model configuration file')
    group.add_argument('--checkpoint', type=str, default=None,
                       help='model checkpoint')                  
    group.add_argument('--load', type=str, default = None,
                       help='model loading directory')
    group.add_argument('--tokenizer', type=str,
                       help='Path to tokenizer json file')
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('--base-path', type=str, default=None,
                       help='Path to the project base directory.')
    group.add_argument('--dataset-name', type=str, default=None,
                       help='Name of the dataset')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--padding-len', type=int, default=512,
                       help='Maximum padding length.')
    group.add_argument('--save-iters', type=int, default=50,
                       help='number of iterations between saves')
    group.add_argument('--inspect-iters', type=int, default=50,
                       help='number of inspecting')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=inf,
                       help='gradient clipping')
    group.add_argument('--seed', type=int, default=123,
                       help='random seed for reproducibility')
    group.add_argument('--epochs', type=int, default=30,
                       help='total number of epochs to train over all training runs')

    # Learning rate.
    group.add_argument('--lr', type=float, default=1.0e-5,
                       help='initial learning rate')
    group.add_argument('--warmup-ratio', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)

    args = parser.parse_args()
    return args