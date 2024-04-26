import os
import argparse


def readable_dir(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError('"{}" is not an existing directory'.format(raw_path))
    if os.access(raw_path, os.R_OK):
        return os.path.abspath(raw_path)
    else:
        raise argparse.ArgumentTypeError('"{}" is not a readable directory'.format(raw_path))


def writeable_dir(raw_path):
    if not os.path.isdir(raw_path):
        try:
            os.mkdir(raw_path)
        except FileExistsError:
            raise argparse.ArgumentTypeError('"{}" can not create a directory'.format(raw_path))

    if os.access(raw_path, os.W_OK):
        return os.path.abspath(raw_path)
    else:
        raise argparse.ArgumentTypeError('"{}" is not a writeable directory'.format(raw_path))


def entry_args():
    parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
    parser.add_argument('input', type=readable_dir, nargs='?', default='/data',
                        help='Input directory containing the dataset')
    parser.add_argument('output', type=writeable_dir, nargs='?', default='/data/output',
                        help='Output directory to store the output')
    parser.add_argument('-t', '--temp', type=writeable_dir, nargs='?', default='/tmpdir',
                        help='A folder to store temporary files')
    parser.add_argument('-r', '--recall', type=int, nargs='?',
                        help='Recall value for the algorithm')
    parser.add_argument('-e', '--epochs', type=int, nargs='?',
                        help='Number of epochs for the algorithm')
    return parser
