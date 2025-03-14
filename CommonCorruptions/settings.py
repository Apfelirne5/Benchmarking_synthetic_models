import argparse


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    # Main parameters
    arg_parser.add_argument("--dataset", type=str,default="3DCC")
    arg_parser.add_argument(
        "--model",
        type=str,
        default='FakeIt',
    )
    
    args = arg_parser.parse_args()

    return args


class Settings:
    """
        Configuration for the project.
    """

    def __init__(self, args):
        # args, e.g. the output of settings.parse_arguments()
        self.args = args
