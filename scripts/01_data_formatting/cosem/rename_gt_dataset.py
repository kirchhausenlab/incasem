import configargparse as argparse
import os

def to_lowercase(base_dir):
    base_dir = os.path.join(base_dir, '0003')
    os.chdir(base_dir)
    for file_or_folder in os.listdir():
        print(file_or_folder)
        os.rename(file_or_folder, file_or_folder.lower())


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add(
        '-d',
        '--dir',
        help='We transform to lowercase all files and folders in this folder')
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    to_lowercase(args.dir)


if __name__ == '__main__':
        main()
