#!/usr/bin/python

"""
Script for generating the train and validation dataset.
"""

from seg3d.data import Fake3DDataset
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", type=str, default='./vol_data_test')
    parser.add_argument("--filename", type=str, default='vol_train_set1.h5')

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--L", type=int, default=512)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    L = args.L
    random_state = args.random_state
    save_path = Path(args.directory) / Path(args.filename).with_suffix('.h5')

    print(f'Generating dataset {save_path}...')

    # create dummy dataset and save to disk
    data = Fake3DDataset.generate_to_file(filename=save_path, L=L, size=(128,128,128),
                                          random_state=random_state, NF_size=45, cube_kernel_size=2,
                                          simplify_factor=1.25)


if __name__ == '__main__':

    main()
