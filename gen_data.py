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

    parser.add_argument("--NF_size", type=int, default=45)
    parser.add_argument("--L_xy", type=int, default=128)
    parser.add_argument("--L_z", type=int, default=128)
    parser.add_argument("--cube_kernel_size", type=int, default=2)
    parser.add_argument("--simplify_factor", type=float, default=1.25)
    parser.add_argument("--baseline_intensity", type=float, default=50)
    parser.add_argument("--v_std", type=float, default=1.0)
    parser.add_argument("--std_intensity_range_low", type=float, default=1.0)
    parser.add_argument("--std_intensity_range_high", type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    L = args.L
    random_state = args.random_state
    save_path = Path(args.directory) / Path(args.filename).with_suffix('.h5')

    NF_size = args.NF_size
    L_xy = args.L_xy
    L_z = args.L_z
    cube_kernel_size = args.cube_kernel_size
    simplify_factor = args.simplify_factor
    baseline_intensity = args.baseline_intensity
    v_std = args.v_std
    std_intensity_range_low = args.std_intensity_range_low
    std_intensity_range_high = args.std_intensity_range_high

    print(f'Generating dataset {save_path}...')

    # create dummy dataset and save to disk
    data = Fake3DDataset.generate_to_file(filename=save_path, L=L, size=(L_xy, L_xy, L_z),
                                          random_state=random_state, NF_size=NF_size,
                                          cube_kernel_size=cube_kernel_size,
                                          simplify_factor=simplify_factor,
                                          baseline_intensity=baseline_intensity,
                                          v_std=v_std,
                                          std_intensity_range=(std_intensity_range_low, std_intensity_range_high)
                                          )


if __name__ == '__main__':

    main()
