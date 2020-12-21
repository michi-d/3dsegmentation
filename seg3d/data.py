
__author__ = ['Michael Drews']

import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import sys
import tqdm
import h5py
import os
import time

import seg3d.fakedata as fakedata
import utils.volutils as volutils
import utils.misc as misc
from pathlib import Path


def _percentile_normalization(image_data, percentiles=(1,99)):
    """Normalize pixel values within a given percentile range

    Args:
        image_data: input image data
        percentiles: tuple defining the percentiles

    Returns:
        clipped image data
    """
    low = np.percentile(image_data.flatten(), percentiles[0])
    high = np.percentile(image_data.flatten(), percentiles[1])


    image_data = (image_data-low) / (high-low)
    return np.clip(image_data, 0, 1)


class ColumnDataset(Dataset):
    """
    Basic dataset class for column detection.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.vol_data = []
        self.vol_labels = []
        self.points = []
        self.e_r = []
        self.e_theta = []
        self.e_phi = []

        self._load_data(**kwargs)

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raise NotImplementedError

    def get_sample(self, i):
        return self.vol_data[i], self.vol_labels[i], self.points[i], \
               self.e_r[i], self.e_theta[i], self.e_phi[i]


class Fake3DDataset(ColumnDataset):
    """
    Basic dataset class for the 2D fake data.
    """

    def __init__(self, L=512, thr=0.3, seed=None, h5path=None, **kwargs):
        """
        Generator function.

        Args:
            L: number of images to be generated
            thr: generate only images with percentage of labeled pixels above this threshold
            seed: random number generator seed
        """
        self.L = int(L)
        self.thr = thr
        self.h5path = h5path
        self.seed = seed
        if seed:
            np.random.seed(self.seed)
        else:
            np.random.seed()

        super().__init__(**kwargs)
        self.objective = None


    def _create_h5(self, path):
        """
        Creates h5 file
        Args:
            path: target path
        """
        size = tuple(self.vol_data[0].shape)
        f = h5py.File(str(path), 'w')
        with h5py.File(str(path), 'a') as hf:
            hf.create_dataset("vol_data", (self.L, size[0], size[1], size[2]), dtype='float16',
                              data=np.zeros((self.L, size[0], size[1], size[2]), dtype=np.float16))
            hf.create_dataset("vol_labels", (self.L, size[0], size[1], size[2]), dtype='float16',
                              data=np.zeros((self.L, size[0], size[1], size[2]), dtype=np.bool))
            f.create_dataset("points", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')), maxshape=(None,))
            f.create_dataset("e_r", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')), maxshape=(None,))
            f.create_dataset("e_theta", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')), maxshape=(None,))
            f.create_dataset("e_phi", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')), maxshape=(None,))

        #f.create_dataset("vol_data", (self.L, size[0], size[1], size[2]), dtype='float16')
        #f.create_dataset("vol_labels", (self.L, size[0], size[1], size[2]), dtype='bool')
        #f.create_dataset("points", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')))
        #f.create_dataset("e_r", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')))
        #f.create_dataset("e_theta", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')))
        #f.create_dataset("e_phi", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float16')))
        #f.close()

    def _save_sample_to_h5(self, i):
        """
        Saves sample to h5-file
        Args:
            i: index
            path: target path
        """
        #f = h5py.File(str(self.h5path), 'w')
        with h5py.File(str(self.h5path), 'a') as hf:
            #for i in range(self.L):
            hf['vol_data'][i] = self.vol_data[i]
            hf['vol_labels'][i] = self.vol_labels[i]
            hf['points'][i] = self.points[i].flatten()
            hf['e_r'][i] = self.e_r[i].flatten()
            hf['e_theta'][i] = self.e_theta[i].flatten()
            hf['e_phi'][i] = self.e_phi[i].flatten()
            #f.close()

    def _load_from_h5(self, path):
        f = h5py.File(str(path), 'r')
        self.vol_data = f['vol_data'][:]
        self.vol_labels = f['vol_labels'][:]
        self.points = [v.reshape((-1,3)) for v in f['points'][:]]
        self.e_r = [v.reshape((-1, 3)) for v in f['e_r'][:]]
        self.e_theta = [v.reshape((-1, 3)) for v in f['e_theta'][:]]
        self.e_phi = [v.reshape((-1, 3)) for v in f['e_phi'][:]]
        f.close()

    def _load_data(self, **kwargs):
        if self.h5path:
            if os.path.isfile(self.h5path):
                print(f'Loading dataset from {self.h5path}')
                self._load_from_h5(self.h5path)
            else:
                #self._create_h5(self.h5path)
                self._gen_all(**kwargs, save_to_h5=True)
                #print(f'Saving dataset to {self.h5path}')
        else:
            self._gen_all(**kwargs)

    def _gen_all(self, save_to_h5=False, **kwargs):
        """
        Generates the whole dataset.
        """
        print('Pre-rendering dataset...')
        if self.seed:
            np.random.seed(self.seed)
        else:
            np.random.seed()

        i = 0
        for _ in tqdm.tqdm(range(self.L), desc='Progress', file=sys.stdout):
            vol_data, vol_labels, points, e_r, e_theta, e_phi = self._gen_sample(**kwargs)
            self.vol_data.append(vol_data)
            self.vol_labels.append(vol_labels)
            self.points.append(points)
            self.e_r.append(e_r)
            self.e_theta.append(e_theta)
            self.e_phi.append(e_phi)
            if save_to_h5:
                if not os.path.isfile(self.h5path):
                    self._create_h5(self.h5path)
                    time.sleep(1)
                print(f'Save sample {i} to h5...')
                self._save_sample_to_h5(i)
            i+=1

    def _gen_sample(self, **kwargs):
        """
        Generate one picture (with fraction of labeled pixels above the given threshold)
        """
        volume = fakedata.RandomVolume(random_geometry=True, **kwargs)
        #volume = fakedata.RandomVolume()

        vol_data = volume.vol_data
        vol_labels = volume.vol_labels
        points = volume.mid_points
        e_r = volume.geometry.e_r
        e_theta = volume.geometry.e_theta
        e_phi = volume.geometry.e_phi

        return vol_data, vol_labels, points, e_r, e_theta, e_phi