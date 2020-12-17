
__author__ = ['Michael Drews']

import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import sys
import tqdm
import h5py
import os

import utils.imutils as imutils
import utils.misc as misc
import seg2d.fakedata as fakedata2d
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

        self.data = []
        self.masks = []
        self.center_points = []
        self.objective = None

        self._load_data(**kwargs)

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raise NotImplementedError

    def get_sample(self, i):
        return self.data[i], self.masks[i], self.center_points[i]


class Fake2DDataset(ColumnDataset):
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
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed()

        super().__init__(**kwargs)
        self.objective = None

    def _save_to_h5(self, path):
        """
        Saves dataset to h5-file
        Args:
            path: target path
        """
        size = tuple(self.data[0].shape)
        f = h5py.File(str(path), 'w')
        f.create_dataset("data", (self.L, size[0], size[1]), dtype='float32')
        f.create_dataset("masks", (self.L, size[0], size[1]), dtype='float32')
        f.create_dataset("center_points", (self.L,), dtype=h5py.vlen_dtype(np.dtype('float32')))
        for i in range(self.L):
            f['data'][i] = self.data[i]
            f['masks'][i] = self.masks[i]
            f['center_points'][i] = self.center_points[i].flatten()
        f.close()

    def _load_from_h5(self, path):
        f = h5py.File(str(path), 'r')
        self.data = f['data'][:]
        self.masks = f['masks'][:]
        self.center_points = [v.reshape((-1,2)) for v in f['center_points'][:]]

    def _load_data(self, **kwargs):
        if self.h5path:
            if os.path.isfile(self.h5path):
                print(f'Loading dataset from {self.h5path}')
                self._load_from_h5(self.h5path)
            else:
                self._gen_all(**kwargs)
                print(f'Saving dataset to {self.h5path}')
                self._save_to_h5(self.h5path)
        else:
            self._gen_all(**kwargs)

    def _gen_all(self, **kwargs):
        """
        Generates the whole dataset.
        """
        print('Pre-rendering dataset...')
        for _ in tqdm.tqdm(range(self.L), desc='Progress', file=sys.stdout):
            image, labels, points = self._gen_sample(**kwargs)
            self.data.append(image)
            self.masks.append(labels)
            self.center_points.append(points)

    def _gen_sample(self, **kwargs):
        """
        Generate one picture (with fraction of labeled pixels above the given threshold)
        """
        while True:
            img = fakedata2d.RandomImage(**kwargs)
            image_data = _percentile_normalization(img.data)

            image = image_data
            labels = img.labels
            points = img.midpoints
            if (labels.mean() >= self.thr) and (len(points) < 128):
                return image, labels, points


class SegmentationFake2DDataset(Fake2DDataset):
    """
    Dataset class for 2D segmentation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective = 'segmentation'

    def __getitem__(self, i):
        X = Tensor(self.data[i][np.newaxis, :, :])
        y = Tensor(self.masks[i][np.newaxis, :, :])
        return X, y


class RegressionFake2DDataset(Fake2DDataset):
    """
    Dataset class for regression of 128 2D-points
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective = 'regression'

    def __getitem__(self, i):
        # get image
        X = Tensor(self.data[i][np.newaxis, :, :])

        # get center points
        points = self.center_points[i]
        n_points = float(len(points))

        # pad to 128 points (repeating first point)
        x0 = points[0, 0]
        y0 = points[0, 1]
        coords = np.pad(points, ((0, int(128-n_points)),
                                 (0, 0)))
        coords[int(n_points):, 0] = x0
        coords[int(n_points):, 1] = y0

        # scale coordinates to -1...1
        n = X.shape[-1]
        coords = (coords-n/2) / n

        y = Tensor(coords[np.newaxis, :, :])
        return X, y


class KeypointDetectionFake2DDataset(Fake2DDataset):
    """
    Dataset class for keypoint detection (predict heatmap).
    """

    def __init__(self, sigma=2.0, type='Gaussian', **kwargs):
        """
        Args:
            sigma: standard deviation of the point kernel
        """
        super().__init__(**kwargs)
        self.objective = 'keypoint_detection'
        self.sigma = sigma
        self.type = type

    @staticmethod
    def _pad_points(points, max_points=128, pad_value=(999,999)):
        n_points = float(len(points))
        coords = np.pad(points, ((0, int(max_points - n_points)),
                                 (0, 0)))
        coords[int(n_points):, 0] = pad_value[0]
        coords[int(n_points):, 1] = pad_value[1]
        return coords

    def __getitem__(self, i):
        data, _, center_points = self.get_sample(i)

        # generate label heat map
        y = np.zeros_like(data)
        for p in center_points:
            y, success = imutils.draw_heatmap(y, p, self.sigma, type=self.type)
            if not success:
                print(f'Warning: point{tuple(p)} is out-of-bounds.')

        # get point coordinates
        z = self._pad_points(center_points)

        X = Tensor(data).unsqueeze(0)
        y = Tensor(y).unsqueeze(0)
        z = Tensor(z)#.unsqueeze(0)
        return X, y, z