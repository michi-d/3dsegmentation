
__author__ = ['Michael Drews']

import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import sys
import tqdm

import seg2d.fakedata as fakedata2d


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

    def __init__(self, **kwargs):
        super().__init__()

        self.data = []
        self.masks = []
        self.center_points = []

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

    def __init__(self, L=512, thr=0.3, seed=None, **kwargs):
        """
        Dataset object for 2D fake data.

        Args:
            L: Number of images to be generated
            thr: Dataset contains only images with percentage of labeled pixels above this threshold.
            seed: Random number generator seed
        """
        self.L = L
        self.thr = thr
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed()

        super().__init__(**kwargs)

    def _load_data(self, **kwargs):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, i):
        X = Tensor(self.data[i][np.newaxis, :, :])
        y = Tensor(self.masks[i][np.newaxis, :, :])
        return X, y


class RegressionFake2DDataset(Fake2DDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        n = X.shape[0]
        coords = (coords-n/2) / n

        y = Tensor(coords[np.newaxis, :, :])
        return X, y