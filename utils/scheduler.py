
__author__ = ['Michael Drews']

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import inspect


class ReduceSigmaOnPlateau(ReduceLROnPlateau):
    """
    Scheduler for Adaptive Loss in Keypoint Detection
    """
    def __init__(self, train_dataset, valid_dataset, min_sigma=0, **kwargs):
        super().__init__(**kwargs)

        # Attach dataset
        self._check_dataset_type(train_dataset)
        self._check_dataset_type(valid_dataset)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.min_sigma = min_sigma

    @staticmethod
    def _check_dataset_type(dataset):
        base_classes = set(inspect.getmro(type(dataset)))
        if Dataset in base_classes:
            if dataset.objective == 'keypoint_detection':
                return True
            else:
                raise TypeError('{} is not suited for Keypoint Detection'.format(
                    type(dataset).__name__))
        else:
            raise TypeError('{} is not a Dataset'.format(
                type(dataset).__name__))

    def _reduce_lr(self, epoch):
        self._reduce_sigma(epoch)

    def _reduce_sigma(self, epoch):
        """Reduces standard deviation of the points in the heatmap.
        Adapted from PyTorch's ReduceLROnPlateau
        """
        assert self.train_dataset.sigma == self.valid_dataset.sigma
        old_sigma = self.train_dataset.sigma
        new_sigma = max(old_sigma * self.factor, self.min_sigma)
        if old_sigma - new_sigma > self.eps:
            self.train_dataset.sigma = new_sigma
            self.valid_dataset.sigma = new_sigma
            if self.verbose:
                print('Epoch {:5d}: reducing sigma'
                      ' to {:.4e}.'.format(epoch, new_sigma))