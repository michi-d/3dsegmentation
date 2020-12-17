
__author__ = ['Michael Drews']

import torch
import numpy as np


def to_numpy(tensor):
    """Converts torch tensor into numpy array
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        tensor: input tensor

    Returns:
        tensor: as numpy array
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    """Converts numpy array into torch tensor
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        ndarray: input numpy array

    Returns:
        ndarray: as torch tesnor
    """
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray).float()
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def count_trainable_parameters(model):
    """
    Counts trainable parameters in a model.
    Args:
        model: PyTorch Model

    Returns:
        params: number of trainable parameters.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Hyperparameters():
    """
    Custom data structure for hyperparameters
    """
    def __init__(self, **kwargs):
        self._keys = []
        for i, (name, value) in enumerate(kwargs.items()):
            setattr(self, name, value)
            self._keys.append(name)

    def items(self):
        for k in self._keys:
            yield (k, getattr(self, k))

    def as_dict(self):
        return {k: v for (k, v) in self.items()}

