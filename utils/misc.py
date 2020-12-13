
__author__ = ['Michael Drews']

import torch


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
