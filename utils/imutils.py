
__author__ = ['Michael Drews']

import numpy as np
from .misc import *


def im_to_numpy(img):
    """Converts image to numpy array with dimension (height x width x channels)
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        img: input image

    Returns:
        img: output numpy array
    """
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img


def im_to_torch(img):
    """Converts image to torch tensor with dimension (channels x height x width)
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        img: input image

    Returns:
        img: output torch tensor
    """
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def gaussian(shape=(7, 7), sigma=1):
    """
    2D gaussian mask
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        shape: size of the kernel
        sigma: std of the gaussian

    Returns:
        out: gaussian kernel as torch tensor
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    out = to_torch(h).float()
    return out


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    """Draws a localized kernel around a given point into an image tensor
    From: https://github.com/bearpaw/pytorch-pose

    Args:
        img: input image tensor
        pt: given points
        sigma: standard deviation
        type: gaussian or cauchy kernel

    Returns:
        img: modified image tensor
        True/False: whether point was in-bounds or not
    """
    img = to_numpy(img).T

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img), False

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img.T), True