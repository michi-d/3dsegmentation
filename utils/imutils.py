
__author__ = ['Michael Drews']

import numpy as np
from .misc import *
from skimage.feature import peak_local_max


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


def draw_heatmap(img, pt, sigma, type='Gaussian'):
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


def get_points_from_heatmap(heatmap, min_distance=5, pad=True, pad_value=(999,999), pad_size=128, threshold_abs=0.75):
    """Get coordinates of maxima from a heatmap

    Args:
        img: 2-dimensional numpy-like array
        min_distance: minimum distance between points
        pad: Whether to create equal length vectors or not
        pad_value: dummy coordinates for padding
        pad_size: output length (maximum number of points)
        threshold_abs: threshold for point detection

    Returns:
        coords: (N_points x 2) array
    """
    coords = peak_local_max(heatmap, min_distance=min_distance, exclude_border=False, threshold_abs=threshold_abs)

    if len(coords) > pad_size:
        print(f'WARNING: More than {pad_size} points detected! Clipping output.')
        coords = coords[:pad_size]

    if pad:
        n_peaks = len(coords)
        n_pad = int(pad_size - n_peaks)
        coords = np.pad(coords, ((0, n_pad),
                                 (0, 0)))
        coords[n_peaks:, 0] = pad_value[0]
        coords[n_peaks:, 1] = pad_value[1]



    return coords


def batch_predict_keypoints(y, max_points=128, threshold_abs=0.75):
    """Get keypoint coordinates for a batch of heatmaps.

    Args:
        y: Tensor (batch_size x 1 x N_x x N_y)
        max_points: maximum number of points
        threshold_abs: threshold for point detection

    Returns:
        batch_coordinates: List of Tensors (1 x N_points x 2)
    """
    y_ = y.detach().cpu().numpy()
    batch_size = y_.shape[0]

    batch_coordinates = []
    for n in range(batch_size):
        coords = get_points_from_heatmap(y_[n,0,:], min_distance=5, pad=True,
                                         pad_size=max_points, threshold_abs=threshold_abs)
        batch_coordinates.append(torch.Tensor(coords).unsqueeze(0))

    return torch.cat(batch_coordinates, 0)
