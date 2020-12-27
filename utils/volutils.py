
__author__ = ['Michael Drews']

import numpy as np
import math
from scipy.linalg import norm
import torch


def cart2spherical(xyz):
    """
    Transforms points to spherical coordinates.

    Args:
        xyz: Cartesian input coordinates

    Returns:
        rtp: Spherical output coordinates (r, theta, phi)
    """
    if np.issubdtype(xyz.dtype, np.integer):
        xyz = xyz.astype(np.float)
    rtp = np.zeros_like(xyz)

    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rtp[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    rtp[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # rtp[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    rtp[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rtp


def spherical2cart(rtp):
    """
    Transforms points to cartesian coordinates.

    Args:
        rtp: Spherical input coordinates (r, theta, phi)

    Returns:
        xyz: Cartesian output coordinates
    """

    xyz = np.zeros_like(rtp)
    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    return xyz


def rodrigues_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_spherical_unit_vectors(rtp):
    """
    Calculates the spherical vectors for a set spherical input coordinates.
    Args:
        coords: Spherical input coordinates

    Returns:
        e_r, e_theta, e_phi: Spherical unit vectors for each point
    """
    e_r = (np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])).reshape(-1, 1) * np.array([[1, 0, 0]]) + \
          (np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])).reshape(-1, 1) * np.array([[0, 1, 0]]) + \
          (np.cos(rtp[:, 1])).reshape(-1, 1) * np.array([[0, 0, 1]])

    e_theta = (np.cos(rtp[:, 1]) * np.cos(rtp[:, 2])).reshape(-1, 1) * np.array([[1, 0, 0]]) + \
              (np.cos(rtp[:, 1]) * np.sin(rtp[:, 2])).reshape(-1, 1) * np.array([[0, 1, 0]]) - \
              (np.sin(rtp[:, 1])).reshape(-1, 1) * np.array([[0, 0, 1]])

    e_phi = (-np.sin(rtp[:, 2])).reshape(-1, 1) * np.array([[1, 0, 0]]) + \
            (np.cos(rtp[:, 2])).reshape(-1, 1) * np.array([[0, 1, 0]])

    return e_r, e_theta, e_phi


def scale_vectors(*arrays, scale_xyz=(1.,1.,1.)):
    """
    Scales 3D vectors independently for each dimension
    Args:
        *arrays: multiple input vector arrays
        scale_xyz: scale factors

    Returns:
        out: list of scaled output arrays
    """
    out = []
    for i, v in enumerate(arrays):
        v = np.asarray(v)
        if v.ndim == 1:
            assert len(v) % 3 == 0
            v = v.reshape((-1,3))
        assert v.shape[1] == 3

        v = v * np.array(scale_xyz)
        out.append(v)
    return out


def rotate_vectors(*arrays, R):
    """
    Rotates 3D vectors
    Args:
        *arrays: multiple input vector arrays
        R: rotation matrix

    Returns:
        out: list of rotated output arrays
    """
    out = []
    for i, v in enumerate(arrays):
        v = np.asarray(v)
        if v.ndim == 1:
            assert len(v) % 3 == 0
            v = v.reshape((-1,3))
        assert v.shape[1] == 3

        v = np.dot(v, R)
        out.append(v)
    return out


def clip_points(*arrays, clip_mask):
    """
    Clip several arrays using the same clipping mask
    Args:
        *arrays: multiple input vector arrays
        clip_mask: boolean mask

    Returns:
        out: list of rotated output arrays
    """
    out = []
    for i, v in enumerate(arrays):
        v = np.asarray(v)
        if v.ndim == 1:
            assert len(v) % 3 == 0
            v = v.reshape((-1,3))
        assert v.shape[1] == 3

        v = v[~clip_mask, :]
        out.append(v)
    return out


def bezier(t, p):
    """Generates bezier curve between point p.

    Args:
        t: moves along curve (0..1)
        p: list of defining points
        order: order of the bezier curve

    Returns:
        Coordinates along Bezier curve as function of t
    """
    order = len(p) - 1
    if order == 1:
        return (1-t)*p[0] + t*p[1]
    elif order == 2:
        return p[1] + ((1-t)**2)*(p[0] - p[1]) + (t**2)*(p[2]-p[1])
    elif order == 3:
        return (1-t)*bezier(t, [p[0], p[1], p[2]]) + \
                t*bezier(t, [p[1], p[2], p[3]])


def unique_vectors(vectors, tol=1e-6):
    """
    Deletes duplicate vectors from an array, within a given tolerance.

    Args:
        vectors: (Nxd) array (N d-dimensional points)
        tol: tolerances

    Returns:
        clean_vectors: cleaned vectors
        remove_indices: indexes removed
    """
    distances = norm(vectors.reshape(1, -1, 3) - vectors.reshape(-1, 1, 3), ord=2, axis=2)
    remove_indices = list()
    for (i, d) in enumerate(distances):
        equal = np.nonzero(distances[i, :] < tol)[0]
        equal = equal[equal > i]
        remove_indices += list(equal)
    remove_indices = np.array(remove_indices).astype(np.int)
    clean_vectors = np.delete(vectors, remove_indices, axis=0)
    return clean_vectors, remove_indices


def get_sub_volume(image, label,
                   orig_x=128, orig_y=128, orig_z=128,
                   output_x=64, output_y=64, output_z=64,
                   max_tries=1000, background_threshold=0.95,
                   start_xyz=None):
    """
    Extract (random) sub-volumes from original images.

    Args:
        image (np.array): original image,
            of shape (1, orig_x, orig_y, orig_z)
        label (np.array): original label.
            labels coded using discrete values rather than
            a separate dimension,
            so this is of shape (1, orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction
            of the sample which can be the background
        start_xyz (tuple): start point if not random

    returns:
        X (np.array): sample of original image of dimension
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension
            (num_classes, output_x, output_y, output_z)
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    if start_xyz: # if start position is given (not random)
        start_x = start_xyz[0]
        start_y = start_xyz[1]
        start_z = start_xyz[2]
        y = label[:, start_x: start_x + output_x, start_y: start_y + output_y, start_z: start_z + output_z]
        X = np.copy(image[:, start_x: start_x + output_x, start_y: start_y + output_y, start_z: start_z + output_z])
        return X, y

    tries = 0
    while tries < max_tries:
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x + 1)
        start_y = np.random.randint(0, orig_y - output_y + 1)
        start_z = np.random.randint(0, orig_z - output_z + 1)

        # extract relevant area of label
        y = label[:,
            start_x: start_x + output_x,
            start_y: start_y + output_y,
            start_z: start_z + output_z
            ]

        # compute the background ratio
        bgrd_ratio = (y == 0).sum() / (output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:
            # make copy of the sub-volume
            X = np.copy(image[:,
                        start_x: start_x + output_x,
                        start_y: start_y + output_y,
                        start_z: start_z + output_z
                        ])

            return X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")
    return None, None


def as_vol_tensor(vol_data):
    """
    Casts input array to a float-precision torch Tensor with shape (1,Nx,Ny,Nz)
    """
    if (vol_data.ndim == 4) and (vol_data.shape[0] == 1):
        # if first dimension is singleton dimension with "channels"
        vol_data = torch.Tensor(vol_data).float()
        return vol_data
    elif (vol_data.ndim == 3):
        vol_data = torch.Tensor(vol_data).float().unsqueeze(0)
        return vol_data


def percentile_normalization(image_data, percentiles=(1,99)):
    """Normalize pixel values within a given percentile range.
    Works for torch.Tensors

    Args:
        image_data: input image data
        percentiles: tuple defining the percentiles

    Returns:
        clipped image data
    """
    low = np.percentile(image_data.flatten(), percentiles[0])
    high = np.percentile(image_data.flatten(), percentiles[1])

    image_data = (image_data-low) / (high-low)
    return torch.clip(image_data, 0, 1)


def predict_whole_volume(model, vol_data, sub_x=64, sub_y=64, sub_z=64, stride=64, normalize=False):
    """
    Predict on all sub-volumes within a larger volume using the model.

    Args:
        model: Segmentation model
        vol_data: (1, Nx, Ny, Nz) array containing the volume data
        sub_x: size of sub-volume in x
        sub_y: size of sub-volume in y
        sub_z: size of sub-volume in z
        stride: stride for the sliding 3D window
        normalize: Whether to normalize the data or not.

    Returns:
        vol_pred: Predicted segmentation labels for vol_data.
    """
    # re-format data
    vol_data = as_vol_tensor(vol_data)

    if normalize:
        vol_data = percentile_normalization(torch.Tensor(vol_data))

    # sliding volume loop
    all_pred = []
    for x in range(0, vol_data.shape[1] - sub_x + 1, stride):
        for y in range(0, vol_data.shape[2] - sub_y + 1, stride):
            for z in range(0, vol_data.shape[3] - sub_z + 1, stride):
                start_xyz = (x, y, z)

                # pre-allocate volume with (-1)
                prediction = np.ones_like(vol_data) * (-1)

                # get sub-volume
                subvol_X, _ = \
                    get_sub_volume(vol_data, np.ones_like(vol_data),
                                   orig_x=vol_data.shape[1], orig_y=vol_data.shape[2],
                                   orig_z=vol_data.shape[3],
                                   output_x=sub_x, output_y=sub_y, output_z=sub_y, background_threshold=100.0,
                                   start_xyz=start_xyz
                                   )

                # run model on sub-volume
                with torch.no_grad():
                    subvol_y_pred = model.forward(torch.Tensor(subvol_X).unsqueeze(0))
                    subvol_y_pred = torch.sigmoid(subvol_y_pred)
                    subvol_y_pred = subvol_y_pred.squeeze().cpu().numpy()

                # save prediction
                prediction[0, x:x + sub_x, y:y + sub_y, z:z + sub_z] = subvol_y_pred
                all_pred.append(prediction)

    all_pred = np.stack(all_pred, 0)

    # merge sub-predictions
    # count how often one voxel was classified as 0
    neg_votes = ((0 < all_pred) & (all_pred <= 0.5)).sum(axis=0)
    # count how often one voxel was classified as 1
    pos_votes = ((0 < all_pred) & (all_pred > 0.5)).sum(axis=0)
    # take the majority vote, in favor of 0 in case of a tie
    vol_pred = ((neg_votes < pos_votes)).astype(np.int)

    return vol_pred