
__author__ = ['Michael Drews']

import numpy as np
import math
from scipy.linalg import norm


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
    remove_indices = np.array(remove_indices)
    clean_vectors = np.delete(vectors, remove_indices, axis=0)
    return clean_vectors, remove_indices


