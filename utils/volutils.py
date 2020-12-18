
__author__ = ['Michael Drews']

import numpy as np
import math


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


def rotation_matrix(axis, theta):
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
