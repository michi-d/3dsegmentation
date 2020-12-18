
__author__ = ['Michael Drews']

import numpy as np
from math import sin, cos, acos, sqrt, pi
from math import atan2


def get_points_icosahedron():
    """Source: https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
    Get corners (coordinates) and triangles (point indices) of an icosahedron

    Returns:
        icoPoints: 3D-coordinates of the corner points
        icoTriangs: point indices for the 20 triangles forming the surface
    """

    s, c = 2 / sqrt(5), 1 / sqrt(5)
    topPoints = [(0, 0, 1)] + [(s * cos(i * 2 * pi / 5.), s * sin(i * 2 * pi / 5.), c) for i in range(5)]
    bottomPoints = [(-x, y, -z) for (x, y, z) in topPoints]
    icoPoints = topPoints + bottomPoints

    icoTriangs = [(0, i + 1, (i + 1) % 5 + 1) for i in range(5)] + \
                 [(6, i + 7, (i + 1) % 5 + 7) for i in range(5)] + \
                 [(i + 1, (i + 1) % 5 + 1, (7 - i) % 5 + 7) for i in range(5)] + \
                 [(i + 1, (7 - i) % 5 + 7, (8 - i) % 5 + 7) for i in range(5)]

    return icoPoints, icoTriangs

def barycentricCoords(p):
    """
    Source: https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
    barycentric coords for triangle (-0.5,0),(0.5,0),(0,sqrt(3)/2)
    """
    x,y = p
    # l3*sqrt(3)/2 = y
    l3 = y*2./sqrt(3.)
    # l1 + l2 + l3 = 1
    # 0.5*(l2 - l1) = x
    l2 = x + 0.5*(1 - l3)
    l1 = 1 - l2 - l3
    return l1,l2,l3

def scalProd(p1,p2):
    return sum([p1[i]*p2[i] for i in range(len(p1))])

def slerp(p0,p1,t):
    """
    Source: https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
    uniform interpolation of arc defined by p0, p1 (around origin)
    t=0 -> p0, t=1 -> p1
    """
    assert abs(scalProd(p0,p0) - scalProd(p1,p1)) < 1e-7
    ang0Cos = scalProd(p0,p1)/scalProd(p0,p0)
    ang0Sin = sqrt(1 - ang0Cos*ang0Cos)
    ang0 = atan2(ang0Sin,ang0Cos)
    l0 = sin((1-t)*ang0)
    l1 = sin(t    *ang0)
    return tuple([(l0*p0[i] + l1*p1[i])/ang0Sin for i in range(len(p0))])

def mapGridpoint2Sphere(p,s1,s2,s3):
    """
    Source: https://stackoverflow.com/questions/46777626/mathematically-producing-sphere-shaped-hexagonal-grid
    map 2D point p to spherical triangle s1,s2,s3 (3D vectors of equal length)
    """
    l1,l2,l3 = barycentricCoords(p)
    if abs(l3-1) < 1e-10: return s3
    l2s = l2/(l1+l2)
    p12 = slerp(s1,s2,l2s)
    return slerp(p12,s3,l3)

def get_hex_points_on_triangle(n=0):
    """Generates center points of a hexagonal grid on
    the triangle (-0.5,0),(0.5,0),(0,sqrt(3)/2).
    Corners of the triangle coincide with center points.

    Args:
        n: number of hexagons within the triangle

    Returns:
        c: list of center points
    """

    # define base vectors
    v1 = np.array((np.cos(np.radians(60)),
                   np.sin(np.radians(60))))
    v2 = np.array((1,
                   0))
    v1 = v1 / (n + 1)
    v2 = v2 / (n + 1)

    # create grid
    p0 = np.array((-0.5, 0))
    c = []
    for i in range(n + 2):
        for j in range(n + 2 - i):
            c.append(p0 + i * v1 + j * v2)

    return c


def get_hex_points_icosahedron(n=1):
    """Get center points of hexagon on a sphere (approximated as icosahedron).

    Args:
        n: number of hexagons to fit into one triangle of the icosahedron

    Returns:
        hex_points: list of 3D coordinates for the center points
    """

    icoPoints, icoTriangs = get_points_icosahedron()

    # create list triangle corner points
    tri_points = []
    for j in range(len(icoTriangs)):  # loop over triangles
        s1, s2, s3 = [icoPoints[i] for i in icoTriangs[j]]
        tri_points.extend([s1, s2, s3])
    tri_points = np.array(tri_points)
    tri_points = np.unique(tri_points, axis=0)

    # create list of hexagonal pattern points within each triangle
    centers = get_hex_points_on_triangle(n=n)
    hex_points = []
    for j in range(len(icoTriangs)):  # loop over triangles
        s1, s2, s3 = [icoPoints[i] for i in icoTriangs[j]]
        hex_points.extend([
            mapGridpoint2Sphere(centers[i], s1, s2, s3) for i in range(len(centers))
        ])
    hex_points = np.array(hex_points)
    hex_points = np.unique(hex_points, axis=0)

    return hex_points


class Point():

    def __init__(self, x, y, z, value):
        self.x = x
        self.y = y
        self.z = z
        self.value = value

    @classmethod
    def from_coords(cls, coords, values):
        """
        Generates many instances from coordinates

        Args:
            coords: array of coordinates (N x 3)
            values: array of values (N,)

        Returns:
            out: list of point instances
        """
        out = []
        for x, v in zip(coords, values):
            out.append(cls(x[0], x[1], x[2], v))
        return out