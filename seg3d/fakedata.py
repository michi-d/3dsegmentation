
__author__ = ['Michael Drews']

import numpy as np
from math import sin, cos, acos, sqrt, pi
from math import atan2

import utils.volutils as volutils
from scipy.linalg import norm

import ipyvolume as ipv
import numbers


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


class OpticLobeGeometry():
    """
    Base class defining geometric objects within the data.
    Can generate random geometries.
    """

    empty = np.zeros(shape=(0, 3))

    def __init__(self, points=empty, e_r=empty, e_theta=empty, e_phi=empty, center=empty,
                 bezier_points=[]):

        self.points = points  # coordinates of entry points
        self.e_r = e_r  # radial unit vector from ellisoid center
        self.e_theta = e_theta  # theta unit vector (dorso-ventral axis)
        self.e_phi = e_phi  # phi unit vector (anterior-posterior axis)
        self.center = center  # center of ellipsoid
        self.bezier_points = bezier_points  # points defining the neurite bezier curves

    @classmethod
    def generate(cls, n_icosahedron=15, scale_xyz=(1., 1., 1.), seed=0, shift_xyz=(0.5, 0.5, 0.5),
                 rotate_xyz=(0., 0., 0.)):
        """
        Generate geometry for a given set of parameters

        n_icosahedron: (n+1)**2 triangles per face of the icosahedron
        """

        # generate a hexagonal pattern on a sphere using the icosahedron method
        points = get_hex_points_icosahedron(n=n_icosahedron)
        points_rtp = volutils.cart2spherical(points)  # transform to spherical coordinates
        e_r, e_theta, e_phi = volutils.get_spherical_unit_vectors(points_rtp)  # get unit vectors

        # scale coordinates
        points, e_r, e_theta, e_phi = volutils.scale_vectors(
            points, e_r, e_theta, e_phi, scale_xyz=scale_xyz
        )

        # choose random surface point to the origin of the coordinate system
        if isinstance(seed, numbers.Number):
            np.random.seed(seed)
        ind = np.random.randint(low=0, high=len(points))
        A = points[ind]
        points = points - A
        center = (-A).reshape(1, 3)

        n = A / norm(A, ord=2)

        # rotate radial vector of this point to z-axis
        z = np.array([0, 0, 1])
        k = (n + z) / 2
        R = volutils.rodrigues_rotation_matrix(k, np.pi)
        points, e_r, e_theta, e_phi, center = volutils.rotate_vectors(
            points, e_r, e_theta, e_phi, center, R=R
        )

        # shift surface into center of volume (0..1)
        points += np.asarray(shift_xyz).reshape(1, 3) + 0.5
        center += np.asarray(shift_xyz).reshape(1, 3) + 0.5

        # rotate the whole volume
        Rx = volutils.rodrigues_rotation_matrix([1, 0, 0], rotate_xyz[0])
        Ry = volutils.rodrigues_rotation_matrix([0, 1, 0], rotate_xyz[1])
        Rz = volutils.rodrigues_rotation_matrix([0, 0, 1], rotate_xyz[2])
        R = np.dot(Rz, np.dot(Ry, Rx))
        points, e_r, e_theta, e_phi, center = volutils.rotate_vectors(
            points, e_r, e_theta, e_phi, center, R=R
        )

        # clip points far outside the volume
        clip_mask = (points < -0.2).any(axis=1) | (points > 1.2).any(axis=1)
        points, e_r, e_theta, e_phi = volutils.clip_points(
            points, e_r, e_theta, e_phi, clip_mask=clip_mask
        )

        if len(points) == 0:  # return empty volume and try again
            return cls(points, e_r, e_theta, e_phi, center)

        # random cutoff
        cutoff_mask = cls._random_cutoff(points, e_theta, e_phi)
        points, e_r, e_theta, e_phi = volutils.clip_points(
            points, e_r, e_theta, e_phi, clip_mask=cutoff_mask
        )

        # get points for bezier curves (later neurites)
        bezier_points = cls._get_bezier_points(points, e_r, e_theta, e_phi, center)

        return cls(points, e_r, e_theta, e_phi, center, bezier_points)

    @staticmethod
    def _get_bezier_points(points, e_r, e_theta, e_phi, center):
        """Generate points for bezier curves
        """
        # choose random plane perpendicular through surface
        ind = np.random.randint(low=0, high=len(points))
        a = points[ind]
        phi = np.random.uniform(0, 2 * np.pi)
        n = e_phi[ind] * np.cos(phi) + e_theta[ind] * np.sin(phi)
        n = n / norm(n, ord=2)

        # get perpendicular foot point from each point to this plane
        distance = np.dot((points - a), n)
        foot_points = points - distance.reshape(-1, 1) * n.reshape(1, 3)
        p2 = foot_points

        # get offset point in radial direction
        p1 = points - e_r * 0.1

        # end points
        p3 = center + n * distance.reshape(-1, 1) * 0.25  # add spread according to distance
        p3 = p3 + np.random.normal(0, 0.05, size=p3.shape)  # add variability
        return [points, p1, p2, p3]

    @staticmethod
    def _random_cutoff(points, e_theta, e_phi):
        """Random cutoff beyond plane perpendicular to the surface
        """
        # choose random point and random plane orientation
        ind = np.random.randint(low=0, high=len(points))
        a = points[ind]
        w = np.random.uniform(1e-6, 1, size=2)
        n = e_phi[ind] * w[0] + e_theta[ind] * w[1]
        n = n / norm(n, ord=2)

        clip_mask = np.dot((points - a), n) < 0
        return clip_mask

    def _count_points_within_volume(self):
        inside = (self.points > 0).all(axis=1) & (self.points < 1).all(axis=1)
        return inside.sum()

    @classmethod
    def generate_randomly(cls, min_points=50,
                          n_icosahedron=[8, 15], scale_xyz=[0.7, 1.5],
                          shift_xyz=[-0.5, 0.5], rotate_xyz=[0, 360],
                          seed=None):

        if isinstance(seed, numbers.Number):
            np.random.seed(seed)

        n_points = 0
        subseed = np.random.randint(0, 1e6)
        while n_points < 100:
            n_icosahedron_ = np.random.randint(n_icosahedron[0], n_icosahedron[1] + 1)
            scale_xyz_ = np.random.uniform(low=scale_xyz[0], high=scale_xyz[1], size=(3,))
            shift_xyz_ = np.random.uniform(low=shift_xyz[0], high=shift_xyz[1], size=(3,))
            rotate_xyz_ = np.random.uniform(low=rotate_xyz[0], high=rotate_xyz[1], size=(3,))

            new = cls.generate(n_icosahedron=n_icosahedron_, scale_xyz=scale_xyz_, shift_xyz=shift_xyz_,
                               rotate_xyz=rotate_xyz_, seed=subseed)
            n_points = new._count_points_within_volume()
            subseed += 1

        return new

    @staticmethod
    def _my_ipv_vectorplot(xyz, uvw, N=10, length=1, **kwargs):
        """Generates a scatter plot of small points along a vector.

        Args:
            xyz: start point
            uvw: direction
            N: number of points rendered
            length: length of the line
            **kwargs: additional arguments of ipv.scatter
        """
        points = []
        for x in np.linspace(0, length, N):
            tmp = xyz + uvw * x
            points.append(tmp)
        points = np.concatenate(points, axis=0)
        ipv.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)

    @staticmethod
    def _plot_bezier_curves(list_p, N=10, **kwargs):
        """Plot neurites as bezier curves.
        """
        curves = []
        for t in np.linspace(0, 1, N):
            tmp = volutils.bezier(t, list_p)
            curves.append(tmp)
        curves = np.concatenate(curves, axis=0)
        ipv.scatter(curves[:, 0], curves[:, 1], curves[:, 2], **kwargs)

    @staticmethod
    def _arrange_bezier_points(bp, order=3):
        if order == 1:
            p = [bp[0], bp[-1]]
        elif order == 2:
            p = [bp[0], bp[2], bp[3]]
        elif order == 3:
            p = [bp[0], bp[1], bp[2], bp[3]]
        return p

    def plot_geometry(self, plot_e_r=True, plot_e_phi=True, plot_e_theta=True, plot_bezier=True,
                      bezier_order=3):
        """
        Generates 3D ipyvolume plot
        """
        fig = ipv.figure()

        scatter = ipv.scatter(self.center[:, 0], self.center[:, 1], self.center[:, 2], color='#ff0000', size=5)

        if plot_e_r:
            self._my_ipv_vectorplot(np.repeat(self.center, len(self.points), axis=0), self.e_r,
                                    length=1, N=1000, size=0.2, color='#00ff00')

        if plot_e_theta:
            self._my_ipv_vectorplot(self.points, self.e_theta,
                                    length=0.05, N=100, size=0.2, color='#ff0000')

        if plot_e_phi:
            self._my_ipv_vectorplot(self.points, self.e_phi,
                                    length=0.05, N=100, size=0.2, color='#ff0000')

        if plot_bezier:
            assert 1 <= bezier_order <= 3
            p = self._arrange_bezier_points(self.bezier_points, order=bezier_order)
            self._plot_bezier_curves(p, N=100, size=0.5, color='#ff00ff')

        scatter = ipv.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], color='#0000bb', size=1)

        ipv.xlim(0, 1)
        ipv.ylim(0, 1)
        ipv.zlim(0, 1)

        return fig


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