
__author__ = ['Michael Drews']

import numpy as np
from math import sin, cos, acos, sqrt, pi
from math import atan2

import utils.volutils as volutils
from scipy.linalg import norm

import ipyvolume as ipv
import numbers
import cv2


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
        """
        Generates empty object.
        Call alternative generator methods instead.
        """
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
        Generate basic geometry for a given set of parameters.
        - Generates hexagonal points on a sphere
        - Performs scaling, shifting and rotations
        - Generates a set of points defining bezier curves for the neurites

        Args:
            n_icosahedron: (n+1)**2 triangles per face of the icosahedron
            scale_xyz: scaling factors for each axis
            seed: random seed
            shift_xyz: shift along xyz
            rotate_xyz: rotation angles around x,y,z axis

        Returns:
            Class object with the given parameters.
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

        # remove duplicate points
        points, remove_indices = volutils.unique_vectors(points, tol=1e-6) # delete duplicates
        e_r = np.delete(e_r, remove_indices, axis=0)
        e_theta = np.delete(e_theta, remove_indices, axis=0)
        e_phi = np.delete(e_phi, remove_indices, axis=0)

        # get points for bezier curves (later neurites)
        bezier_points = cls._get_bezier_points(points, e_r, e_theta, e_phi, center)
        #import pdb; pdb.set_trace()
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
        w_r = np.random.uniform(0, 0.2, size=(len(e_r))).reshape(-1,1)
        w_phi = np.random.uniform(0, 0.1, size=(len(e_phi))).reshape(-1,1)
        w_theta = np.random.uniform(0, 0.1, size=(len(e_theta))).reshape(-1,1)
        p1 = points - e_r*w_r + e_phi*w_phi + e_theta*w_theta

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
    def generate_randomly(cls, min_points=100,
                          n_icosahedron=[8, 15], scale_xyz=[0.7, 1.5],
                          shift_xyz=[-0.5, 0.5], rotate_xyz=[0, 360],
                          seed=None):
        """
        Generates a random volume with random geometry.
        Calls "generate" with random parameters.
        Args:
            min_points: acceptable minimum number of points
            n_icosahedron: [low high] range of values for n_icosahedron
            scale_xyz: [low high] range of values for scaling factors
            shift_xyz: [low high] range of values for shifts
            rotate_xyz: [low high] range of values for rotation angles
            seed: random seed

        Returns:
            Randomly initiated class object.
        """

        if isinstance(seed, numbers.Number):
            np.random.seed(seed)

        n_points = 0
        subseed = np.random.randint(0, 1e6)
        while n_points < min_points:
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
        """Arranges list of bezier points according to given order.
        """
        if order == 1:
            p = [bp[0], bp[-1]]
        elif order == 2:
            p = [bp[0], bp[2], bp[3]]
        elif order == 3:
            p = [bp[0], bp[1], bp[2], bp[3]]
        return p

    def plot_geometry(self, plot_e_r=True, plot_e_phi=True, plot_e_theta=True, plot_bezier=True,
                      bezier_order=3):
        """Generates 3D ipyvolume plot
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


class RandomVolume():

    def __init__(self, L_xy=256, L_z=256,
                 relative_diameter=0.70, relative_depth=3 * 0.75,
                 type_ratios=[2, 3, 3, 3], var_range=[0.7, 1.0],
                 cube_kernel_size=4, NF_size=91, simplify_factor=1.,
                 random_geometry=False, vol_dtype='float32',
                 seed=None, verbose=1, **kwargs):
        """
        Generates FakeData3D object


        **kwargs: additional arguments are for the geometry generator
        """
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed()

        # generate underlying geometry
        if random_geometry:
            if verbose:
                print("Generating random geometry...")
            self.geometry = OpticLobeGeometry.generate_randomly(**kwargs)
        else:
            if verbose:
                print("Generating standard geometry...")
            self.geometry = OpticLobeGeometry.generate(**kwargs)

        self.relative_diameter = relative_diameter
        self.relative_depth = relative_depth
        self.L_xy = L_xy
        self.L_z = L_z
        self.cube_kernel_size = cube_kernel_size
        self.NF_size = NF_size
        self.vol_dtype = vol_dtype
        self.type_ratios = type_ratios
        self.var_range = var_range
        self.verbose = verbose
        self.simplify_factor = simplify_factor

        self._gen_space()
        self._gen_ground_truth()

        # choose random columns for the different types (empty, A, B, C)
        self.mid_points = self.geometry.points + self.geometry.e_r * self.depth / 2
        fractions = self.type_ratios / np.sum(self.type_ratios)
        fractions = np.cumsum(fractions)
        ind = (fractions * len(self.mid_points)).astype(np.int)
        permutation = np.random.permutation(range(len(self.mid_points)))

        ind_empty = permutation[:ind[0]]
        ind_A = permutation[ind[0]:ind[1]]
        ind_B = permutation[ind[1]:ind[2]]
        ind_C = permutation[ind[2]:ind[3]]
        # print(len(ind_empty), len(ind_A), len(ind_B), len(ind_C))

        # render columns type A
        if self.verbose:
            print("Generating columns type A...")
        self._generate_knobs(
            self.mid_points[ind_A], self.geometry.e_r[ind_A], self.geometry.e_theta[ind_A], self.geometry.e_phi[ind_A],
            n_points=10, n_points_final=250, baseline_intensity=50,
            std_intensity=1.2, v_std=1.0, std_radial=0.12/self.simplify_factor, std_axial=0.10 * 2/self.simplify_factor,
            n_iterations=3, mode='A'
        )

        if self.verbose:
            print("Generating columns type B...")
        self._generate_knobs(
            self.mid_points[ind_B], self.geometry.e_r[ind_B], self.geometry.e_theta[ind_B], self.geometry.e_phi[ind_B],
            n_points=2, n_points_final=250, baseline_intensity=50,
            std_intensity=1.0, v_std=1.0, std_radial=0.12/self.simplify_factor, std_axial=0.10 * 2/self.simplify_factor,
            n_iterations=4, mode='B'
        )
        if self.verbose:
            print("Generating columns type C...")
        self._generate_knobs(
            self.mid_points[ind_C], self.geometry.e_r[ind_C], self.geometry.e_theta[ind_C], self.geometry.e_phi[ind_C],
            n_points=75, n_points_final=250, baseline_intensity=50,
            std_intensity=1.0, v_std=1.0, std_radial=0.06/self.simplify_factor, std_axial=0.10 * 2/self.simplify_factor,
            n_iterations=2, mode='C', ring_radius=0.3, ring_std=0.1
        )

        if self.verbose:
            print("Generating neurites...")
        self._generate_neurites()

        if self.verbose:
            print("Applying filters...")
        self._sim_2p_imaging(s=self.cube_kernel_size, NF_size=self.NF_size)

        # scale variance of distribution randomly
        self.vol_data = self._scale_variance_randomly(self.vol_data, var_range=self.var_range)

    def _gen_space(self):

        self.vol_data = np.zeros((self.L_xy, self.L_xy, self.L_z), dtype=self.vol_dtype)
        self.vol_labels = np.zeros((self.L_xy, self.L_xy, self.L_z), dtype='bool')

        # generate space coordinates
        X, Y, Z = np.mgrid[0:self.L_xy, 0:self.L_xy, 0:self.L_z]
        self.X = (X / self.L_xy).astype(self.vol_dtype)
        self.Y = (Y / self.L_xy).astype(self.vol_dtype)
        self.Z = (Z / self.L_z).astype(self.vol_dtype)

    def _gen_ground_truth(self):

        if self.verbose:
            print("Generating ground truth...")

        # find mean distances between neighbors (at entry points)
        distances = norm(self.geometry.points.reshape(1, -1, 3) - \
                         self.geometry.points.reshape(-1, 1, 3), ord=2, axis=2)
        mean_spacing = np.mean(np.ma.masked_equal(distances, 0.0, copy=True).min(axis=0).data)
        self.mean_spacing = mean_spacing

        # define radius at entry points
        r0 = mean_spacing * self.relative_diameter / 2.
        self.depth = mean_spacing * self.relative_depth

        # find end points
        end_points = self.geometry.points + self.depth * self.geometry.e_r
        end_distances = norm(end_points.reshape(1, -1, 3) -
                             end_points.reshape(-1, 1, 3), ord=2, axis=2)
        end_mean_spacing = np.mean(np.ma.masked_equal(end_distances, 0.0, copy=True).min(axis=0).data)
        self.end_mean_spacing = end_mean_spacing

        # find radius factor at end points
        depth_factor = end_mean_spacing / mean_spacing

        # calculate optimal step width for 3D loop
        dphi = 0.6 / (max(self.L_xy, self.L_z) * r0 * depth_factor)
        dr = 0.6 / max(self.L_xy, self.L_z)
        dz = 0.6 / max(self.L_xy, self.L_z)

        # generate labeled 3D-cones at each point
        phi_range = np.arange(0, 2 * np.pi, dphi)
        r_range = np.arange(0, r0, dr)
        z_range = np.arange(0, self.depth, dz)

        for phi in phi_range:
            for r in r_range:
                for z in z_range:
                    r_ = r * ((z / self.depth) * (depth_factor - 1) + 1)

                    p = self.geometry.points + r_ * np.cos(phi) * self.geometry.e_phi + \
                        r_ * np.sin(phi) * self.geometry.e_theta + \
                        z * self.geometry.e_r

                    ind = np.round(p * np.array([self.L_xy, self.L_xy, self.L_z]), 2).astype(np.int)
                    x_clip = (0 <= ind[:, 0]) & (ind[:, 0] < self.L_xy)
                    y_clip = (0 <= ind[:, 1]) & (ind[:, 1] < self.L_xy)
                    z_clip = (0 <= ind[:, 2]) & (ind[:, 2] < self.L_z)
                    ind = ind[x_clip & y_clip & z_clip]

                    self.vol_labels[ind[:, 0], ind[:, 1], ind[:, 2]] = 1

    @staticmethod
    def _generate_gaussian_cloud(P, n_points, v_mean, v_std, x_mean, x_std, y_mean, y_std, z_mean, z_std):
        """
        Generates a cloud of points within a 3D Gaussian ellipsoid around P.

        Args:
            P: (r, e_r, e_theta, e_phi) of central points
            n_points: how many points to place in vicinity of P
            v_mean: mean intensity of points
            v_std: std of intensity
            x_mean: center of gaussian along e_phi direction (usually 0)
            x_std: std of gaussian along e_phi direction
            y_mean, y_std: same of e_theta direction
            z_mean, z_std: same for e_r direction

        Returns:
            cloud: set of points (n_points x 3)
            values: intensity values for each point
        """

        r, e_r, e_theta, e_phi = P

        x = np.random.normal(loc=x_mean, scale=x_std, size=n_points).reshape(-1, 1)
        y = np.random.normal(loc=y_mean, scale=y_std, size=n_points).reshape(-1, 1)
        z = np.random.normal(loc=z_mean, scale=z_std, size=n_points).reshape(-1, 1)

        values = np.random.normal(loc=v_mean, scale=v_std, size=n_points)

        cloud = r + x * e_phi.reshape(1, 3) + y * e_theta.reshape(1, 3) + z * e_r.reshape(1, 3)

        return cloud, values

    @staticmethod
    def _generate_ring_cloud(P, n_points, v_mean, v_std, ring_radius, ring_std, z_mean, z_std):
        """
        Generates a cloud of points within a 3D "empty" cylinder around P

        Args:
            P: (r, e_r, e_theta, e_phi) of central points
            n_points: how many points to place in vicinity of P
            v_mean: mean intensity of points
            v_std: std of intensity
            ring_radius: radius of cylinder
            r_std: std of radius
            z_mean: center of gaussian along e_r direction (usually 0)
            z_std: std of gaussian along e_r direction

        Returns:
            cloud: set of points (n_points x 3)
            values: intensity values for each point
        """

        r, e_r, e_theta, e_phi = P

        radius = np.random.normal(loc=ring_radius, scale=ring_std, size=n_points).reshape(-1, 1)
        phi = np.random.uniform(low=0, high=360, size=n_points).reshape(-1, 1)
        z = np.random.normal(loc=z_mean, scale=z_std, size=n_points).reshape(-1, 1)

        values = np.random.normal(loc=v_mean, scale=v_std, size=n_points)

        cloud = r + np.cos(phi) * radius * e_phi.reshape(1, 3) + \
                np.sin(phi) * radius * e_theta.reshape(1, 3) + \
                z * e_r.reshape(1, 3)

        return cloud, values

    def _generate_knobs(self, points, e_r, e_theta, e_phi,
                        n_points=10, n_points_final=250,
                        baseline_intensity=50, std_intensity=1.2, v_std=1.0,
                        std_radial=0.12, std_axial=0.10 * 2,
                        n_iterations=1, current_depth=1, mode='A',
                        ring_radius=0.3, ring_std=0.1):
        """
        Recursively generates gaussian clouds around gaussian clouds of a set of input points.

        Args:
            points: seed set of points
            e_r, e_theta, e_phi: associated unit vectors
            n_points: number of points to generate at each iteration
            n_points_final: number of points in final cloud around each point
            baseline_intensity: general baseline intensity
            std_intensity: std of lognormal distribution around baseline
            v_std: std of intensity for each local point cloud
            std_radial: std in x or y direction (e_phi, e_theta)
            std_axial: std along z direction (e_r)
            n_iteration: Iteration depth
            mode: A (more dense gaussians), B (more clustered), C (ring structure)
            ring_radius: cylinder radius for type C
            ring_std: std for cylinder radius for type C
        """

        for idx in range(len(points)):  # for each point

            # set point
            P = (points[idx],
                 e_r[idx],
                 e_theta[idx],
                 e_phi[idx])

            # generate sub-cloud of points around P
            v_mean = np.random.lognormal(mean=1.0, sigma=std_intensity, size=(1)) * baseline_intensity
            v_std = v_std
            x_mean = 0
            x_std = self.mean_spacing * std_radial
            y_mean = 0
            y_std = self.mean_spacing * std_radial
            z_mean = 0
            z_std = self.mean_spacing * std_axial

            if mode == 'A':
                cloud_points, values = self._generate_gaussian_cloud(
                    P, n_points, v_mean, v_std, x_mean, x_std, y_mean, y_std, z_mean, z_std
                )
            elif mode == 'B':
                cloud_points, values = self._generate_gaussian_cloud(
                    P, n_points + current_depth, v_mean, v_std, x_mean, x_std, y_mean, y_std, z_mean, z_std
                )
            elif mode == 'C':
                # "ring structure"
                ring_radius_ = ring_radius * self.mean_spacing
                ring_std_ = ring_std * self.mean_spacing
                cloud_points, values = self._generate_ring_cloud(
                    P, n_points, v_mean, v_std, ring_radius_, ring_std_, z_mean, z_std
                )

            # if max iteration depth is reached: generate a final cloud with 'n_points_final' points
            if current_depth == n_iterations:

                # randomly chose a new z-variation for variation in "vertical structures"
                z_std = np.random.normal(loc=z_std, scale=z_std)
                z_std = max(0, z_std)

                # generate final point cloud
                cloud_points, values = self._generate_gaussian_cloud(
                    P, n_points_final, v_mean, v_std, x_mean, x_std, y_mean, y_std, z_mean, z_std
                )

                # finally, render points
                ind = np.round(cloud_points * np.array([self.L_xy, self.L_xy, self.L_z]), 2).astype(np.int)
                x_clip = (0 <= ind[:, 0]) & (ind[:, 0] < self.L_xy)
                y_clip = (0 <= ind[:, 1]) & (ind[:, 1] < self.L_xy)
                z_clip = (0 <= ind[:, 2]) & (ind[:, 2] < self.L_z)
                ind = ind[x_clip & y_clip & z_clip]
                values = values[x_clip & y_clip & z_clip]

                self.vol_data[ind[:, 0], ind[:, 1], ind[:, 2]] = values

            else:

                # generate subcloud with unit vectors for the given center point
                new_e_r = e_r[idx].reshape(1, 3).repeat(len(cloud_points), axis=0)
                new_e_theta = e_theta[idx].reshape(1, 3).repeat(len(cloud_points), axis=0)
                new_e_phi = e_phi[idx].reshape(1, 3).repeat(len(cloud_points), axis=0)

                # call itself with same parameters but 1 more depth
                self._generate_knobs(
                    cloud_points, new_e_r, new_e_theta, new_e_phi,
                    n_points=n_points, n_points_final=n_points_final, baseline_intensity=baseline_intensity,
                    std_intensity=std_intensity, v_std=v_std, std_radial=std_radial, std_axial=std_axial,
                    n_iterations=n_iterations, current_depth=current_depth + 1, mode=mode
                )

    def _generate_neurites(self, baseline_intensity=50, std_intensity=1.0, intensity_factor=2,
                           frac_empty=0.3, position_noise=0.025):
        """
        Walk along bezier curves and fill voxels.
        """

        # get bezier points
        list_p = self.geometry._arrange_bezier_points(self.geometry.bezier_points, order=3)

        # seed intensity values
        values_ = np.random.lognormal(mean=1.0, sigma=std_intensity,
                                      size=(len(self.geometry.points))) * baseline_intensity

        # delete some neurites
        ind = np.random.permutation(range(len(values_)))[:int(frac_empty * len(values_))]
        values_[ind] = 0

        # generate points
        curves = []
        values = []
        for t in np.linspace(0, 1, max(self.L_xy, self.L_z)):
            tmp = volutils.bezier(t, list_p)
            curves.append(tmp)
            values.append(values_)
        curves = np.concatenate(curves, axis=0)
        values = np.concatenate(values, axis=0)

        # add position noise
        curves = np.repeat(curves, 4, axis=0)
        values = np.repeat(values, 4, axis=0)*intensity_factor

        noise = np.random.normal(loc=0, scale=self.mean_spacing * position_noise, size=curves.shape)
        curves = curves + noise

        # finally, render points
        ind = np.round(curves * np.array([self.L_xy, self.L_xy, self.L_z]), 2).astype(np.int)
        x_clip = (0 <= ind[:, 0]) & (ind[:, 0] < self.L_xy)
        y_clip = (0 <= ind[:, 1]) & (ind[:, 1] < self.L_xy)
        z_clip = (0 <= ind[:, 2]) & (ind[:, 2] < self.L_z)
        ind = ind[x_clip & y_clip & z_clip]
        values = values[x_clip & y_clip & z_clip]

        self.vol_data[ind[:, 0], ind[:, 1], ind[:, 2]] += values

    def _sim_2p_imaging(self, s=4, NF_size=91, NF_factor=0.0005, plateau_factor=1 / 2.,
                        poisson_factor=20.):
        """
        Filter data to simulate real fluorescence imaging

        Args:
            s: Magnification kernel size for points
        """
        tmp = np.copy(self.vol_data)

        # 3d convolution as separated 2d convolutions
        kernel_xy = np.ones((s, s), dtype=np.float32)
        kernel_z = np.ones((1, s), dtype=np.float32)

        NF = np.ones_like(tmp)  # background fluorescence

        for i in range(self.vol_data.shape[2]):  # along z-axis
            img = tmp[:, :, i]
            tmp[:, :, i] = cv2.filter2D(img, -1, kernel_xy)
            NF[:, :, i] = cv2.GaussianBlur(tmp[:, :, i], (121, 121), sigmaX=NF_size)

        for i in range(self.vol_data.shape[0]):  # along x-axis (could also be y)
            img = tmp[i, :, :]
            tmp[i, :, :] = cv2.filter2D(img, -1, kernel_z)
            NF[i, :, :] = cv2.GaussianBlur(NF[i, :, :], (121, 121), sigmaX=NF_size)

        tmp = tmp / np.percentile(tmp.flatten(), 99)
        tmp = tmp + NF * NF_factor + np.percentile(tmp.flatten(), 99) * plateau_factor
        tmp = np.random.poisson(tmp * poisson_factor)
        tmp[tmp < 0] = 0

        self.vol_data = tmp

    @staticmethod
    def _scale_variance_randomly(data, var_range=[0.6, 1.0]):

        new_var = np.random.uniform(low=var_range[0], high=var_range[1])

        data_ = np.log(data + 1e-7)
        mu = data_.mean()
        data_ = (data_ - mu) * new_var
        data_ = data_ + mu
        data_ = np.exp(data_)

        return data_


