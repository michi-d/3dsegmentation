"""
Modulate for creation of fake data in 2D.
"""

__author__ = ['Michael Drews']

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Point():    
    """Defines a point in 2D space with an intensity value.
    """
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

        
class Dendrite():
    """Defines a dendrite as a cloud of points.
    """
    
    def __init__(self, x0, y0, intensity, n_points, std_x, std_y, std_v, 
                 ring_d, ring_std, render_size, mode='ring'):
        """Generator method.
        
        Args:
            x0: x-coordinate center point
            y0: y-coordinate center point
            intensity: intensity of the dendrite
            n_points: how many points in the cloud
            std_x: standard deviation in x (for "gaussian" mode)
            std_y: standard deviation in y (for "gaussian" mode)
            std_v: standard deviation of the intensity
            ring_d: distance ring (for "ring" mode)
            ring_std: standard deviation in distance from ring (for "ring" mode)
            render_size: size of rendering window
            mode: "ring" or "gaussian"
        """
        
        self.points = []
        center_point = Point(x0, y0, intensity)
        self.points.append(center_point)
        
        self.x0 = x0
        self.y0 = y0
        self.intensity = intensity
        self.std_x = std_x
        self.std_y = std_y
        self.std_v = std_v
        self.n_points = n_points
        self.ring_d = ring_d
        self.ring_std = ring_std
        self.render_size = render_size
        self.mode = mode

    def clean_mask_out_of_bounds(self, A):
        """Deletes all points with any coordinate larger than a bound.

        Args:
            A: input array

        Returns:
            mask: boolean mask for points within the bounds
        """
        bound = self.render_size[0]
        mask_low = ~(A < 0).any(axis=1)
        mask_high = ~(A >= bound).any(axis=1)
        return mask_low & mask_high
        
    def grow(self):
        """Generate point cloud around the center point
        """
        n_points = self.n_points
        std_x = self.std_x
        std_y = self.std_y
        std_v = self.std_v
        mode = self.mode
          
        if mode == 'ring':
            r = np.round(np.random.normal(loc=self.ring_d, scale=self.ring_std, size=n_points), 2)
            phi = np.round(np.random.uniform(low=0, high=360, size=n_points), 2)
            x = self.x0 + (np.cos(phi)*r).astype(np.int)
            y = self.y0 + (np.sin(phi)*r).astype(np.int)
        elif mode == 'gaussian':
            x = np.round(np.random.normal(loc=self.x0, scale=std_x, size=n_points), 2).astype(np.int)
            y = np.round(np.random.normal(loc=self.y0, scale=std_y, size=n_points), 2).astype(np.int)

        v = np.random.normal(loc=self.intensity, scale=std_v, size=n_points)
        
        xy = np.stack([x, y], axis=1)
        clean_mask = self.clean_mask_out_of_bounds(xy)
        xy = xy[clean_mask, :]
        v  = v[clean_mask]
        
        for i in range(len(xy)):
            self.points.append(Point(xy[i, 0], xy[i, 1], v[i]))
            
            
class RandomImage():
    """RandomImage fake data class
    """
    
    def __init__(self, basis_angle=60, d=10, baseline_intensity=50, std_intensity=0.5, n_points=10, 
                 std_x=2, std_y=2, std_v=5, dendrite_kernel=(2,2), gaussian_kernel=(5,5),
                 ring_d=3.5, ring_std=0.1, ring_frac=0.5, randomCutOff=True, randomRotation=True,
                 randomAngles = (-10, 30), randomFraction=True,
                 randomScaling = (1.28*0.9, 1.28*1.5), size=(128,128)):

        assert size[0] == size[1]
    
        self.radius_ground_truth = 3.5
        
        self.basis_angle = basis_angle
        self.d = d
        self.baseline_intensity = baseline_intensity
        self.std_intensity = std_intensity
        self.n_points = n_points
        self.std_x = std_x
        self.std_y = std_y
        self.std_v = std_v
        self.dendrite_kernel = dendrite_kernel
        self.gaussian_kernel = gaussian_kernel
        self.size = size
        self.ring_d = ring_d
        self.ring_std = ring_std
        self.ring_frac = ring_frac
        self.randomCutOff = randomCutOff
        self.randomRotation = randomRotation
        self.randomAngles = randomAngles
        self.randomFraction = randomFraction
        self.randomScaling = randomScaling
        
        # generate scaling factor
        scaling_factor = np.random.uniform(*self.randomScaling)
        
        # rescale parameters
        self.scaling_factor = scaling_factor
        self.d = self.d*self.scaling_factor
        self.std_x = self.std_x*scaling_factor
        self.std_y = self.std_y*scaling_factor
        self.ring_d = self.ring_d*scaling_factor
        self.ring_std = self.ring_std*scaling_factor
        self.dendrite_kernel = (np.round(self.dendrite_kernel[0]*scaling_factor, 2).astype('int'),
                                np.round(self.dendrite_kernel[1]*scaling_factor, 2).astype('int'))
        self.gaussian_kernel = (self._next_uneven_number(self.gaussian_kernel[0]*scaling_factor),
                                self._next_uneven_number(self.gaussian_kernel[1]*scaling_factor))
        self.radius_ground_truth = self.radius_ground_truth*scaling_factor
        self.n_points = int(self.n_points*scaling_factor)
        
        # generate picture
        self._generate()

    def clean_mask_out_of_bounds(self, A):
        """Deletes all points with any coordinate larger than a bound.

        Args:
            A: input array

        Returns:
            mask: boolean mask for points within the bounds
        """
        bound = self.size[0]
        mask_low = ~(A < 0).any(axis=1)
        mask_high = ~(A >= bound).any(axis=1)
        return mask_low & mask_high

    @staticmethod
    def _next_uneven_number(i):
        """Finds next lower uneven number
        """
        return int(np.ceil(i/2)*2-1)

    def _cut_off_points_randomly(self, points):
        """Cut off all points above a random line.
        """
        
        # generate random normal vector
        phi = np.random.uniform(low=0, high=2*np.pi)
        n = np.array([np.cos(phi), np.sin(phi)])
        
        # generate random support vector
        a = [np.random.uniform(low=0, high=self.size[0]), np.random.uniform(low=0, high=self.size[1])]
        a = np.array(a)
        
        # mask all points on one side of the line
        proj = np.sum(n * (points-a), axis=1)
        #print(proj)
        
        mask = proj > 0
        
        return points[mask]
    
    def _rotate_points_randomly(self, points):
        """Rotate all points randomly
        """
        e_0 = np.array([self.size[0]/2., self.size[1]/2.])
        rot_angle = np.random.uniform(0, 2*np.pi)
        M = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle), np.cos(rot_angle)]])
        points = (np.dot(points-e_0, M) + e_0).astype(np.int)
        return points
        
    def _generate(self):
        """Generates the image
        """

        # initialize empty image array
        C = np.zeros(self.size)

        # define grid basis
        basis_angle = self.basis_angle
        basis_angle = basis_angle + np.random.uniform(*self.randomAngles)
        e_1 = np.array([np.sin(np.degrees(0)), np.cos(np.degrees(0))])
        e_2 = np.array([np.sin(np.radians(basis_angle)), np.cos(np.radians(basis_angle))])

        e_0 = np.array([self.size[0]/2., self.size[1]/2.])
        X,Y = np.mgrid[-10:10, -10:10]

        # create center points
        midpoints = (e_0 + X.flatten()[:,np.newaxis] * e_1*self.d + Y.flatten()[:,np.newaxis] * e_2*self.d)
        midpoints = np.round(midpoints, 2).astype(np.int)
        
        # cut-off points randomly
        if self.randomCutOff:
            midpoints = self._cut_off_points_randomly(midpoints)
        
        # random image rotation
        if self.randomRotation:
            midpoints = self._rotate_points_randomly(midpoints)
            
        # clean out-of-bounds points
        midpoints = midpoints[self.clean_mask_out_of_bounds(midpoints)]

        # generate random ring fraction
        if self.randomFraction:
            self.ring_frac = np.random.uniform(0, 1)
            
        # generate dendrites
        dendrites = []
        for point in midpoints:
            intensity = np.random.lognormal(mean=1.0, sigma=self.std_intensity, 
                                            size=(1))*self.baseline_intensity
            if np.random.uniform(low=0, high=1) > 1 - self.ring_frac:
                mode = 'ring'
            else:
                mode = 'gaussian'
                
            d = Dendrite(point[0], point[1], intensity, self.n_points, self.std_x, self.std_y, self.std_v, 
                         self.ring_d, self.ring_std, self.size, mode)
            d.grow()
            dendrites.append(d)

        # render points
        for d in dendrites:
            for p in d.points:
                C[p.x, p.y] = p.value
        kernel = np.ones(self.dendrite_kernel,np.float32)/(self.dendrite_kernel[0]*self.dendrite_kernel[1])
        
        # enlarge points
        C = cv2.filter2D(C, -1, kernel)

        # blur points
        C = cv2.GaussianBlur(C, self.gaussian_kernel, 0)
        C[C<0] = 0
        
        # neuropil background fluorescence
        #C = C+C.mean()
        NF = cv2.GaussianBlur(C, (21,21), 0)
        C = C + NF + C.mean()

        # poisson noise
        P = np.zeros_like(C)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i,j] = np.random.poisson(C[i,j])

                
        # gaussian noise
        P = P + np.random.normal(loc=0, scale=P.std()/10.0, size=P.shape)
                
        self.data = P
        self.labels = self._generate_ground_truth(midpoints)
        self.midpoints = midpoints
        
    def _generate_ground_truth(self, midpoints):
        """Generates binary labels.
        """
        
        C = np.zeros(self.size)
        X,Y = np.mgrid[0:self.size[0], 0:self.size[1]]
        xy = np.stack([X.flatten(), Y.flatten()], axis=1)
        difference = (xy[:, :, np.newaxis] - np.array(midpoints).T[np.newaxis, :, :])
        distances = norm(difference, axis=1)
        
        mask = np.any(distances<self.radius_ground_truth, axis=1)
        mask = np.reshape(mask, self.size)
        return mask.astype(np.int)

    def show_data_plus_labels(self, figsize=(15,10), red_crosses=False):
        """Show image and labels in matplotlib figure.
        """
        
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(121)
        im = plt.imshow(self.data)
        plt.title('data')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        if red_crosses:
            ax.scatter(y=self.midpoints[:,0], x=self.midpoints[:,1], s=25, marker='x', color='r')

        ax = plt.subplot(122)
        im = plt.imshow(self.labels, alpha = 1)
        plt.title('labels')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        if red_crosses:
            ax.scatter(y=self.midpoints[:,0], x=self.midpoints[:,1], s=25, marker='x', color='r')
        
        return fig

    def get_parameters(self):
        """Get generative parameters.
        """
        
        params = {'radius_ground_truth': self.radius_ground_truth,
                  'basis_angle': self.basis_angle,
                  'd': self.d, 
                  'baseline_intensity': self.baseline_intensity,
                  'std_intensity': self.std_intensity,
                  'n_points': self.n_points,
                  'std_x': self.std_x,
                  'std_y': self.std_y,
                  'std_v': self.std_v,
                  'dendrite_kernel': self.dendrite_kernel, 
                  'gaussian_kernel': self.gaussian_kernel,
                  'size': self.size, 
                  'ring_d': self.ring_d, 
                  'ring_std': self.ring_std, 
                  'ring_frac': self.ring_frac, 
                  'scaling_factor': self.scaling_factor}
        
        return params