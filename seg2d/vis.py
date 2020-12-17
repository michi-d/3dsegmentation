"""
Helper functions for 2D data visualization
"""

__author__ = ['Michael Drews']

import matplotlib.pyplot as plt
import numpy as np


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    fig = plt.figure(figsize=(16, 5))
    for i, (name, item) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())

        if isinstance(item, tuple):
            image = np.squeeze(item[0])
            points = np.squeeze(item[1])
            plt.imshow(image)
            plt.scatter(x=points[:,1], y=points[:,0], s=10, color='red')
        else:
            image = np.squeeze(item)
            plt.imshow(image)

        plt.colorbar()
    plt.show()

    return fig
