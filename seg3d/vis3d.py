import ipyvolume as ipv
import time
import numpy as np
from ipywidgets import Layout, Box, VBox, HBox, HTML, Image
import matplotlib.pyplot as plt


def get_perspective(fig=None):
    """Get camera angle and rotation from ipyvolume figure. 
    
    Args:
        fig: target figure
    
    Returns:
        view: camera position
        rotation: camera angle
    """
    
    if fig is None:
        fig = ipv.pylab.gcf()
        
    view = ipv.pylab.view()
    rotation = fig.camera.rotation
    
    return view, rotation


def set_perspective(view=(0,0,0), rotation=(0,0,0, 'XYZ'), perspective=None):
    """Set camera angle and rotation for ipyvolume figure.
    
    Args:
        view: camera position
        rotation: camera angle
    """
    
    # set camera position and angle
    if perspective is None:
        return 
    elif perspective == 'oblique':
        # oblique view
        view = (-133.60365788878377, 54.701906702426264, 2.4643234833362575)
        rotation = (-2.0250209294420607, -0.43170845694524784, -0.165007940051523, 'XYZ')
    elif perspective == 'Y':
        # view from y-axis down
        view = (-180, 90, 2)
        rotation = (-1., 0, 0, 'XYZ')
    elif perspective == 'Z':
        # view along z-axis
        view = (180, 0, 2)
        rotation = (1., 0, 0, 'XYZ')
    elif perspective == 'X':
        # view from X-axis down
        view = (90, 0, 2)
        rotation = (-np.pi/2, np.pi/2, 0.0, 'XYZ')

    fig = ipv.pylab.gcf()
    ipv.pylab.view(azimuth=view[0], elevation=view[1], distance=view[2])
    fig.camera.rotation = rotation
    
    
def my_volshow_1channel(volume, view=(0,0,0), rotation=(0,0,0, 'XYZ'), perspective=None, **kwargs):
    """Customized volume show function, with specified camera position and angle.
    
    Args:
        volume: volume data
        view: camera position
        rotation: camera angle
    """
        
    ipv.quickvolshow(volume, **kwargs)
    ipv.show()
    
    # wait for plot to be generated (JavaScript)
    time.sleep(0.6)
    
    # set camera position and angle
    set_perspective(view=view, rotation=rotation, perspective=perspective)
    
    
def my_volshow_2channel(volume, view=(0,0,0), rotation=(0,0,0, 'XYZ'), perspective=None, 
                        plot_params=None):
    """Customized volume show function, with specified camera position and angle.
    
    Args:
        volume: multi-channnel volume data
        view: camera position
        rotation: camera angle
        perspective: alternatively, pre-saved camera perspective
        plot_params: list of key-word arguments as a dictionary for ipv.volshow
    """
    
    n_channels = len(volume)
    if plot_params is None:
        plot_params = [{}]*n_channels
    assert len(plot_params) == n_channels 
       
    ipv.pylab.figure()
    for ch in range(n_channels):
        ipv.volshow(volume[ch, :], **plot_params[ch])
    ipv.show()
    
    # wait for plot to be generated (JavaScript)
    time.sleep(1)
    
    # set camera position and angle
    set_perspective(view=view, rotation=rotation, perspective=perspective)
    
    
def generate_3dmovies_1channel(stack, T=5, perspectives=['oblique', 'X', 'Y', 'Z'], name='', **kwargs):
    """Generate 4 movies for each perspective
    
    Args:
        stack: Input data
        T: length of each movie in seconds
    """
    assert len(perspectives) > 0

    # generate plot
    my_volshow(stack[0,:], perspective=perspectives[0], **kwargs)

    N_frames = len(stack)
    fps = N_frames/T

    # make 4 movies from 4 perspectives
    for perspective in perspectives:
        print(f'Generating {perspective} perspective...')

        # set perspective
        set_perspective(perspective=perspective)
        time.sleep(1)

        # define helper function
        fig = ipv.pylab.gcf()
        def go_through_volume(fig, i, fraction):
            frame = np.round(fraction*N_frames).astype(np.int)
            fig.volumes[0].data = stack[frame, :]

        # create movie
        tmpdir = ipv.pylab.movie(f'{name}{perspective}.gif', go_through_volume, fps=fps, frames=N_frames, endpoint=False)

        # convert movie
        #command = f'convert -delay 5.0 -loop 0 {tmpdir}/frame-*.png {perspective}.gif'
        #stream = os.popen(command)
        #output = stream.read()
        
        
def generate_3dmovies_2channel(stack, T=5, perspectives=['oblique', 'X', 'Y', 'Z'], name='',
                               plot_params=None):
    """Generate 4 movies for each perspective
    
    Args:
        stack: Input data
        T: length of each movie in seconds
        plot_params: list of key-word arguments as a dictionary for ipv.volshow
    """
    assert len(perspectives) > 0
    
    n_channels = stack.shape[1]
    if plot_params is None:
        plot_params = [{}]*n_channels
    assert len(plot_params) == n_channels 

    # generate plot
    my_volshow_2channel(stack[0,:], perspective=perspectives[0], plot_params=plot_params)

    N_frames = len(stack)
    fps = N_frames/T

    # make 4 movies from 4 perspectives
    for perspective in perspectives:
        print(f'Generating {perspective} perspective...')

        # set perspective
        set_perspective(perspective=perspective)
        time.sleep(1)

        # define helper function
        fig = ipv.pylab.gcf()
        def go_through_volume(fig, i, fraction):
            frame = np.round(fraction*N_frames).astype(np.int)
            for ch in range(n_channels):
                fig.volumes[ch].data = stack[frame, ch, :]

        # create movie
        tmpdir = ipv.pylab.movie(f'{name}{perspective}.gif', go_through_volume, fps=fps, frames=N_frames, endpoint=False)

        # convert movie
        #command = f'convert -delay 5.0 -loop 0 {tmpdir}/frame-*.png {perspective}.gif'
        #stream = os.popen(command)
        #output = stream.read()
        

def make_box_for_grid(image_widget, title, size=(250,250)):
    """
    Make a VBox to hold caption/image for demonstrating
    option_fit values.
    """
    modes = ['oblique', 'X', 'Y', 'Z']

    # make layout
    box_layout = Layout()
    box_layout.width = f'{size[0]}px'
    box_layout.height = f'{size[1]}px'
    box_layout.border = '0px'

    # Make the caption
    if title is not None:
        title_str = "'{}'".format(title)
    else:
        title_str = str(title)

    h = HTML(value='' + str(title_str) + '')

    # Make the box with the image widget inside it
    boxb = Box()
    boxb.layout = box_layout
    boxb.children = [image_widget]

    # Compose into a vertical box
    vb = VBox()
    vb.layout.align_items = 'center'
    vb.children = [h, boxb]
    
    return vb


def open_image(filename='oblique.gif', size=(250,250)):
    
    file = open(filename , "rb")
    image = file.read()
    progress = Image(
        value=image,
        format='gif',
        width=size[0],
        height=size[1])
    return progress


def show_movies(name=''):
    
    modes = ['oblique', 'X', 'Y', 'Z']

    # define layout
    hbox_layout = Layout()
    hbox_layout.width = '100%'
    #hbox_layout.justify_content = 'space-around'

    # Use this margin to eliminate space between the image and the box
    image_margin = '0 0 0 0'

    # Set size of captions in figures below
    caption_size = 'h4'
    
    # open GIF files
    boxes = []
    for mode in modes:
        ib = open_image(filename=f'{name}{mode}.gif')
        ib.layout.object_fit = 'contain'
        ib.layout.margin = image_margin

        boxes.append(make_box_for_grid(ib, title=mode))

    # generate widgets
    vb = HBox()
    vb.layout.align_items = 'center'
    hb = HBox()
    hb.layout = hbox_layout
    hb.children = boxes

    vb.children = [hb]
    
    return vb

def visualize_volumes_for_slider(idx=0, axis=2, vminmax=None, alphaflag=False, **volumes):
    """PLot images in one row."""
    n = len(volumes)
    fig = plt.figure(figsize=(len(volumes)*7, 5))
    images = dict()
    for i, (name, item) in enumerate(volumes.items()):

        alpha = 1
        if not alphaflag:
            plt.subplot(1, n, i + 1)
            alpha = 1
        else:
            if i > 0:
                alpha = 0.2

        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')))

        volume_ = np.squeeze(item)
        slice_object = [slice(None), slice(None), slice(None)]
        slice_object[axis] = idx

        if volume_.dtype == np.bool:
            cmap = plt.get_cmap('Greys_r')
        else:
            cmap = plt.get_cmap('viridis')

        if vminmax:
            im = plt.imshow(volume_[slice_object], vmin=vminmax[0], vmax=vminmax[1],
                            alpha=alpha, cmap=cmap)
        else:
            im = plt.imshow(volume_[slice_object], alpha=alpha, cmap=cmap)
        images[name] = im

        plt.colorbar()

    return fig, images