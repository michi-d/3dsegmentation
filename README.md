# 3dsegmentation

A toolkit for 3D segmentation of anatomical data from the fly brain. This repository contains a module for generation of artificial neuroimaging data with annotated ground truth labels, as well as a pipeline for training different deep learning models (U-Net, Stacked Hourglass Network) on the 3D segmentation task, implemented in PyTorch.

## Installation

To get the code running, please set up a python virtual environment and install all dependencies as follows:

`virtualenv segmentation_env` \
`source bin/activate/segmentation_env` \
`pip install -r requirements.txt`

## Training

Currently, there are two model architectures implemented:

* **U-net**: The original U-Net architecture (with one input channel) (https://arxiv.org/abs/1505.04597).
* **Stacked Hourglass**: A stacked version of U-Net with intermediate supervision, according to the Stacked Hourglass Network (https://arxiv.org/abs/1603.06937).

To train one model, first generate the training dataset (see below), and then run the following command:

`python train_basic_3dunet.py --arch ARCHITECTURE`

## Artificial Data Generation

This respository contains functions for generating both 3D and 2D data, for test purposes.
In the 3D case, data generation is based on a 3D model of the fly optic lobe with visual columns distributed hexagonally over a randomly positioned and scaled cut-out of an ellipsoid 3D surface. Neurites are approximated as Bezier curves of third grade from each column to the center of the ellipsoid. 

After establishing a basic and random geometry for each sample, the artifical imaging volume is rendered by distributing "fluorescence clusters" randomly as a function of the distance of the center for each column. To induce more variance, three different types of probability densities can govern the distribution for each column. Randomly, columns are also left out from rendering to simulate stochastic underexpression of fluorescence markers in the biological substrate.

Binary ground truth segmentation masks are generated as 3D elongated cones around the central axis for each column. The orientation, curvature and position are randomized for each sample.

### 3D-Data

To generate a full training set of 2048 samples and a validation set of 64 samples please run 

`./gen_data.sh`

The data is then saved into five different HDF5-files in the folder **./vol_data**.

The following animation visualizes one training sample as an image stack as we move along the z-axis:
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/slice_demo_3.gif" alt="drawing" width="500"/>

The following two animations show the above image stack visualized in 3D (left) and another sample from the training set (right):
<p float="left">
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_3.gif" alt="drawing" width="200"/>
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_20.gif" alt="drawing" width="200"/>
</p>

### 2D-Data

2D test data can be generated online by generating a `SegmentationFake2DDataset` object in Python-code as follows:

`from utils.data import SegmentationFake2DDataset` \
`train_dataset = SegmentationFake2DDataset(L=SIZE, seed=SEED, h5path=FILE_PATH)`

where `SEED` the the seed for the random number generator and `SIZE` the number of samples. If `FILE_PATH` is set to `None` the dataset will not be saved to disk.



