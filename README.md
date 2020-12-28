# 3dsegmentation

A toolkit for 3D segmentation of anatomical data from the fly brain. This repository comprises a module for generation of artifical neuro-imaging data with  annotated ground truth labels, as well as a pipeline for training deep learning models (U-Net, Stacked Hourglass Network) on the 3D segmentation task. 

## Installation

To get the code running, please set up a python virtual environment and install all dependencies as follows:

`virtualenv segmentation_env`
`source bin/activate/segmentation_env`
`pip install -r requirements.txt`

## Data generation

This respository contains functions for generating both 3D and 2D data, for test purposes.
In the 3D case, data generation is based on a 3D model of the fly optic lobe with visual columns distributed hexagonally over a randomly positioned and scaled cut-out of an ellipsoid 3D surface. Neurites are approximated as Bezier curves of 3rd grade from each column to the center of the ellipsoid. 

After establishing a basic and random geometry for each sample, the artifical imaging volume is rendered by distributing "fluorescence clusters" randomly as a function of the distance of the center for each column. To induce more variance, three different types of probability densities can govern the distribution for each column. Randomly, columns are also left out from rendering to simulate stochastic underexpression of fluorescence markers in the biological substrate.

Binary ground truth segmentation masks are generated as 3D elongated cones around the central axis for each column. The orientation, curvature and position are randomized for each sample.

### 3D 

To generate a full training set of 2048 samples and a validation set of 64 samples please run 

`./gen_data.sh`

The data is then saved in HDF5-format in the folder `vol_data`.

To customize dataset size run:

`python gen_data.py --filename FILENAME --directory DIRECTORY --L SIZE --random_state SEED`

where `DIRECTORY` defines the parent directory, `SIZE` the number of samples and `SEED` the random generator seed.

The following animation visualizes the image stack as we go along the z-axis:

<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/slice_demo_3.gif" alt="drawing" width="600"/>

The following two animations show the above image stack visualized in 3D (left) and another sample from the training set (right).

<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_3.gif" alt="drawing" width="600"/>
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_20.gif" alt="drawing" width="600"/>

### 2D

2D test data can be generated online by generating a `SegmentationFake2DDataset` object in Python-code as follows:

`from utils.data import SegmentationFake2DDataset`
`train_dataset = SegmentationFake2DDataset(L=SIZE, seed=SEED, h5path=FILE_PATH)`

If `FILE_PATH` is set to `None` the dataset will not be saved to disk.



