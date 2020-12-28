# 3dsegmentation

A toolkit for 3D segmentation of anatomical data from the fly brain. This repository contains a module for generation of artificial neuroimaging data with annotated ground truth labels, as well as a pipeline for training different deep learning models (U-Net, Stacked Hourglass Network) on the 3D segmentation task, implemented in PyTorch.

## Installation

To get the code running, please set up a python virtual environment and install all dependencies as follows:

```
virtualenv segmentation_env
source bin/activate/segmentation_env
pip install -r requirements.txt
```

## Artificial Data Generation

This respository contains functions for generating both 3D and 2D data (for test purposes).
In the 3D case, data generation is based on a 3D model of the fly optic lobe with visual columns distributed hexagonally over a randomly positioned and scaled cut-out of an ellipsoid 3D surface. Neurites are approximated as Bezier curves of third grade from each column to the center of the ellipsoid. 

After establishing a basic and random geometry for each sample, the artifical imaging volume is rendered by distributing "fluorescence clusters" randomly as a function of the distance of the center for each column. To induce more variance, three different types of probability densities can govern the distribution for each column. Randomly chosen columns are left out from rendering to simulate stochastic underexpression of fluorescence markers in the biological substrate.

Binary ground truth segmentation masks are generated as 3D elongated cones around the central axis for each column. The orientation, curvature and position, i.e. the geometric arrangement of the simulated fly optic lobe, are randomized for each sample.

### 3D-Data

To generate a training set of 2048 samples and a validation set of 64 samples please run the data generator script:

`./gen_data.sh`

This will save training and validation data into five different HDF5-files located in the sub-folder **./vol_data**.

The following animation visualizes one training sample as an image stack as we move along the z-axis:

<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/slice_demo_3.gif" alt="drawing" width="500"/>

The following two animations show the above image stack visualized in 3D (left) and another sample from the training set (right), with simulated fluorescence data in blue and segmentation masks in red:
<p float="left">
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_3.gif" alt="drawing" width="200"/>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img src="https://github.com/michi-d/3dsegmentation/blob/main/assets/3d_demo_20.gif" alt="drawing" width="200"/>
</p>

### 2D-Data

2D data can be generated on-the-fly by creating a `SegmentationFake2DDataset` object in Python-code as follows:

```python
from utils.data import SegmentationFake2DDataset
train_dataset = SegmentationFake2DDataset(L=SIZE, seed=SEED, h5path=FILE_PATH)
```

where `SEED` the the seed for the random number generator and `SIZE` the number of samples. If `FILE_PATH` is set to `None` the dataset will not be saved to disk.


## Model Training

### 3D Segmentation

Currently, there are two model architectures implemented:

* **U-net**: The original U-Net architecture (with one input channel) (https://arxiv.org/abs/1505.04597).
* **Stacked Hourglass Network**: A stack of several U-Net's with intermediate supervision and intermediate layers implemented according to the original paper of this architecture (https://arxiv.org/abs/1603.06937).

To train a model, first generate a training and validation dataset in the folder **./vol_data** (see above), and then run the command

`python train_basic_3dunet.py --options`

The following options are available:

` --experiment_title STR`: Name for the experiment \
` --log_path STR`: Parent directory for the log files \
` --checkpoint_policy (last/best/all)`: Which checkpoints to save during training 

` --batch_size INT`: Batch size \
` --epochs INT`: Maximum number of epochs to train \
` --early_stopping_patience INT`: Training will stop if validation loss does not decrease after this number of epochs. 

` --arch (unet/stacked_unet)`: Model architecture (U-Net / Stacked Hourglass Network based on U-Net) \
` --depth INT`: Number of layers in each half of the U-Net \
` --start_channels INT`: Number of filters in first layer \
` --num_stacks INT`: Number of U-Nets to stack for the Stacked Hourglass Network \
` --conv_kernel_size INT`: Kernel size of the 3D convolutional layers in the network 

` --optimizer (adam/rmsprop)`: Optimizer algorithm \
` --loss_type (dice/bce/weighted_bce)`: Loss function (Dice loss / Binary cross-entropy / Label-balanced binary cross-entropy) \
` --weight_decay FLOAT`: Weight decay parameter for the optimizer 

` --start_lr FLOAT`: Learning rate at the beginning of training \
` --lr_scheduler_factor FLOAT`: Multiplicative factor for the learning rate scheduler \
` --lr_scheduler_patience INT`: Number of epochs to wait before decreasing the learning rate 

` --MAX_PARAMS INT`: Training will not start if parameter count of the model exceeds this limit. \
` --MIN_PARAMS INT`: Training will not start if parameter count of the model is below this limit. 

Standard values for these parameters can be found in the script `train_basic_3dunet.py`.

### 2D Segmentation

Similar to above, 2D models can be trained using the script `train_basic_2dunet.py` with options examinable at the top of file.
