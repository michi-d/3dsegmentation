
__author__ = ['Michael Drews']

import numpy as np
from torch import nn
import torch


class Unet(nn.Module):
    """
    My UNet implementation
    """
    
    def __init__(self, depth=5, start_channels=64, batchnorm=True):
        """
        Creates the network.
        
        Args:
            depth: number of dual convolutional layers
            start_channels: number of output channels in first layer
        """
        
        super(Unet, self).__init__()
        self.depth = depth
        self.start_channels = start_channels
        self.batchnorm = batchnorm

        # generate network
        self._make_layers()

    def _make_layers(self):

        # create downsample path
        self.down_path = nn.ModuleList()
        for n in range(self.depth):
            if n == 0:
                ch_in = 1
            else:
                ch_in  = self.start_channels * (2**(n-1))
            ch_out = self.start_channels * (2**n)
            
            self.down_path.append(self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm))
        
        # create maxpool operation
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        # create upsample path
        self.up_path_trans = nn.ModuleList()
        self.up_path_conv = nn.ModuleList()
        for n in range(self.depth)[::-1]:
            if n == 0:
                ch_out = 1
            else:
                ch_out  = self.start_channels * (2**(n-1))
            ch_in = self.start_channels * (2**n)
            
            trans = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            conv = self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm)
            self.up_path_trans.append(trans)
            self.up_path_conv.append(conv)   
            
        # create output layer
        self.out = nn.Conv2d(ch_in, 1, kernel_size=1)

    @staticmethod
    def _dual_conv(in_channel, out_channel, batchnorm=True):
        """
        Returns a dual convolutional layer with ReLU activations in between. 
        """
        if batchnorm:
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel),

                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel)
            )
        else:
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        return conv

    def forward(self, x):
        "Forward pass through the network"
        
        # pass through downsample path
        self.feature_maps = []
        for n in range(self.depth):
            down_conv = self.down_path[n]
            x = down_conv(x)
            if n < self.depth-1:
                self.feature_maps.append(x)
                x = self.maxpool(x)

        # pass through upsample path
        for n in range(self.depth-1):
            trans = self.up_path_trans[n]
            conv = self.up_path_conv[n]
            
            x = trans(x)
            y = self.feature_maps[-(n+1)]
            x = conv(torch.cat([x,y], 1))
            
        # pass through output layer
        x = self.out(x)
        
        return x

    def count_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def get_gradients(self):
        gradients = dict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                gradients[name] = param.grad.numpy()


class RegressionUnet(Unet):
    """
    Extension of the standard UNet for regression appending a dense network as a prediction head.
    Conv1d and AvgPool layers are used between to reduce the dimensionality of the UNet output.
    """
    __default_head__ = [256]

    def __init__(self, unet_output_channels=4, avg_pool_size=32, head_cfgs=__default_head__,
                 **kwargs):
        """
        Args:
            unet_output_channels: number of output channels for the Unet
            avg_pool_size: square length of the AvgPool layer

        """
        super().__init__(**kwargs)

        self.unet_output_channels = unet_output_channels
        self.avg_pool_size = avg_pool_size
        self.head_cfgs = head_cfgs

        # downsample layer for channel dimension
        self.conv1d = nn.Sequential(
            nn.Conv2d(self.start_channels, self.unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.unet_output_channels)
        )

        # downsample layer for spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((self.avg_pool_size, self.avg_pool_size))
        self.flatten = nn.Flatten()

        # create head network
        input_channels = (self.avg_pool_size**2) * self.unet_output_channels
        self.head = self._make_head(self.head_cfgs, input_channels)

    def _make_head(self, head_cfgs, input_channels, batchnorm=True):

        n = len(head_cfgs)
        modules = nn.ModuleList()
        for i, c in enumerate(head_cfgs):
            if i==0:
                in_channel = input_channels
            out_channel = int(c)
            modules.append(nn.Linear(in_channel, out_channel))
            if i < n-1:
                modules.append(nn.ReLU(inplace=True))
                if batchnorm:
                    modules.append(nn.BatchNorm1d(out_channel))

            in_channel = out_channel

        head = nn.Sequential(*modules)
        return head

    def forward(self, x):
        "Forward pass through the network"

        # pass through downsample path
        self.feature_maps = []
        for n in range(self.depth):
            down_conv = self.down_path[n]
            x = down_conv(x)
            if n < self.depth - 1:
                self.feature_maps.append(x)
                x = self.maxpool(x)

        # pass through upsample path
        for n in range(self.depth - 1):
            trans = self.up_path_trans[n]
            conv = self.up_path_conv[n]

            x = trans(x)
            y = self.feature_maps[-(n + 1)]
            x = conv(torch.cat([x, y], 1))

        # sample down output from UNet along channel and spatial dimensions
        x = self.conv1d(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        # regression head
        x = self.head(x)

        return x