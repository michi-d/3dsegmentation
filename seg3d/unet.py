__author__ = ['Michael Drews']

import numpy as np
from torch import nn
import torch


class Unet(nn.Module):
    """
    My UNet implementation
    """

    def __init__(self, depth=5, input_channels=1, start_channels=64, conv_kernel_size=3, batchnorm=True):
        """
        Creates the network.

        Args:
            depth: number of dual convolutional layers
            start_channels: number of output channels in first layer
        """

        super().__init__()
        self.depth = depth
        self.start_channels = start_channels
        self.batchnorm = batchnorm
        self.input_channels = input_channels
        self.conv_kernel_size = conv_kernel_size

        # generate network
        self._make_layers()

    def _make_layers(self):

        # create downsample path
        self.down_path = nn.ModuleList()
        for n in range(self.depth):
            if n == 0:
                ch_in = self.input_channels
            else:
                ch_in = self.start_channels * (2 ** (n - 1))
            ch_out = self.start_channels * (2 ** n)

            self.down_path.append(self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm, conv_kernel_size=self.conv_kernel_size))

        # create maxpool operation
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # create upsample path
        self.up_path_trans = nn.ModuleList()
        self.up_path_conv = nn.ModuleList()
        for n in range(self.depth)[::-1]:
            if n == 0:
                ch_out = self.input_channels
            else:
                ch_out = self.start_channels * (2 ** (n - 1))
            ch_in = self.start_channels * (2 ** n)

            trans = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2)
            conv = self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm, conv_kernel_size=self.conv_kernel_size)
            self.up_path_trans.append(trans)
            self.up_path_conv.append(conv)

            # create output layer
        self.out = nn.Conv3d(ch_in, 1, kernel_size=1)

    @staticmethod
    def _dual_conv(in_channel, out_channel, batchnorm=True, conv_kernel_size=3):
        """
        Returns a dual convolutional layer with ReLU activations in between.
        """
        padding = conv_kernel_size // 2
        if batchnorm:
            conv = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=conv_kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channel),

                nn.Conv3d(out_channel, out_channel, kernel_size=conv_kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channel)
            )
        else:
            conv = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=conv_kernel_size, padding=padding),
                nn.ReLU(inplace=True),

                nn.Conv3d(out_channel, out_channel, kernel_size=conv_kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            )
        return conv

    def forward_unet(self, x):
        """Forward pass through  down- and upsample path"""

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
        return x

    def forward(self, x):
        x = self.forward_unet(x)
        # pass through output layer
        x = self.out(x)
        return x
