__author__ = ['Michael Drews']

import numpy as np
from torch import nn
import torch


class Unet(nn.Module):
    """
    My UNet implementation
    """

    def __init__(self, depth=5, input_channels=1, start_channels=64, conv_kernel_size=3,
                 batchnorm=True, resblocks=False):
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
        self.resblocks = bool(resblocks)

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

            conv_block = self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm, conv_kernel_size=self.conv_kernel_size)
            if self.resblocks:
                conv_block = ResidualBlock(ch_in, ch_out, conv_block, batchnorm=self.batchnorm)
            self.down_path.append(conv_block)

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
            conv_block = self._dual_conv(ch_in, ch_out, batchnorm=self.batchnorm, conv_kernel_size=self.conv_kernel_size)
            if self.resblocks:
                conv_block = ResidualBlock(ch_in, ch_out, conv_block, batchnorm=self.batchnorm)
            self.up_path_trans.append(trans)
            self.up_path_conv.append(conv_block)

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
        """
        Forward pass through  down- and upsample path WITHOUT output layer.
        """
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
        """
        Forward pass WITH output layer
        """
        x = self.forward_unet(x)
        # pass through output layer
        x = self.out(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block around a given processing block
    """

    def __init__(self, in_channels, out_channels, block, batchnorm=True):

        super().__init__()
        self.batchnorm = batchnorm
        self.in_channels, self.out_channels = in_channels, out_channels

        # build processing block
        self.block = block

        # build final activation function
        self.activate = nn.ReLU(inplace=True)

        # build shortcut connection (with 1x1 kernel in case of channel mismatch)
        self.shortcut = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.out_channels)
        ) if self.should_apply_shortcut else None

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels