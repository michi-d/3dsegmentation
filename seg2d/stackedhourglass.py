
__author__ = ['Michael Drews']

import numpy as np
from torch import nn
import torch

from .unet import Unet

class StackedUnet(nn.Module):
    """Stacked Unet Network
    """
    def __init__(self, num_stacks = 2, depth=5, start_channels=64, input_channels=1, batchnorm=True):

        super().__init__()
        self.num_stacks = num_stacks
        self.depth = depth
        self.start_channels = start_channels
        self.batchnorm = batchnorm
        self.input_channels = input_channels

        self._make_layers()

    def _make_hourglasses(self):
        """
        Makes the main hourglass submodules
        """
        hg = nn.ModuleList()
        for n in range(self.num_stacks):
            if n == 0:
                ch_in = self.input_channels
            else:
                ch_in = self.start_channels
            unet = Unet(depth=self.depth, start_channels=self.start_channels,
                        input_channels=ch_in, batchnorm=self.batchnorm)
            hg.append(unet)
        return hg

    def _make_fc(self, ch_in, ch_out):
        """
        Make fully connected layer via 1x1 convolution
        """
        modules = nn.ModuleList()
        modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=1))
        modules.append(nn.ReLU(inplace=True))
        if self.batchnorm:
            modules.append(nn.BatchNorm2d(ch_out))
        return nn.Sequential(*modules)

    def _make_glue(self):
        """
        Makes the intermediate stage which glues two Unets together.
        """
        ch_in = self.start_channels
        fc = nn.ModuleList()
        fc_ = nn.ModuleList()
        score = nn.ModuleList()
        score_ = nn.ModuleList()
        for n in range(self.num_stacks):
            fc.append(self._make_fc(ch_in, ch_in))
            fc_.append(self._make_fc(ch_in, ch_in))
            score.append(nn.Conv2d(ch_in, 1, kernel_size=1))
            score_.append(nn.Conv2d(1, ch_in, kernel_size=1))
        return fc, fc_, score, score_

    def _make_layers(self):
        """
        Generates full architecture.
        """
        # unet submodules
        self.hg = self._make_hourglasses()
        # intermediate stages
        self.fc, self.fc_, self.score, self.score_ = self._make_glue()
        # output layer
        #self.out = nn.Conv2d(self.start_channels, 1, kernel_size=1)

    def forward(self, x):
        out = []
        for n in range(self.num_stacks):
            y = self.hg[n].forward_unet(x) # pass through unet without output layer
            y = self.fc[n](y)

            # intermediate prediction
            score = self.score[n](y)
            out.append(score)

            # next module
            if n < self.num_stacks - 1:
                score_ = self.score_[n](score)
                fc_ = self.fc_[n](y)
                x = x + fc_ + score_

        return out
