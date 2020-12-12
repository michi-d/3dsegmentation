
__author__ = ['Michael Drews']

import numpy as np
from torch import nn
import torch
import numbers


class VGG(nn.Module):
    """
    My parametrized VGG-like implementation
    """
    __default_cnn__ = [64, 64, 'M', 64, 'M']
    __default_head__ = []

    def __init__(self, cnn_cfgs=__default_cnn__, head_cfgs=__default_head__, output_units=256,
                 avg_pool_size=7, batchnorm=True):
        """
        Creates the network.
        
        Args:
            cnn_cfgs (List): layer configuration for the ConvNet
            head_cfgs (List): layer configuration for the output head
            output_units: number of output units
            avg_pool_size: square lenght of the AvgPool layer
            batchnorm: include BatchNorm layers or not
        """
        
        super(VGG, self).__init__()
        self.cnn_cfgs = cnn_cfgs
        self.head_cfgs = head_cfgs
        self.batchnorm = batchnorm
        self.output_units = int(output_units)

        self.conv_net, conv_output_channels = self._make_conv_layers(self.cnn_cfgs, batchnorm=self.batchnorm)
        self.avgpool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))
        self.flatten = nn.Flatten()

        h_cfgs = self.head_cfgs + [self.output_units]
        head_input_channels = (avg_pool_size**2)*conv_output_channels
        self.head = self._make_head(h_cfgs, head_input_channels, batchnorm=self.batchnorm)

    def _make_conv_layers(self, cnn_cfgs, batchnorm=True):

        modules = nn.ModuleList()
        for i, c in enumerate(cnn_cfgs):

            if isinstance(c, numbers.Number):
                if i==0:
                    in_channel = 1
                out_channel = int(c)
                modules.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
                modules.append(nn.ReLU(inplace=True))
                if batchnorm:
                    modules.append(nn.BatchNorm2d(out_channel))
                in_channel = out_channel

            elif isinstance(c, str):
                modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        conv_net = nn.Sequential(*modules)
        return conv_net, out_channel

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
        x = self.conv_net(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
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

