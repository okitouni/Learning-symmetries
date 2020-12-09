import torch
import torch.nn as nn
from .. import utils
from collections.abc import Iterable
conv_output_shape = utils.conv_output_shape


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10,hidden=None,
                 kernel_size=28, stride=1, activation='relu', readout_activation=None,
                 padding=0,bias=True, invar_reduction = None, *args, **kwargs):
        super().__init__()
        self.activation = activation_func(activation)
        self.readout_activation = readout_activation
        self.nfilters = nfilters
        self.out_channels = out_channels 
        self.invar_reduction = invar_reduction 
        if isinstance(nfilters, Iterable):
            convlayers = []
            for nfilters,channels in zip(nfilters,[in_channels,*nfilters]):
                convlayers.append(nn.Conv2d(channels, nfilters, kernel_size=kernel_size,
                    stride=stride, padding=padding, bias=bias))
                #convlayers.append(nn.BatchNorm2d(nfilters))
                convlayers.append(self.activation)
                h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)
            self.conv_blocks = nn.Sequential(*convlayers)
        else:
            self.conv_blocks = nn.Sequential(nn.Conv2d(in_channels, nfilters, 
                kernel_size=kernel_size,stride=stride, padding=0, bias=True),
                #nn.BatchNorm2d(nfilters),
                self.activation)
            h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)

        if invar_reduction == "max":
            self.reduction = torch.nn.AdaptiveMaxPool2d((1,1))
            h,w = 1,1
        elif invar_reduction == "mean":
            self.reduction = torch.nn.AdaptiveAvgPool2d((1,1))
            h,w = 1,1

        if hidden is not None:
            if not isinstance(hidden, Iterable): hidden = [hidden]
            hidden = [h*w*nfilters, *hidden]
            layers = [] 
            for i in range(len(hidden)-1):
                layers.append(nn.Linear(hidden[i], hidden[i+1]))
                layers.append(self.activation)
            self.decoder = nn.Sequential(*layers, nn.Linear(hidden[-1], out_channels))
        else:
            self.decoder = nn.Linear(h*w*nfilters, out_channels)

    def forward(self, x):
        x = self.conv_blocks(x)
        if self.invar_reduction is not None:
            x = self.reduction(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        if self.readout_activation is not None:
            x = self.readout_activation(x)
        return x
