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


class FCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10,hidden=None,
                 kernel_size=28, stride=1, activation='relu', readout_activation=None,padding=0,bias=True, *args, **kwargs):
        super().__init__()
        self.activation = activation_func(activation)
        self.readout_activation = readout_activation
        self.nfilters = nfilters
        self.out_channels = out_channels 
        if isinstance(nfilters, Iterable):
            mainlayers = []
            for nfilters,channels in zip(nfilters,[in_channels*h*w,*nfilters]):
                mainlayers.append(nn.Linear(channels, nfilters)) 
                #mainlayers.append(nn.BatchNorm2d(nfilters))
                mainlayers.append(self.activation)
                # Would be used if truly wanted FCN embedding of a CNN
                #h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)
            self.main_blocks = nn.Sequential(*mainlayers)
        else:
            self.main_blocks = nn.Sequential(nn.Linear(in_channels*h*w, nfilters), 
                #nn.BatchNorm2d(nfilters),
                self.activation)
#            h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)

        if hidden is not None:
            if not isinstance(hidden, Iterable): hidden = [hidden]
            hidden = [nfilters, *hidden]
            layers = [] 
            for i in range(len(hidden)-1):
                layers.append(nn.Linear(hidden[i], hidden[i+1]))
                layers.append(self.activation)
            self.decoder = nn.Sequential(*layers, nn.Linear(hidden[-1], out_channels))
        else:
            self.decoder = nn.Linear(h*w*nfilters, out_channels)

    def forward(self, x):
        x = x.flatten(1)
        x = self.main_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        if self.readout_activation is not None:
            x = self.readout_activation(x)
        return x
