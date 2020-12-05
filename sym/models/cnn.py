import torch
import torch.nn as nn
from .. import utils
conv_output_shape = utils.conv_output_shape

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10,
                 kernel_size=28, stride=1, activation='relu', readout_activation=None, *args, **kwargs):
        super().__init__()
        outdim = conv_output_shape(
            h_w=(h, w), kernel_size=kernel_size, stride=stride)
        outdim = outdim[0]*outdim[1]

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, nfilters, kernel_size=kernel_size,
                      stride=stride, padding=0, bias=True),
            #            nn.BatchNorm2d(nfilters),
            activation_func(activation)
        )

        self.decoder = nn.Linear(outdim*nfilters, out_channels)
        self.readout_activation = readout_activation

    def forward(self, x):
        x = self.conv_block1(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        if self.readout_activation is not None:
            x = self.readout_activation(x)
        return x
