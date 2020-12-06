import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from math import sqrt
from torch.nn import init
from .. import utils
from collections.abc import Iterable
conv_output_shape = utils.conv_output_shape


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class LCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10, kernel_size=28, stride=28, padding=0, activation='relu',
                 bias=True, readout_activation=None, hidden=None):
        super().__init__()
        self.activation = activation_func(activation)
        self.readout_activation = readout_activation
        self.nfilters = nfilters
        if isinstance(nfilters, Iterable):
            convlayers = []
            for nfilters,channels in zip(nfilters,[in_channels,*nfilters]):
                convlayers.append(Conv2d_Local(channels, nfilters, kernel_size=kernel_size,h=h,w=w,
                    stride=stride, padding=padding, bias=bias))
                #convlayers.append(nn.BatchNorm2d(nfilters))
                convlayers.append(self.activation)
                h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)
            self.conv_blocks = nn.Sequential(*convlayers)
        else:
            self.conv_blocks = nn.Sequential(Conv2d_Local(in_channels, nfilters,h=h,w=w, 
                kernel_size=kernel_size,stride=stride, padding=0, bias=True),
                #nn.BatchNorm2d(nfilters),
                self.activation)
            h,w = conv_output_shape(h_w=(h, w), kernel_size=kernel_size, stride=stride,padding=padding)

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
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        if self.readout_activation is not None:
            x = self.readout_activation(x)
        return x


class Conv2d_Local(nn.Module):
    def __init__(self, in_channels=1, nfilters=10, h=280, w=280, kernel_size=28, stride=28, padding=0, bias=True):
        super().__init__()
        self.height_span, self.width_span = conv_output_shape(
            h_w=(h, w), kernel_size=kernel_size, stride=stride)
        self.weight = nn.Parameter(
            torch.Tensor(self.width_span*self.height_span*nfilters,
                         in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.width_span*self.height_span *
                             nfilters, in_channels)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.in_channels = in_channels
        self.nfilters = nfilters
        self.pad = padding
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        _, c, h, w = x.size()
        x = nn.functional.pad(
            x, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh)
        x = x.unfold(3, kw, dw)
        x = x.view(x.size(0), -1, self.in_channels, kh, kw)
        x = x.repeat(1, self.nfilters, 1, 1, 1)
        x = (x * self.weight).sum([-1, -2])
        if self.bias is not None:
            x += self.bias
        x = x.view(-1, self.nfilters, self.height_span, self.width_span)
        return x
