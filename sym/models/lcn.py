import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class LCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10, ks=28, activation='relu', bias=True):
        super(LCN, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(h*w//(ks**2)*nfilters, in_channels, ks, ks)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(h*w//(ks**2)*nfilters, in_channels)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(ks)
        self.stride = _pair(ks)
        self.activation = activation_func(activation)
        self.decoder = nn.Linear(h*w//ks**2*nfilters, out_channels)
        self.in_channels = in_channels
        self.nfilters = nfilters

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw).reshape(
            x.size(0), -1, self.in_channels, kh, kw)
        x = x.repeat(1, self.nfilters, 1, 1, 1)
        x = (x * self.weight).sum([-1, -2])
        if self.bias is not None:
            x += self.bias
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
