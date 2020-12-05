import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from math import sqrt
from torch.nn import init

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class LCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10, kernel_size=28, stride=28, pad=0, activation='relu', bias=True, readout_activation=None):
        super().__init__()
        width_span = int((w-kernel_size+2*pad)/stride) + 1
        height_span = int((h-kernel_size+2*pad)/stride) + 1
       # self.weight = nn.Parameter(
       #     (torch.rand(width_span*height_span*f, in_channels, kernel_size, kernel_size)*2-1)*sqrt(in_channels)
       # )
       # self.weight[width_span*height_span*(f-1):]*=2
       # if bias:
       #     self.bias = nn.Parameter(
       #        ( torch.rand(width_span*height_span*f, in_channels)*2-1)*sqrt(in_channels)
       #     )
        self.weight = nn.Parameter(
            torch.Tensor(width_span*height_span*nfilters, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
               torch.Tensor(width_span*height_span*nfilters, in_channels)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.activation = activation_func(activation)
        self.decoder = nn.Linear(width_span*height_span*nfilters, out_channels)
        self.in_channels = in_channels
        self.nfilters = nfilters
        self.pad = pad
        self.readout_activation = readout_activation
        init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        _, c, h, w = x.size()
        x = nn.functional.pad(x,(self.pad,self.pad,self.pad,self.pad),'constant',0)
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh)
        x = x.unfold(3, kw, dw)
        x = x.reshape(x.size(0),-1,self.in_channels,kh,kw)
        x = x.repeat(1,self.nfilters,1,1,1)
        x = (x * self.weight).sum([-1, -2])
        if self.bias is not None:
            x += self.bias
        x = self.activation(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        if self.readout_activation is not None: 
            x = self.readout_activation(x)
        return x
