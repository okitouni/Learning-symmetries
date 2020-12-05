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
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, f=10, ks=28, s=28, p=0, activation='relu', bias=True):
        super(LCN, self).__init__()
        width_span = int((w-ks+2*p)/s) + 1
        height_span = int((h-ks+2*p)/s) + 1
       # self.weight = nn.Parameter(
       #     (torch.rand(width_span*height_span*f, in_channels, ks, ks)*2-1)*sqrt(in_channels)
       # )
       # self.weight[width_span*height_span*(f-1):]*=2
       # if bias:
       #     self.bias = nn.Parameter(
       #        ( torch.rand(width_span*height_span*f, in_channels)*2-1)*sqrt(in_channels)
       #     )
        self.weight = nn.Parameter(
            torch.Tensor(width_span*height_span*f, in_channels, ks, ks)
        )
        if bias:
            self.bias = nn.Parameter(
               torch.Tensor(width_span*height_span*f, in_channels)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(ks)
        self.stride = _pair(s)
        self.activation = activation_func(activation)
        self.decoder = nn.Linear(width_span*height_span*f, out_channels)
        self.in_channels = in_channels
        self.f = f
        self.pad = p

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        _, c, h, w = x.size()
        x = nn.functional.pad(x,(self.pad,self.pad,self.pad,self.pad),'constant',0)
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh)
        x = x.unfold(3, kw, dw)
        x = x.reshape(x.size(0),-1,self.in_channels,kh,kw)
        x = x.repeat(1,self.f,1,1,1)
        x = (x * self.weight).sum([-1, -2])
        if self.bias is not None:
            x += self.bias
        x = self.activation(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return x
