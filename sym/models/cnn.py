import torch
import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10, ks=28, activation='relu', *args, **kwargs):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, nfilters, kernel_size=ks,
                      stride=ks, padding=0, bias=True),
            nn.BatchNorm2d(f),
            activation_func(activation)
        )

        self.decoder = nn.Linear(h*w//ks**2*nfilters, out_channels)

    def forward(self, x):
        x = self.conv_block1(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
