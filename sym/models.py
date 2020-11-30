import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.modules.utils import _pair
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]

class LCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, f=10, ks=28, activation='relu', bias=True):
        super(LCN, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(h*w//(ks**2)*f, in_channels, ks, ks)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(h*w//(ks**2)*f, in_channels)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(ks)
        self.stride = _pair(ks)
        self.activation = activation_func(activation)
        self.decoder = nn.Linear(h*w//ks**2*f, out_channels)
        self.in_channels = in_channels
        self.f = f

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw).reshape(x.size(0),-1,self.in_channels,kh,kw)
        x = x.repeat(1,self.f,1,1,1)
        x = (x * self.weight).sum([-1, -2])
        if self.bias is not None:
            x += self.bias
        x = self.activation(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return x



class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h = 280, w = 280, f = 10, ks = 28, activation='relu', *args, **kwargs):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
					nn.Conv2d(in_channels, f, kernel_size = ks, stride = ks, padding = 0, bias=True),
					nn.BatchNorm2d(f),
					activation_func(activation)
        )
        
        self.decoder = nn.Linear(h*w//ks**2*f, out_channels)

    def forward(self, x):
        x = self.conv_block1(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return x

class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.Loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self, learning_rate=1e-2):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(trainset, [
                                                                             55000, 5000])
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = testset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, num_workers=6, batch_size=320)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, num_workers=6, batch_size=320)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, num_workers=6, batch_size=320)
