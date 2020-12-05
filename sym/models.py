import numpy as np
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
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, f=10, ks=28, s=28, p=0, activation='relu', bias=True):
        super(LCN, self).__init__()
        width_span = int((w-ks+2*p)/s) + 1
        height_span = int((h-ks+2*p)/s) + 1
        self.weight = nn.Parameter(
            torch.ones(width_span*height_span*f, in_channels, ks, ks)
        )
        self.weight[width_span*height_span*(f-1):]*=2
        if bias:
            self.bias = nn.Parameter(
                torch.ones(width_span*height_span*f, in_channels)
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

class PIN1(nn.Module):
    
    def __init__(self, N, width=10):
        super(PIN1, self).__init__()  
        self.phi = nn.Sequential(nn.Linear(1,width),
                                 nn.ReLU(),
                                 nn.Linear(width,N+1),
                                 nn.ReLU())
        
        self.rho1 = nn.Linear(N+1,width)
        self.rho2 = nn.Linear(width,1)
        self.N = N

    def forward(self, x):
        W = torch.zeros((x.size(0),self.N+1))
        for i in range(len(x[0])):
            W += self.phi(x[:,i].view(-1,1))
        
        x = self.rho1(W)
        x = nn.ReLU()(x)
        x = self.rho2(x)
        return x
    
class SNN1(nn.Module):
    def __init__(self, N, width=10):
        super(SNN1,self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(N,width),
																			 torch.nn.ReLU(),
																			 torch.nn.Linear(width,N+1),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(N+1,width),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(width,1))
    def forward(self,x):
        x = self.mlp(x)
        return x
    
class PIN2(nn.Module):
    
    def __init__(self, N, depth=3):
        super(PIN2, self).__init__()
        self.eqvL = nn.Parameter( torch.abs(torch.randn(depth)) )
        self.eqvG = nn.Parameter( torch.abs(torch.randn(depth)) )
        self.N = N
        self.depth = depth
        self.linear = nn.Linear(N,1)
        self.pin1 = PIN1(N)

    def forward(self, x):
        for i in range(self.depth):
            L = self.eqvL[i]*torch.eye(self.N)
            G = self.eqvG[i]*torch.ones((self.N,self.N))
            x = torch.matmul(x,L+G)
            x = torch.nn.ReLU()(x)
        out = self.pin1(x)#torch.sum(x,1)
        return out
    
class SNN2(nn.Module):
    def __init__(self, N, depth=3):
        super(SNN2,self).__init__()
        self.layerList = []
        for i in range(depth):
            self.layerList.append(nn.Sequential(nn.Linear(N,N),
                                                nn.ReLU()))
        self.snn1 = SNN1(N)
        
    def forward(self,x):
        for layer in self.layerList:
            x = layer(x)
        out = self.snn1(x)
        return out
    
class PIN3(nn.Module):
    
    def __init__(self, N):
        super(PIN3, self).__init__()
        self.params = nn.Parameter( torch.randn(N) )
        self.N = N
        self.pin1 = PIN1(np.math.factorial(N))
        
    def forward(self,x):
        W = torch.zeros(np.math.factorial(self.N),self.N)
        for i,perm in enumerate(list(permutations(range(self.N)))):
            perm = torch.LongTensor(perm)
            W[i] = self.params[perm]
        print(W)
        x = torch.matmul(x,torch.transpose(W,0,1))
        out = self.pin1(x)
        return out
    
class SNN3(nn.Module):
    
    def __init__(self, N):
        super(SNN3, self).__init__()
        self.layer1 = nn.Linear(N,np.math.factorial(N))
        self.pin1 = PIN1(np.math.factorial(N))
        
    def forward(self,x):
        x = self.layer1(x)
        out = self.pin1(x)
        return out


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
