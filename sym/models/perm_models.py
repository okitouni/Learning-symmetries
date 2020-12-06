import numpy as np
import torch
import torch.nn as nn

class PIN1(nn.Module):
    
    def __init__(self, N, width=10, ):
        super(PIN1, self).__init__()
        self.phi = nn.Sequential(nn.Linear(1,N),
                                 nn.ReLU())
                                 #nn.Linear(width,N+1),
                                 #nn.ReLU())
        
        self.rho1 = nn.Linear(N,1)#width)
        self.rho2 = nn.Linear(width,1)
        self.N = N

    def forward(self, x):
        W = torch.zeros((x.size(0),self.N))
        for i in range(len(x[0])):
            W += self.phi(x[:,i].view(-1,1))
        
        x = self.rho1(W)
        #x = nn.ReLU()(x)
        #x = self.rho2(x)
        return x
    
class SNN1(nn.Module):
    def __init__(self, N, width=10):
        super(SNN1,self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(N,N),
                                       #torch.nn.ReLU(),
                                       #torch.nn.Linear(width,N+1),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(N,1))#width),
                                       #torch.nn.ReLU(),
                                       #torch.nn.Linear(width,1))
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
