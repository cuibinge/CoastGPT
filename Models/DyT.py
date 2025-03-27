import torch
import torch.nn as nn
class DyT(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(1)*0.5)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self,x):
        x=torch.tanh(self.alpha*x)
        return x*self.weight +self.bias