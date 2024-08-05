from typing import Optional

import torch
from torch import nn, Tensor


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0    


class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0    


class Globalpooling(nn.Module):
    def __init__(self, PoolType='mean', KeepDim=False):
        super(Globalpooling, self).__init__()
        self.PoolType = PoolType
        self.KeepDim = KeepDim
        
    def globalPool(self, x):
        assert x.dim() == 4, "Got: {}".format(x.shape)
        if self.PoolType == 'rms':
            x = x ** 2
            x = torch.mean(x, dim=[-2, -1], keepdim=self.KeepDim)
            x = x ** -0.5
        elif self.PoolType == 'abs':
            x = torch.mean(x, dim=[-2, -1], keepdim=self.KeepDim)
        else:
            # same as AdaptiveAvgPool
            x = torch.mean(torch.abs(x), dim=[-2, -1], keepdim=self.KeepDim)# use default method "mean"
        
        return x
        
    def forward(self, x: Tensor) -> Tensor:
        return self.globalPool(x)
    
    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0
    