import numpy as np
from typing import Any, Callable

import torch
from torch import nn, Tensor

from ..layers import BaseConv2d, AdaptiveAvgPool2d
from ...utils import shuffleTensor, setMethod, callMethod, makeDivisible


class MoCAttention(nn.Module):
    # Monte carlo attention
    def __init__(
        self,
        InChannels: int,
        HidChannels: int=None,
        SqueezeFactor: int=4,
        PoolRes: list=[1, 2, 3],
        Act: Callable[..., nn.Module]=nn.ReLU,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        MoCOrder: bool=True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)
            
        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
        
    def monteCarloSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None] # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)
            
        return AttnMap
        
    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.monteCarloSample(x)
        return x * self.SELayer(AttnMap)
    