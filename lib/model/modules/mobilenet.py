from typing import Any, Callable

from torch import nn, Tensor

from ..layers import BaseConv2d, AdaptiveAvgPool2d
from ...utils import makeDivisible


class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(
        self,
        InChannels: int,
        HidChannels: int=None,
        SqueezeFactor: int=4,
        Act: Callable[..., nn.Module]=nn.ReLU,
        ScaleAct: Callable[..., nn.Module]=nn.Sigmoid,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)
        
        self.SELayer = nn.Sequential()
        
        Avgpool = AdaptiveAvgPool2d(1)
        Fc1 = BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act)
        Fc2 = BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct)
        
        self.SELayer.add_module(name="global_pool", module=Avgpool)
        self.SELayer.add_module(name="fc1", module=Fc1)
        self.SELayer.add_module(name="fc2", module=Fc2)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.SELayer(x) # attention mechanisms, use global info