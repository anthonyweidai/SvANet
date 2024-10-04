from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ....layers import BaseConv2d, AdaptiveAvgPool2d, SeparableConv, Dropout2d


class ASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers, `here <https://arxiv.org/abs/1606.00915>`_ 
    and `here <https://arxiv.org/abs/1706.05587>`_

    Args:
        opt: command-line arguments
        InChannels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        OutChannels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        AtrousRates (Tuple[int]): atrous rates for different branches.
        IsSepConv (Optional[bool]): Use separable convolution instead of standaard conv. Default: False
        dropout (Optional[float]): Apply dropout. Default is 0.0

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self,
        InChannels: int,
        OutChannels: int,
        AtrousRates: Tuple[int],
        IsSepConv: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        **kwargs
    ) -> None:
        InProj = BaseConv2d(
            InChannels,
            OutChannels,
            kernel_size=1,
            stride=1,
            BNorm=True,
            ActLayer=nn.ReLU,
        )
        OutProj = BaseConv2d(
            5 * OutChannels,
            OutChannels,
            kernel_size=1,
            stride=1,
            BNorm=True,
            ActLayer=nn.ReLU,
        )
        ASPPLayer = ASPPSeparableConv if IsSepConv else ASPPConv

        assert len(AtrousRates) == 3

        Modules = [InProj]
        Modules.extend(
            [
                ASPPLayer(
                    InChannels=InChannels,
                    OutChannels=OutChannels,
                    dilation=rate,
                )
                for rate in AtrousRates
            ]
        )
        Modules.append(ASPPPooling(InChannels, OutChannels))

        super().__init__()
        self.Convs = nn.ModuleList(Modules)
        self.Project = OutProj

        self.InChannels = InChannels
        self.OutChannels = OutChannels
        self.AtrousRates = AtrousRates
        self.IsSepConvLayer = IsSepConv
        self.n_atrous_branches = len(AtrousRates)
        self.DropoutLayer = Dropout2d(p=dropout)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        out = []
        for conv in self.Convs:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        out = self.Project(out)
        out = self.DropoutLayer(out)
        return out
    

class ASPPConv(BaseConv2d):
    """
    Convolution with a dilation for the ASPP module
    Args:
        opt: command-line arguments
        InChannels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        OutChannels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        dilation (int): Dilation rate

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """
    def __init__(self, InChannels: int, OutChannels: int, dilation: int, **kwargs) -> None:
        super().__init__(
            InChannels, 
            OutChannels, 
            kernel_size=3, 
            stride=1,
            BNorm=True,
            ActLayer=nn.ReLU,
            dilation=dilation,
            )

    def adjustAtrousRate(self, rate: int) -> None:
        """This function allows to adjust the dilation rate"""
        self.Conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.Conv.padding = rate


class ASPPSeparableConv(SeparableConv):
    """
    Separable Convolution with a dilation for the ASPP module
    Args:
        opt: command-line arguments
        InChannels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        OutChannels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        dilation (int): Dilation rate

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self, InChannels: int, OutChannels: int, dilation: int, **kwargs
    ) -> None:
        super().__init__(
            InChannels=InChannels,
            OutChannels=OutChannels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            use_norm=True,
            use_act=True,
        )

    def adjustAtrousRate(self, rate: int) -> None:
        """This function allows to adjust the dilation rate"""
        self.DwConv.Conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.DwConv.Conv.padding = rate


class ASPPPooling(nn.Module):
    """
    ASPP pooling layer, spatial pyramid pooling
    Args:
        opt: command-line arguments
        InChannels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        OutChannels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self, InChannels: int, OutChannels: int, **kwargs
    ) -> None:

        super().__init__()
        self.AsppPool = nn.Sequential()
        self.AsppPool.add_module(
            name="global_pool", module=AdaptiveAvgPool2d(output_size=1)
        )
        self.AsppPool.add_module(
            name="conv_1x1",
            module=BaseConv2d(InChannels, OutChannels, 1, 1, BNorm=True, ActLayer=nn.ReLU),
        )

        self.InChannels = InChannels
        self.OutChannels = OutChannels

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.shape[-2:]
        x = self.AsppPool(x)
        x = F.interpolate(x, size=x_size, mode="bilinear", align_corners=False)
        return x