import math
import numpy as np
from typing import Union, Sequence, Tuple, Optional, Any, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .vit import Transformer
from .submodules import LinearAttnFFN
from ..layers import BaseConv2d, MyLayernorm, StochasticDepth
from ..misc import moduleProfile
from ...utils import pair, makeDivisible


class BaseFormer(nn.Module):
    def __init__(
        self,
        InChannels: int,
        FfnMultiplier: Optional[Union[Sequence[Union[int, float]], int, float]]=2.0,
        NumAttnBlocks: Optional[int]=2,
        AttnDropRate: Optional[float]=0.0,
        DropRate: Optional[float]=0.0,
        FfnDropRate: Optional[float]=0.0,
        PatchRes: Optional[int]=2,
        Dilation: Optional[int]=1,
        ViTSELayer: Optional[nn.Module]=None,
        **kwargs: Any,
    ) -> None:
        DimAttnUnit = InChannels // 2
        DimCNNOut = DimAttnUnit

        Conv3x3In = BaseConv2d(
            InChannels, InChannels, 3, 1, dilation=Dilation, 
            BNorm=True, ActLayer=nn.SiLU,
        ) # depth-wise separable convolution
        ViTSELayer = ViTSELayer(InChannels, **kwargs) if ViTSELayer is not None else nn.Identity()
        Conv1x1In = BaseConv2d(InChannels, DimCNNOut, 1, 1, bias=False)

        super(BaseFormer, self).__init__()
        self.LocalRep = nn.Sequential(Conv3x3In, ViTSELayer, Conv1x1In)

        self.GlobalRep, DimAttnUnit = self.buildAttnLayer(
            DimAttnUnit, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate, FfnDropRate,
        )
        self.ConvProj = BaseConv2d(DimCNNOut, InChannels, 1, 1, BNorm=True)

        self.DimCNNOut = DimCNNOut
        
        self.HPatch, self.WPatch = pair(PatchRes)
        self.PatchArea = self.WPatch * self.HPatch

    def buildAttnLayer(
        self,
        DimModel: int,
        FfnMult: Union[Sequence, int, float],
        NumAttnBlocks: int,
        AttnDropRate: float,
        DropRate: float,
        FfnDropRate: float,
    ) -> Tuple[nn.Module, int]:

        if isinstance(FfnMult, Sequence) and len(FfnMult) == 2:
            DimFfn = (
                np.linspace(FfnMult[0], FfnMult[1], NumAttnBlocks, dtype=float) * DimModel
            )
        elif isinstance(FfnMult, Sequence) and len(FfnMult) == 1:
            DimFfn = [FfnMult[0] * DimModel] * NumAttnBlocks
        elif isinstance(FfnMult, (int, float)):
            DimFfn = [FfnMult * DimModel] * NumAttnBlocks
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        DimFfn = [makeDivisible(d, 16) for d in DimFfn]

        GlobalRep = [
            LinearAttnFFN(DimModel, DimFfn[block_idx], AttnDropRate, DropRate, FfnDropRate)
            for block_idx in range(NumAttnBlocks)
        ]
        GlobalRep.append(nn.BatchNorm2d(DimModel))
        return nn.Sequential(*GlobalRep), DimModel

    def unfolding(self, FeatureMap: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        B, C, H, W = FeatureMap.shape

        # [B, C, H, W] --> [B, C, P, N]
        Patches = F.unfold(
            FeatureMap,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )
        Patches = Patches.reshape(
            B, C, self.HPatch * self.WPatch, -1
        )

        return Patches, (H, W)
    
    def folding(self, Patches: Tensor, OutputSize: Tuple[int, int]) -> Tensor:
        B, C, P, N = Patches.shape # BatchSize, DimIn, PatchSize, NumPatches

        # [B, C, P, N]
        Patches = Patches.reshape(B, C * P, N)

        FeatureMap = F.fold(
            Patches,
            output_size=OutputSize,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )

        return FeatureMap
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        Fm = self.LocalRep(x)

        # convert feature map to patches
        Patches, OutputSize = self.unfolding(Fm)

        # learn global representations on all patches
        Patches = self.GlobalRep(Patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        Fm = self.folding(Patches, OutputSize)
        Fm = self.ConvProj(Fm)

        return Fm


class AssembleFormer(BaseFormer):
    """
    Inspired by MobileViTv3.
    Adapted from https://github.com/micronDLA/MobileViTv3/blob/main/MobileViTv3-v2/cvnets/modules/mobilevit_block.py
    """
    def __init__(
        self,
        InChannels: int,
        FfnMultiplier: Optional[Union[Sequence[Union[int, float]], int, float]]=2.0,
        NumAttnBlocks: Optional[int]=2,
        AttnDropRate: Optional[float]=0.0,
        DropRate: Optional[float]=0.0,
        FfnDropRate: Optional[float]=0.0,
        PatchRes: Optional[int]=2,
        Dilation: Optional[int]=1,
        SDProb: Optional[float]=0.0,
        ViTSELayer: Optional[nn.Module]=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(InChannels, FfnMultiplier, NumAttnBlocks, AttnDropRate, 
                         DropRate, FfnDropRate, PatchRes, Dilation, ViTSELayer, **kwargs)
        # AssembleFormer: input changed from just global to local + global
        self.ConvProj = BaseConv2d(2 * self.DimCNNOut, InChannels, 1, 1, BNorm=True)
        
        self.Dropout = StochasticDepth(SDProb)
        
    def forward(self, x: Tensor) -> Tensor:
        FmConv = self.LocalRep(x)

        # convert feature map to patches
        Patches, OutputSize = self.unfolding(FmConv)

        # learn global representations on all patches
        Patches = self.GlobalRep(Patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        Fm = self.folding(Patches, OutputSize)

        # AssembleFormer: local + global instead of only global
        Fm = self.ConvProj(torch.cat((Fm, FmConv), dim=1))

        # AssembleFormer: skip connection
        return x + self.Dropout(Fm)