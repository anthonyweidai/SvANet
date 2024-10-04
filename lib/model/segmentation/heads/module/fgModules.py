import random
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from .resLinkModules import UpLink
from ....layers import (
    getChannelsbyStage, computeMaxStage,
    BaseConv2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, StochasticDepth,
)
from .....utils import makeDivisible, shuffleTensor, setMethod, callMethod


class FGBottleneck(nn.Module):
    def __init__(
        self,
        InChannels: int,
        HidChannels: int=None,
        Expansion: float=2.,
        Stride: int=1,
        Dilation: int=1,
        DropRate: float=0.0, # 0 would be better
        SELayer: nn.Module=None,
        ActLayer: nn.Module=None,
        ViTBlock: nn.Module=None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # Feature guide bottleneck
        if HidChannels is None:
            HidChannels = makeDivisible(InChannels * Expansion, 8)
        self.Bottleneck = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(HidChannels, HidChannels, 3, Stride, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU),
            SELayer(InChannels=HidChannels, **kwargs) if SELayer is not None else nn.Identity(),
            BaseConv2d(HidChannels, InChannels, 1, BNorm=True)
        )
        
        self.ActLayer = ActLayer(inplace=True) if ActLayer is not None else nn.Identity()
        self.Dropout = StochasticDepth(DropRate) if DropRate > 0 else nn.Identity()
        
        self.ViTLayer = ViTBlock(InChannels, **kwargs) if ViTBlock is not None else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        Out = self.Bottleneck(x)
        Out = self.ActLayer(x + self.Dropout(Out))
        return self.ViTLayer(Out)
    

class BasicBlock(nn.Module):
    def __init__(
        self,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # lower performance with vit
        self.Conv = nn.Sequential(
            BaseConv2d(InChannels, OutChannels, 3, Stride, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(OutChannels, OutChannels, 1, BNorm=True, ActLayer=nn.ReLU),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.Conv(x)


class CSLayer(nn.Module):
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        NumBlocks: int=1,
        Stage: int=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # lower performance with vit
        self.NumBlocks = NumBlocks
        self.SvAttn  = opt.fg_svattn

        Block = BasicBlock
        
        for i in range(NumBlocks + 1):
            if i < NumBlocks:
                OutChannelsC1 = InChannels if i < NumBlocks - 1 else OutChannels
                Layer = Block(InChannels, OutChannelsC1, Stride, Reparamed=opt.reparamed, **kwargs)
                Name = 'Conv%d' % (i + 1)
                setMethod(self, Name, Layer)
            
            if opt.fg_svattn:
                if opt.fg_svattn == -1 and i > 0:
                    # vanilla attention only has one squeeze module
                    continue
                OutChannelsC2 = InChannels if i < NumBlocks else OutChannels
                Layer = BaseConv2d(OutChannelsC2, OutChannels, 1)
                Name = 'Squeeze%d' % (i + 1)
                setMethod(self, Name, Layer)
            
        if opt.fg_svattn:
            # Pool tensor to be 1 / (opt.fg_svattn_divisor ** 2) image size for final stage attention
            if opt.fg_svattn > 0:
                DefaultRes = opt.resize_shape // 2 ** (Stage + NumBlocks)
                self.PoolSize = int(max(DefaultRes // opt.fg_svattn_divisor, 4))
                self.PoolLayer = AdaptiveAvgPool2d(self.PoolSize)
                
                ImgMask = torch.randint(1, NumBlocks + 2, (self.PoolSize, self.PoolSize))
                ImgMask = ImgMask[None, None]
                
                self.AttnMask = []
                for i in range(NumBlocks + 1):
                    Mask = deepcopy(ImgMask)
                    AttnMask = Mask.masked_fill(Mask != i + 1, float(0.0)).masked_fill(Mask == i + 1, float(1.0))
                    self.AttnMask.append(AttnMask.to(opt.device))
            
            self.PoolOne = AdaptiveAvgPool2d(1) if opt.fg_svattn >= -2 else AdaptiveMaxPool2d(1)
                
            self.ActLayer = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        SubFeatureTuple = [x]
        for i in range(self.NumBlocks):
            Name = 'Conv%d' % (i + 1)
            Output = callMethod(self, Name)(SubFeatureTuple[-1])
            SubFeatureTuple.append(Output)
        
        if self.SvAttn :
            if self.SvAttn < -1 or self.SvAttn > 0:
                # inconstant attention region, high loss fluctuation at first
                PoolFeatureTuple = []
                
                if self.training and self.SvAttn > 0:
                    # randomise affected regions of each mask
                    AttnMask = shuffleTensor(self.AttnMask)
                    # randomise mask order
                    random.shuffle(AttnMask)
                for i, f in enumerate(SubFeatureTuple):
                    Name = 'Squeeze%d' % (i + 1)
                    # pool first and then conv achieves better performance
                    if self.SvAttn > 0:
                        PoolFeature = callMethod(self, Name)(self.PoolLayer(f))
                        PoolFeatureTuple.append(PoolFeature * (AttnMask[i] if self.training else 1))
                    else:
                        PoolFeature = callMethod(self, Name)(self.PoolOne(f))
                        PoolFeatureTuple.append(PoolFeature)
                
                AttnMap = sum(PoolFeatureTuple)
                if not self.training:
                    # keep the same scale
                    AttnMap /= len(PoolFeatureTuple)
                AttnMap = self.PoolOne(AttnMap)
                
            else:
                Name = 'Squeeze1'
                AttnMap = callMethod(self, Name)(self.PoolOne(x))
                
            return SubFeatureTuple[-1] * self.ActLayer(AttnMap) # AttnMap
        else:
            return SubFeatureTuple[-1]    


class FGLink(nn.Module):
    '''
    similar to reslink, used in feature guide
    '''
    def __init__(self, opt, ModelConfigDict, ViTBlock: nn.Module=None, **kwargs: Any) -> None:
        super().__init__()
        self.opt = opt
        self.UpLink = []
        # self.StageList = [] # different from reslink, it use encoded feature tuple
        MaxStage = computeMaxStage(ModelConfigDict)
        
        InChannels = getChannelsbyStage(ModelConfigDict, MaxStage)
        for i, s in enumerate(reversed(range(opt.fg_start_stage, MaxStage))):
            OutChannels = getChannelsbyStage(ModelConfigDict, s)
 
            UpConv = UpLink(InChannels, OutChannels)
            setMethod(self, 'UpLink%d' % (i + 1), UpConv)
            
            if opt.fg_link == 2:
                CatLinkConv = BasicBlock(OutChannels * 2, OutChannels)
                setMethod(self, 'CatLinkConv%d' % (i + 1), CatLinkConv)
                
            ViTLayer = ViTBlock(OutChannels, **kwargs) if ViTBlock is not None else nn.Identity()
            setMethod(self, 'ViTLayer%d' % (i + 1), ViTLayer)
            
            InChannels = OutChannels
            # self.StageList.append(s) 

    def forwardFeature(self, FeaturesTuple: list) -> list:
        DecodeFeatureTuple = []
        DecodeFeature = FeaturesTuple[-1]
        for i, f in enumerate(reversed(FeaturesTuple[self.opt.fg_start_stage - 1:-1])):
            UpLink = callMethod(self, 'UpLink%d' % (i + 1))
            DecodeFeature = UpLink(DecodeFeature)
            if self.opt.fg_link == 1:
                DecodeFeature = DecodeFeature + f
            else:
                DecodeFeature = torch.cat([DecodeFeature, f], axis=1)
                
                CatLinkConv = callMethod(self, 'CatLinkConv%d' % (i + 1))
                DecodeFeature = CatLinkConv(DecodeFeature)
                
            ViTLayer = callMethod(self, 'ViTLayer%d' % (i + 1))
            DecodeFeature = ViTLayer(DecodeFeature)
            
            DecodeFeatureTuple.append(DecodeFeature)
        return DecodeFeatureTuple
        
    def forward(self, FeaturesTuple: list) -> Tensor:
        return self.forwardFeature(FeaturesTuple)[-1]
