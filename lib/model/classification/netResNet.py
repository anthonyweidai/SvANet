from typing import Type, Any, Union, List, Optional

import torch.nn as nn
from torch import Tensor

from . import registerClsModels
from ..modules import BaseEncoder
from ..layers import BaseConv2d, Linearlayer, initWeight
from ...utils import setMethod


class BasicBlock(nn.Module):
    Expansion: int = 1

    def __init__(
        self,
        InPlanes: int,
        Planes: int,
        Stride: int = 1,
        Downsample: Optional[nn.Module] = None,
        Groups: int = 1,
        BaseWidth: int = 64,
        Dilation: int = 1,
    ) -> None:
        super().__init__()
        if Groups != 1 or BaseWidth != 64:
            raise ValueError('BasicBlock only supports groups=1 and BaseWidth=64')
        if Dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.Downsample layers Downsample the input when stride != 1
        self.Conv1 = BaseConv2d(InPlanes, Planes, 3, Stride, BNorm=True, ActLayer=nn.ReLU)
        self.Conv2 = BaseConv2d(Planes, Planes, 3, 1, BNorm=True)
        
        self.Relu = nn.ReLU(inplace=True)
        self.Downsample = Downsample
        self.Stride = Stride

    def forward(self, x: Tensor) -> Tensor:
        if self.Downsample is not None:
            Identity = self.Downsample(x)
        else:
            Identity = x

        Out = self.Conv1(x)

        Out = self.Conv2(Out)

        

        Out += Identity # Skip connection in each Block
        Out = self.Relu(Out)

        return Out


class Bottleneck(nn.Module):
    Expansion: int = 4

    def __init__(
        self,
        InPlanes: int,
        Planes: int,
        Stride: int = 1,
        Downsample: Optional[nn.Module] = None,
        Groups: int = 1,
        BaseWidth: int = 64,
        Dilation: int = 1,
    ) -> None:
        super().__init__()
        Width = int(Planes * (BaseWidth / 64.)) * Groups
        # Both self.conv2 and self.Downsample layers Downsample the input when stride != 1
        self.Conv1 = BaseConv2d(InPlanes, Width, 1, BNorm=True, ActLayer=nn.ReLU)
        self.Conv2 = BaseConv2d(Width, Width, 3, Stride, groups=Groups, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU)
        self.Conv3 = BaseConv2d(Width, Planes * self.Expansion, 1, BNorm=True)
        self.Relu = nn.ReLU(inplace=True)
        self.Downsample = Downsample # projection, not identity mapping
        self.Stride = Stride

    def forward(self, x: Tensor) -> Tensor:
        Identity = x

        Out = self.Conv1(x)
        Out = self.Conv2(Out)
        Out = self.Conv3(Out)

        if self.Downsample is not None:
            Identity = self.Downsample(x)

        Out += Identity # Skip connection in each Block
        Out = self.Relu(Out)

        return Out


class ResNet(BaseEncoder):
    def __init__(
        self,
        opt,
        Block: Type[Union[BasicBlock, Bottleneck]],
        Layers: List[int],
        Groups: int = 1,
        WidthPerGroup: int = 64,
        ReplaceStrideWithDilation: Optional[List[bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(opt, **kwargs)
        
        self.InPlanes = 64
        self.Dilation = 1
        if ReplaceStrideWithDilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            ReplaceStrideWithDilation = [False, False, False]
        if len(ReplaceStrideWithDilation) != 3:
            raise ValueError(
                "ReplaceStrideWithDilation should be None "
                f"or a 3-element tuple, got {ReplaceStrideWithDilation}"
            )        
        self.Groups = Groups
        self.BaseWidth = WidthPerGroup
        
        InChannels = 3
        OutChannels = self.InPlanes
        self.Conv1 = BaseConv2d(InChannels, OutChannels, kernel_size=7, stride=2, padding=3, BNorm=True, ActLayer=nn.ReLU)
        self.ModelConfigDict['conv1'] = {'in': InChannels, 'out': OutChannels, 'stage': 1}

        ## Modified:
        # self.Conv1 = nn.Sequential(
        #     BaseConv2d(InChannels, OutChannels, 3, 2, BNorm=True, ActLayer=nn.ReLU),
        #     BaseConv2d(OutChannels, OutChannels, 3, 2, groups=OutChannels, BNorm=True, ActLayer=nn.ReLU))
        # self.ModelConfigDict['conv1'] = {'in': InChannels, 'out': OutChannels, 'stage': 1} # with stage 1 and 2

        # The layerX has many layers with similar identification
        StageCount = 2
        StrideList = [1, 2, 2, 2]
        OutChannelsList = [64, 128, 256, 512]
        for i in range(4):
            if StrideList[i] >= 2:
                StageCount += 1
            
            InChannels = OutChannels
            OutChannels = OutChannelsList[i]
            Layer, OutChannels = self.makeLayer(Block, OutChannels, Layers[i], StrideList[i], 
                                   ReplaceStrideWithDilation[i - 1] if i > 0 else False)
            
            setMethod(self, 'Layer%d' % (i + 1), 
                      Layer if i > 0 else nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), Layer))
            
            self.ModelConfigDict['layer%d' % (i + 1)] = {'in': InChannels, 'out': OutChannels, 'stage': StageCount}
        
        self.Classifier = Linearlayer(OutChannels, self.NumClasses)

        if opt.init_weight:
            self.apply(initWeight)
            # ZeroInitRes: # no improvement in my case 18/06/2022

    def makeLayer(
        self, 
        Block: Type[Union[BasicBlock, Bottleneck]], 
        Planes: int, 
        Blocks: int,
        Stride: int = 1, 
        dilate: bool = False,
        ) -> nn.Sequential:
        Downsample = None
        PreviousDilation = self.Dilation
        if dilate:
            self.Dilation *= Stride
            Stride = 1
        if Stride != 1 or self.InPlanes != Planes * Block.Expansion:
            Downsample = BaseConv2d(self.InPlanes, Planes * Block.Expansion, 1, Stride, BNorm=True)

        Layers = [] # init layers
        Layers.append(Block(self.InPlanes, Planes, Stride, Downsample, self.Groups,
                            self.BaseWidth, PreviousDilation))
        self.InPlanes = Planes * Block.Expansion
        for _ in range(1, Blocks):
            Layers.append(Block(self.InPlanes, Planes, Groups=self.Groups,
                                BaseWidth=self.BaseWidth, Dilation=self.Dilation
                                ))

        return nn.Sequential(*Layers), self.InPlanes
    

def resnet(
    Block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    opt,
    **kwargs: Any
) -> ResNet:
    Model = ResNet(opt, Block, layers, **kwargs)
    return Model


@registerClsModels("resnet18")
def resNet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


@registerClsModels("resnet34")
def resNet34(**kwargs: Any) -> ResNet:
    return resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


@registerClsModels("resnet50")
def resNet50(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


@registerClsModels("resnet101")
def resNet101(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


@registerClsModels("resnet152")
def resNet152(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


@registerClsModels("resnext50_32x4d")
def resnext50_32x4d(**kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return resnet(Bottleneck, [3, 4, 6, 3], Groups=32, WidthPerGroup=4, **kwargs)


@registerClsModels("resnext101_32x8d")
def resnext101_32x8d(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], Groups=32, WidthPerGroup=8, **kwargs)


@registerClsModels("resnext101_64x4d")
def resnext101_64x4d(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], Groups=64, WidthPerGroup=4, **kwargs)