import math
import numpy as np
from typing import Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


def reduceTensor(InpTensor: torch.Tensor) -> torch.Tensor:
    Size = dist.get_world_size() if dist.is_initialized() else 1
    InpTensorClone = InpTensor.detach().clone()
    # dist_barrier()
    dist.all_reduce(InpTensorClone, op=dist.ReduceOp.SUM)
    InpTensorClone /= Size
    return InpTensorClone


def tensor2PythonFloat(
    InpTensor: Union[int, float, torch.Tensor], IsDistributed: bool
) -> Union[int, float, np.ndarray]:
    if IsDistributed and isinstance(InpTensor, torch.Tensor):
        InpTensor = reduceTensor(InpTensor=InpTensor)

    if isinstance(InpTensor, torch.Tensor) and InpTensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return InpTensor.cpu().numpy()
    elif hasattr(InpTensor, "item"):
        return InpTensor.item()
    elif isinstance(InpTensor, (int, float)):
        return InpTensor * 1.0
    else:
        raise NotImplementedError(
            "The data type is not supported yet in tensor_to_python_float function"
        )
    

def normaliseTensor(x: Tensor, SumOne=False) -> Tensor:
    Out = x - x.min(1, keepdim=True)[0]
    Out /= Out.max(1, keepdim=True)[0]
    
    if SumOne:
        RowVals = torch.sum(Out, 1)
        Out = torch.div(Out.T, RowVals).T
    
    return Out


def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    # shuffle multiple tensors with the same indexs
    # all tensors must have the same shape
    if isinstance(Feature, Tensor):
        Feature = [Feature]

    Indexs = None
    Output = []
    for f in Feature:
        # not in-place operation, should update output
        B, C, H, W = f.shape
        if Mode == 1:
            # fully shuffle
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            # shuflle along y and then x axis
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device), 
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output


def tensorRepeatLike(Feature: Tensor, RefTensor: Tensor, Res: bool=False, Reduce: bool=False) -> Tensor:
        _, _, H, W = RefTensor.shape
        # if Reduce:
        #     Feature = Feature.flatten(2).unsqueeze(3)
        #     _, _, Region, _ = Feature.shape
        #     Step = math.ceil(H / Region)
        #     Feature = Feature.repeat(1, 1, Step, 1)
        #     Feature = Feature[:, :, 0:H, :]
        #     Output = Feature
        # else:
        _, _, HRegion, WRegion = Feature.shape
        HStep = math.ceil(H / HRegion)
        WStep = math.ceil(W / WRegion)
        Feature = Feature.repeat(1, 1, HStep, WStep)
        Feature = Feature[:, :, 0:H, 0:W]
        Output = RefTensor + Feature if Res else Feature
        return Output


def tensorUpRepeat(x: Tensor, Weight: Tensor, Bias=None, Stride=2, Padding=0, OutputPadding=0, Groups=1, Dilation=1) -> Tensor:
    # # transpose conv
    # TransConv = TransposeConvLayer2d(OutChannels, OutChannels, 2, 2, output_padding=0, bias=False, ActLayer=nn.Sigmoid)
    # ## weight will be initialised automatically
    # TransConv.Conv.weight = nn.Parameter(torch.ones((OutChannels, OutChannels, 2, 2)), requires_grad=False)
    # self.TransConv(AttnMap)
    '''
    Weight = torch.ones((OutChannels, OutChannels, 2, 2)).to(opt.device)
    '''
    
    return F.conv_transpose2d(x, Weight, Bias, Stride, Padding, OutputPadding, Groups, Dilation)
