import math
from typing import Optional, Callable, Union, Tuple, Any

import torch
from torch import nn, Tensor

from .weightInit import initWeight
from ...utils import pair


class BaseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int]=1,
        padding: Optional[int]=None,
        groups: Optional[int]=1,
        bias: Optional[bool]=None,
        BNorm: bool=False,
        # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        dilation: int=1,
        Momentum: Optional[float]=0.1,
        **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
            
        if bias is None:
            bias = not BNorm
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        self.Conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)
        
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()
        
        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer
        
        self.apply(initWeight)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

    def profileModule(self, Input: Tensor):
        if Input.dim() != 4:
            print('Conv2d requires 4-dimensional Input (BxCxHxW). Provided Input has shape: {}'.format(Input.size()))

        BatchSize, in_channels, in_h, in_w = Input.size()
        assert in_channels == self.in_channels, '{}!={}'.format(in_channels, self.in_channels)

        k_h, k_w = pair(self.kernel_size)
        stride_h, stride_w = pair(self.stride)
        pad_h, pad_w = pair(self.padding)
        groups = self.groups

        out_h = (in_h - k_h + 2 * pad_h) // stride_h + 1
        out_w = (in_w - k_w + 2 * pad_w) // stride_w + 1

        # compute MACs
        MACs = (k_h * k_w) * (in_channels * self.out_channels) * (out_h * out_w) * 1.0
        MACs /= groups

        if self.bias:
            MACs += self.out_channels * out_h * out_w

        # compute parameters
        Params = sum([p.numel() for p in self.parameters()])

        Output = torch.zeros(size=(BatchSize, self.out_channels, out_h, out_w), dtype=Input.dtype, device=Input.device)
        # print(MACs)
        return Output, Params, MACs


class SeparableConv(nn.Module):
    """
    Applies a `2D depth-wise separable convolution 
    <https://arxiv.org/abs/1610.02357>`_ over a 4D input tensor
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Optional[Union[int, Tuple]]=1,
        dilation: Optional[Union[int, Tuple]]=1,
        BNorm: Optional[bool]=True,
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        bias: Optional[bool]=None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if bias is None:
            bias = not BNorm
        
        self.DwConv = BaseConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=in_channels,
            BNorm=True,
        )
        self.PwConv = BaseConv2d(
            in_channels,
            out_channels,
            1,
            1,
            dilation=1,
            groups=1,
            bias=bias,
            BNorm=BNorm,
            ActLayer=ActLayer,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        x = self.DwConv(x)
        x = self.PwConv(x)
        return x

    def profileModule(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params, macs = 0.0, 0.0
        input, p, m = self.DwConv.profileModule(input)
        params += p
        macs += m

        input, p, m = self.PwConv.profileModule(input)
        params += p
        macs += m
        return input, params, macs


class TransposeConvLayer2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Optional[Union[int, Tuple]] = 1,
        padding: Optional[int]=None,
        dilation: Optional[Union[int, Tuple]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool]=None,
        padding_mode: Optional[str] = "zeros",
        BNorm: bool=False,
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        output_padding: Optional[Union[int, Tuple]] = None,
        Momentum: Optional[float]=0.1,
        **kwargs
    ):
        super().__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
            
        if bias is None:
            bias = not BNorm
        
        if output_padding is None:
            output_padding = stride - 1
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
            
        self.Conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                        padding, output_padding, groups, bias, dilation, 
                                        padding_mode=padding_mode)
        
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()
        
        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer
            
        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

    def profileModule(self, input: Tensor) -> Tuple[Tensor, float, float]:
        assert input.dim() == 4, "Conv2d requires 4-dimensional input (BxCxHxW). \
            Provided input has shape: {}".format(input.size())

        b, in_c, in_h, in_w = input.size()
        assert in_c == self.in_channels, "{}!={}".format(in_c, self.in_channels)

        stride_h, stride_w = self.stride
        groups = self.groups

        out_h = in_h * stride_h
        out_w = in_w * stride_w

        k_h, k_w = self.kernel_size

        # compute MACS
        macs = (k_h * k_w) * (in_c * self.out_channels) * (out_h * out_w) * 1.0
        macs /= groups

        if self.bias:
            macs += self.out_channels * out_h * out_w

        # compute parameters
        params = sum([p.numel() for p in self.parameters()])

        output = torch.zeros(
            size=(b, self.out_channels, out_h, out_w),
            dtype=input.dtype,
            device=input.device,
        )
        # print(macs)
        return output, params, macs


def computeConvTensorHW(OriRes, KernelSize=3, Stride=2, Padding=None, Dilation=1, **kwargs: Any):
    OriH, OriW = pair(OriRes)
    
    if Padding is None:
        Padding = int((KernelSize - 1) // 2 * Dilation)
        
    OutH = math.floor((OriH + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1)
    OutW = math.floor((OriW + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1)
            
    return OutH, OutW


def getTensorHWbyStage(InputRes, Stage, **kwargs: Any):
    OutH, OutW = pair(InputRes)
    for _ in range(Stage):
        OutH, OutW = computeConvTensorHW((OutH, OutW))
    return OutH, OutW
