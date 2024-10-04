from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .weightInit import initWeight


class MyLayernorm(nn.LayerNorm):
    # 3.1 Layernorm(LN) is applied before every block
    def __init__(
        self, 
        normalized_shape, 
        eps: float = 0.00001, 
        elementwise_affine: bool = True, 
        device=None, dtype=None
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
    
    def profileModule(self, Input: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0


class DualLayernorm(nn.Module):
    # 3.1 Layernorm(LN) is applied before every block
    def __init__(self, DimEmb, Fn=None):
        super(DualLayernorm, self).__init__()
        self.LayerNorm = nn.LayerNorm(DimEmb)
        self.Fn = Fn
        
        self.apply(initWeight)
        
    def forward(self, x, Feature, **kwargs):
        x = self.LayerNorm(x)
        if self.Fn:
            x = self.Fn(x, **kwargs)
            
        Feature = self.LayerNorm(Feature)
        if self.Fn:
            Feature = self.Fn(Feature, **kwargs)
        return x, Feature
    
    def profileModule(self, Input: Tensor, Feature: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0
    

class NextLayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
    def profileModule(self, Input: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies a `Syncronized Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over the input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`*` is the remaining input dimensions
        - Output: same shape as the input

    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profileModule(self, Input: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0


class SyncBatchNormFP32(SyncBatchNorm):
    """
    Synchronized BN in FP32
    Sync-BN with 0 batch size does not work well with AMP. To avoid that,
    we replace all sync_bn in mask rcnn head with FP32 ones.
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        in_dtype = x.dtype
        return super().forward(x.to(dtype=torch.float)).to(dtype=in_dtype)

    def profileModule(self, Input: Tensor):
        # Since normalization layers can be fused, we do not count their operations
        Params = sum([p.numel() for p in self.parameters()])
        return Input, Params, 0.0
    