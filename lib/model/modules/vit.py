from typing import Callable, Optional
from einops import rearrange

import torch
from torch import nn, Tensor

from timm.models.vision_transformer import PatchEmbed

from ..layers import Linearlayer, MyLayernorm, Dropout


class TimmPatchEmbed(PatchEmbed):
    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, 
                            flatten, output_fmt, bias, strict_img_size, dynamic_img_pad)
        self.NumPatches = self.num_patches
        self.PatchesRes = self.grid_size # image size // patch size


class MHSAttention(nn.Module):
    # Multi-head Self-attention mechanism
    def __init__(self, DimEmb, NumHead=8, DimHead=64, DropRate=0., bias=False):
        super(MHSAttention, self).__init__()
        DimModel = NumHead * DimHead # d_k = d_v = d_model/h = 64 (by default)
        
        self.NumHead = NumHead
        self.Scale = DimHead ** -0.5 # 1 / sqrt(d_k)
        
        self.Softmax = nn.Softmax(dim=-1)
        self.toQKV = Linearlayer(DimEmb, DimModel * 3, bias)
        
        self.toOut = nn.Sequential(
            Linearlayer(DimModel, DimEmb, bias),
            Dropout(DropRate)
        ) if not (NumHead == 1 and DimHead == DimEmb) else nn.Identity()
        
    def forward(self, x):
        qkv = self.toQKV(x).chunk(3, dim=-1)
        Query, Key, Value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.NumHead), qkv)
        
        x = torch.matmul(Query, Key.transpose(-1, -2)) * self.Scale # scaled dot product
        x = self.Softmax(x)
        
        x = torch.matmul(x, Value)
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.toOut(x)
    

class ViTMLP(nn.Module):
    # 3.1 The MLP contains two layers with a GELU non-linearity 
    def __init__(self, DimEmb, MLPSize, DropRate1=0., DropRate2=0., bias=False):
        super(ViTMLP, self).__init__()
        self.MLP = nn.Sequential(
            Linearlayer(DimEmb, MLPSize, bias),
            nn.GELU(),
            Dropout(DropRate1),
            Linearlayer(MLPSize, DimEmb, bias),
            Dropout(DropRate2),
        )
        
    def forward(self, x):
        return self.MLP(x)

    
class Transformer(nn.Module):
    def __init__(self, DimEmb, Depth, NumHead, DimHead, MLPSize, AttDropRate=0., MLPDropRate1=0., MLPDropRate2=0., bias=False):
        super(Transformer, self).__init__()
        self.Depth = Depth
        
        self.Layers = nn.ModuleList([])
        for _ in range(Depth):
            self.Layers.append(nn.ModuleList([
                nn.Sequential(MyLayernorm(DimEmb),
                              MHSAttention(DimEmb, NumHead, DimHead, AttDropRate, bias)),
                nn.Sequential(MyLayernorm(DimEmb),
                              ViTMLP(DimEmb, MLPSize, MLPDropRate1, MLPDropRate2, bias))
            ])) # Trainable
            
    def forward(self, x):
        # Residual connections after every block
        for AttBlock, MLPBlock in self.Layers:
            x = AttBlock(x) + x # Eq(2)
            x = MLPBlock(x) + x # Eq(3)
        return x
    