from typing import Callable, Optional
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn, Tensor

from timm.models.vision_transformer import PatchEmbed

from ..layers import Linearlayer, MyLayernorm, Dropout
from ..misc import moduleProfile
from ...utils import pair


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


class PatchEmbeddingv1(nn.Module):
    def __init__(self, ResizeRes, PatchRes=16, InChannel=3, DimEmb=768, NormLayer=None):
        super(PatchEmbeddingv1, self).__init__()
        
        self.ImgHeight, self.ImgWidth = pair(ResizeRes)
        PatchHeight, PatchWidth = pair(PatchRes)
        assert self.ImgHeight % PatchHeight == 0 and self.ImgWidth % PatchWidth == 0, \
            'Image dimensions must be divisible by the patch size.'
            
        PatchesRes = [self.ImgHeight // PatchHeight, self.ImgWidth // PatchWidth]
        self.PatchesRes = PatchesRes
        self.NumPatches = PatchesRes[0] * PatchesRes[1]
        
        self.Projection = nn.Conv2d(InChannel, DimEmb, kernel_size=PatchRes, stride=PatchRes)
        self.Norm = NormLayer(DimEmb) if NormLayer else nn.Identity()
        
        
    def forward(self, x):
        _, _, H, W = x.shape
        assert self.ImgHeight == H and self.ImgWidth == W, \
            f"Input image size ({H}*{W}) doesn't match model ({self.ImgHeight}*{self.ImgWidth})."
        
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.Projection(x).flatten(2).transpose(1, 2)
        return self.Norm(x)
    

class PatchEmbeddingv2(nn.Module):
    def __init__(self, ResizeRes, PatchRes=16, InChannel=3, DimEmb=768, NormLayer=None):
        super(PatchEmbeddingv2, self).__init__()
        
        ImgHeight, ImgWidth = pair(ResizeRes)
        PatchHeight, PatchWidth = pair(PatchRes)
        assert ImgHeight % PatchHeight == 0 and ImgWidth % PatchWidth == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        PatchesRes = [ImgHeight // PatchHeight, ImgWidth // PatchWidth]
        self.PatchesRes = PatchesRes
        self.NumPatches = PatchesRes[0] * PatchesRes[1]
        
        DimPatch = InChannel * PatchHeight * PatchWidth
        
        self.Embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PatchHeight, p2=PatchWidth),
            nn.Linear(DimPatch, DimEmb)
        )
        self.Norm = NormLayer(DimEmb) if NormLayer else nn.Identity()
    
    
    def forward(self, x):
        x = self.Embedding(x)
        return self.Norm(x)


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
    
    def profileModule(self, Input):
        BatchSize, SeqLen, InChannels = Input.shape
        Params = MACs = 0.0

        _, p, m = moduleProfile(module=self.toQKV, x=Input)
        Params += p
        MACs += (m * SeqLen * BatchSize)

        # number of operations in QK^T
        m_qk = (SeqLen * InChannels * InChannels) * BatchSize
        MACs += m_qk

        # number of operations in computing weighted sum
        m_wt = (SeqLen * InChannels * InChannels) * BatchSize
        MACs += m_wt

        _, p, m = moduleProfile(module=self.toOut, x=Input)
        Params += p
        MACs += (m * SeqLen * BatchSize)

        return Input, Params, MACs
    

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
    
    def profileModule(self, Input: Tensor):
        return moduleProfile(module=self.MLP, x=Input)
    
    
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
        # 3.1 Residual connections after every block
        for AttBlock, MLPBlock in self.Layers:
            x = AttBlock(x) + x # Eq(2)
            x = MLPBlock(x) + x # Eq(3)
        return x
    
    def profileModule(self, Input: Tensor):
        MACs, Params = 0, 0
        BatchSize, SeqLen = Input.shape[:2]

        for AttBlock, MLPBlock in self.Layers:
            ## The input shape doesn't change from each block
            _, p_mha, m_mha = moduleProfile(module=AttBlock, x=Input)

            _, p_ffn, m_ffn = moduleProfile(module=MLPBlock, x=Input)
            
            m_ffn = (m_ffn * BatchSize * SeqLen)
            
            MACs += m_mha + m_ffn
            Params += p_mha + p_ffn

        return Input, Params, MACs
    