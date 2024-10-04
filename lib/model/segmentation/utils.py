import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..layers import initWeight
from ...utils import pair


class ContextSeg(nn.Module):
    def __init__(self, opt, InChannels, KernelSize=1):
        super().__init__()
        self.opt = opt
        self.OutChannels = opt.seg_num_classes
        self.AttnDepth = opt.cattn_depth - 1
        
        self.PatchSize = opt.cattn_patch
        self.ConvX = nn.Conv2d(InChannels, self.OutChannels, KernelSize, padding=KernelSize // 2)

        AttnDepth = self.AttnDepth + 1
        self.Weight = torch.ones((self.OutChannels, 1, 2 ** AttnDepth, 2 ** AttnDepth)).to(opt.device)
        
        self.Unfold = nn.Unfold(kernel_size=self.PatchSize, stride=self.PatchSize)

        self.Activation = nn.LogSoftmax(dim=1)
        
        if opt.init_weight:
            self.apply(initWeight)

    def forward(self, x: Tensor, Attentions: Tensor):
        x = self.ConvX(x)
        x = F.interpolate(x, size=pair(self.opt.resize_shape), mode="bilinear", align_corners=True)

        XArgmax = torch.argmax(x, dim=1)
        Temp = torch.zeros(x.shape).to(x.device)
        Src = torch.ones(x.shape).to(x.device)
        XSoftmax = Temp.scatter(dim=1, index=XArgmax.unsqueeze(1), src=Src)

        AttnDepth = self.AttnDepth + 1
        ArgxFeamap = F.conv2d(
            XSoftmax, self.Weight, None, pair(2 ** AttnDepth), groups=self.OutChannels
        ) / (2 ** (AttnDepth * 2))

        Correction = []
        for i in range(x.size()[1]):
            NonZeros = torch.unsqueeze(torch.count_nonzero(Attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)

            Attention = torch.matmul(
                Attentions[:, i:i + 1, :, :] / NonZeros, 
                torch.unsqueeze(self.Unfold(ArgxFeamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2)
            )

            Attention = torch.squeeze(Attention, dim=1)
            
            Attention = F.fold(Attention.transpose(-1, -2), x.shape[-2:], self.PatchSize, stride=self.PatchSize)

            Correction.append(Attention)

        Correction = torch.cat(Correction, dim=1)

        return self.Activation((Correction + 1) * x), Attentions