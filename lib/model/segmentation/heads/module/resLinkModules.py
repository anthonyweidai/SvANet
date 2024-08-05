from torch import nn, Tensor
from ....layers import BaseConv2d, TransposeConvLayer2d


class UpLink(nn.Module):
    def __init__(self, InChannels, OutChannels):
        super(UpLink, self).__init__()
        self.TPConv = nn.Sequential(
            BaseConv2d(InChannels, InChannels // 4, 1, 1, 0, BNorm=True, ActLayer=nn.ReLU),
            TransposeConvLayer2d(InChannels // 4, InChannels // 4, 3, 2, 
                               padding=1, output_padding=1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(InChannels // 4, OutChannels, 1, 1, 0, BNorm=True, ActLayer=nn.ReLU)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.TPConv(x)
