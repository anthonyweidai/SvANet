import re
from typing import Any

import torch
from torch import nn, Tensor

from ...layers import BaseConv2d


class BaseViTEncoder(nn.Module):
    def __init__(self, opt, **kwargs: Any):
        super(BaseViTEncoder, self).__init__()
        self.opt = opt
        self.NumClasses = opt.cls_num_classes
        self.PatchSize = int(re.findall(r'patch(\d+)', opt.model_name)[0])
        self.Classifier = None

        # vit has only one layer stage
        self.ModelConfigDict = dict()
        self.ModelConfigDict['layer1'] = {'in': 3, 'out': 3, 'stage': 1}
        
    def checkModel(self):
        assert self.Classifier is not None, 'Please implement self.Classifier'
        
    def decoderInit(self):
        self.ModelConfigDict['layer1'] = {'in': 3, 'out': 16, 'stage': 1}
        self.ModelConfigDict['layer2'] = {'in': 16, 'out': 16, 'stage': 2}
        self.ModelConfigDict['layer3'] = {'in': 16, 'out': 16, 'stage': 3}
        self.ModelConfigDict['layer4'] = {'in': 16, 'out': 16, 'stage': 4}
        self.ModelConfigDict['layer4'] = {'in': 16, 'out': 16, 'stage': 5}
        return nn.Sequential(
            BaseConv2d(3, 16, 3, 2, BNorm=True, ActLayer=nn.ReLU),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def patchify(self, Imgs: Tensor) -> Tensor:
        """
        Imgs: (B, 3, H, W)
        x: (B, L, patch_size ** 2 * 3)
        """
        p = self.PatchSize
        assert Imgs.shape[2] == Imgs.shape[3] and Imgs.shape[2] % p == 0
        
        h = w = Imgs.shape[2] // p
        x = Imgs.reshape(shape=(Imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)
        return x.reshape(shape=(Imgs.shape[0], h * w, p**2 * 3))

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        x: (B, L, patch_size ** 2 * 3)
        Imgs: (B, 3, H, W)
        """
        p = self.PatchSize
        h = w = int(x.shape[1] ** .5)
        if h * w < x.shape[1]:
            # with token
            x = x[:, 1:, :]
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('bhwpqc->bchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, h * p))

    def d2Patchify(self, x: Tensor) -> Tensor:
        """
        x: (B, L, patch_size ** 2 * 3)
        Output: (B, patch_size ** 2 * 3, H, W)
        """
        p = self.PatchSize
        h = w = int(x.shape[1] ** .5)
        if h * w < x.shape[1]:
            # with token
            x = x[:, 1:, :]
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p ** 2 * 3))
        return torch.einsum('bhwe->behw', x)

    def forwardFeatures(self, x: Tensor) -> Tensor:
        # vit does not change the feature size in this process
        # support deeplabv3, pspnet in segmentaiton
        pass

    def forwardTuple(self, x: Tensor) -> Tensor:
        return [self.forwardFeatures(x)]

    def forwardHead(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.forwardFeatures(x)
        return self.forwardHead(x)