import random
import numpy as np
from typing import Dict
from PIL import Image, ImageFilter

import torch
from torch import Tensor
from torchvision import transforms as T
import torchvision.transforms.functional as F


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        Sigma = random.uniform(self.sigma[0], self.sigma[1])
        Data["image"] = Img.filter(ImageFilter.GaussianBlur(radius=Sigma))
        return Data


class GaussianBlurWithKernel(T.GaussianBlur):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__(kernel_size, sigma)

    def forward(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        Sigma = self.get_params(self.sigma[0], self.sigma[1])
        Data["image"] = F.gaussian_blur(Img, self.kernel_size, [Sigma, Sigma])
        return Data
    

class RandomGaussianBlur(object):
    def __init__(self, p=0.5, **kwargs):
        super().__init__()
        self.p = p

    def __call__(self, Data: Dict) -> Dict:
        if random.random() < self.p:
            Img = Data.pop("image")
            # radius is the standard devaition of the gaussian kernel
            Data["image"] = Img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return Data


def randomMaskingPytorch(x: Tensor, MaskRatio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [B, C, H, W], sequence
    """
    C, H, W = x.shape

    XMaksed = x.reshape(C, H * W)
    C, L = XMaksed.shape  # batch, length, dim
    LenKeep = int(L * (1 - MaskRatio))

    # sort noise for each sample, 
    # x is not shuflle, just get random index
    # ascend: small is keep, large is remove
    Noise = torch.rand(L, device=x.device)  # noise in [0, 1]
    IdxShuffle = torch.argsort(Noise, dim=0)
    
    # generate the binary mask: 1 is keep, 0 is remove
    Mask = torch.zeros([L], device=x.device)
    Mask[:LenKeep] = 1
    Mask = torch.gather(Mask, dim=0, index=IdxShuffle)
    Mask = Mask.reshape(H, W).unsqueeze(0)

    return x.masked_fill(Mask==0, 0), Mask


def randomMaskingPil(x: Image.Image, MaskRatio):
    W, H = x.size

    L = H * W
    LenKeep = int(L * (1 - MaskRatio))

    # sort noise for each sample, 
    # x is not shuflle, just get random index
    # ascend: small is keep, large is remove
    Noise = np.random.rand(L)  # noise in [0, 1]
    IdxShuffle = np.argsort(Noise, axis=0)

    # generate the binary mask: 0 is keep, 1 is remove
    NpMask = np.ones([L])
    NpMask[:LenKeep] = 0
    NpMask = NpMask[IdxShuffle]
    NpMask = NpMask.reshape((H, W))
    
    Mask = Image.fromarray(NpMask == 1)
    BlackImg = Image.new(x.mode, (W, H))
    x.paste(BlackImg, (0, 0), Mask)

    return x, Mask


class RandomMasking(object):
    """ Apply random mask to the PIL image. 
    Effect like Guassian blur
    """
    def __init__(self, MaskRatio):
        self.MaskRatio = MaskRatio

    def __call__(self, Data):
        if self.MaskRatio is not None and 0 < self.MaskRatio < 1:
            if isinstance(Data, Dict):
                Img = Data.pop("image")
            else:
                Img = Data
                
            if isinstance(Img, Tensor):
                OutImg, _ = randomMaskingPytorch(Img, self.MaskRatio) # after to tensor
            else:
                OutImg, _ = randomMaskingPil(Img, self.MaskRatio) # before to tensor
              
            if isinstance(Data, Dict):
                Data["image"] = OutImg
            else:
                Data = OutImg
            
        return Data
