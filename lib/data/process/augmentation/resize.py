import random
from typing import Dict

from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

from .utils import resizeFn, resizedCropFn


class RandomShortSizeResize(object):
    """
    This class implements random resizing such that shortest side is 
    between specified minimum and maximum values.
    It will downsize the image, but keep the smallest short size.
    """
    def __init__(
        self, 
        MinShortSide=256, 
        MaxShortSide=768, 
        DimMaxImg=1024, 
        Interpolation=F.InterpolationMode.BILINEAR
        ) -> None:
        super().__init__()
        self.MinShortSide = MinShortSide
        self.MaxShortSide = MaxShortSide
        self.DimMaxImg = DimMaxImg
        self.Interpolation = Interpolation

    def __call__(self, Data: Dict) -> Dict:
        ShortSide = random.randint(self.MinShortSide, self.MaxShortSide)
        W, H = Data["image"].size
        Scale = min(
            ShortSide / min(H, W), self.DimMaxImg / max(H, W)
        )
        W = int(W * Scale)
        H = int(H * Scale)
        Data = resizeFn(Data, Size=(H, W), Interpolation=self.Interpolation)
        return Data


class Resize(T.Resize):
    def __init__(self, size, Interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, Interpolation, max_size, antialias)
        self.Interpolation = Interpolation
    def forward(self, Data: Dict) -> Dict:
        return resizeFn(Data, self.size, self.Interpolation)


class RandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), Interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, Interpolation)
        self.Interpolation = Interpolation
    
    def forward(self, Data: Dict) -> Dict:
        i, j, h, w = self.get_params(Data["image"], self.scale, self.ratio)
        return resizedCropFn(Data, i, j, h, w, self.size, self.Interpolation)

