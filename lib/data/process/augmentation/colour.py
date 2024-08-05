import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Dict

import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        
    def forward(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                Img = F.adjust_brightness(Img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                Img = F.adjust_contrast(Img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                Img = F.adjust_saturation(Img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                Img = F.adjust_hue(Img, hue_factor)

        Data['image'] = Img
        return Data


class PhotometricDistort(object):
    """
    This class implements Photometeric distorion.

    .. note::
        Hyper-parameters of PhotoMetricDistort in PIL and OpenCV are different. Be careful
    """

    def __init__(self, p=0.5) -> None:
        # Contrast
        alpha_min = 0.5
        alpha_max = 1.5
        
        Contrast = T.ColorJitter(contrast=[alpha_min, alpha_max])

        # Brightness
        beta_min = 0.875
        beta_max = 1.125
        Brightness = T.ColorJitter(brightness=[beta_min, beta_max])

        # Saturation
        gamma_min = 0.5
        gamma_max = 1.5
        Saturation = T.ColorJitter(saturation=[gamma_min, gamma_max])

        # Hue
        delta_min = -0.05
        delta_max = 0.05
        Hue = T.ColorJitter(hue=[delta_min, delta_max])

        super().__init__()
        self.Brightness = Brightness
        self.Contrast = Contrast
        self.Hue = Hue
        self.Saturation = Saturation
        self.p = p
    
    def applyTransformations(self, Img):
        r = np.random.rand(7)

        if r[0] < self.p:
            Img = self.Brightness(Img)

        contrast_before = r[1] < self.p
        if contrast_before and r[2] < self.p:
            Img = self.Contrast(Img)

        if r[3] < self.p:
            Img = self.Saturation(Img)

        if r[4] < self.p:
            Img = self.Hue(Img)

        if not contrast_before and r[5] < self.p:
            Img = self.Contrast(Img)

        if r[6] < self.p and Img.mode != "L":
            # Only permute channels for RGB images
            # [H, W, C] format
            NpImg = np.asarray(Img)
            NumChannels = NpImg.shape[2]
            NpImg = NpImg[..., np.random.permutation(range(NumChannels))]
            Img = Image.fromarray(NpImg)
        return Img

    def __call__(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        Data["image"] = self.applyTransformations(Img)
        return Data


class Solarization(object):
    """ Apply Solarization to the PIL image. """
    def __init__(self, p):
        self.p = p

    def __call__(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        
        if random.random() < self.p:
            Img = ImageOps.solarize(Img)
            
        Data['image'] = Img
        return Data


class RandomGrayscale(T.RandomGrayscale):
    def __init__(self, p=0.1):
        super().__init__(p)
    
    def forward(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        
        # num_output_channels, _, _ = F.get_dimensions(Img)
        NumChannels = len(Img.mode)
        if torch.rand(1) < self.p:
            Img = F.rgb_to_grayscale(Img, num_output_channels=NumChannels)
        Data['image'] = Img
        return Data
