from typing import Dict

import torch
from torch import Tensor
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    def forward(self, Data: Dict) -> Dict:
        if torch.rand(1) < self.p:
            Img = Data.pop("image")
            W, _ = F.get_image_size(Img)
            Data["image"] = F.hflip(Img)
            
            if "mask" in Data:
                if Data["mask"] is not None:
                    Mask = Data.pop("mask")
                    Data["mask"] = F.hflip(Mask)
            
            if "box_coordinates" in Data:
                Boxes = Data.pop("box_coordinates")
                Boxes[..., 0::2] = W - Boxes[..., 2::-2]
                Data["box_coordinates"] = Boxes

            if "instance_mask" in Data:
                assert "instance_coords" in Data

                InsCoords = Data.pop("instance_coords")
                InsCoords[..., 0::2] = W - InsCoords[..., 2::-2]
                Data["instance_coords"] = InsCoords

                InsMasks = Data.pop("instance_mask")
                Data["instance_mask"] = F.hflip(InsMasks)
        return Data


class RandomRotation(T.RandomRotation):
    # waite for box rotation
    def __init__(self, degrees, Interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=0):
        super().__init__(degrees, Interpolation, expand, center, fill)

    def rotateProcess(self, Img, Angle, Fill, Interpolation):
        # NumChannels, _, _ = F.get_dimensions(Img)
        NumChannels = len(Img.mode)
        if isinstance(Img, Tensor):
            if isinstance(Fill, (int, float)):
                Fill = [float(Fill)] * NumChannels
            else:
                Fill = [float(f) for f in Fill]
        return F.rotate(Img, Angle, Interpolation, self.expand, self.center, Fill)

    def forward(self, Data: Dict) -> Dict:
        Angle = self.get_params(self.degrees)
        
        Img = Data.pop("image")
        Data['image'] = self.rotateProcess(Img, Angle, 0, self.interpolation)
        
        if 'mask' in Data:
            if Data["mask"] is not None:
                Mask = Data['mask']
                Data['mask'] = self.rotateProcess(Mask, Angle, self.fill, T.InterpolationMode.NEAREST)
        return Data