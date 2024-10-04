import random
import numpy as np
from typing import Dict

import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F


class RandomApply(T.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)
        
    def forward(self, Data: Dict) -> Dict:
        if self.p < torch.rand(1):
            return Data
        for t in self.transforms:
            Data = t(Data)
        return Data
    

class RandomOrder(T.RandomOrder):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, Data: Dict) -> Dict:
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            Data = self.transforms[i](Data)
        return Data


class Normalisation(T.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
    def forward(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        Data["image"] = F.normalize(Img, self.mean, self.std, self.inplace)
        return Data
    

class ToTensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, Data: Dict) -> Dict:
        # HWC --> CHW
        Img = Data.pop("image")
        Data["image"] = F.to_tensor(Img)

        if "mask" in Data:
            if Data["mask"] is not None:
                Mask = Data.pop("mask")
                Mask = np.array(Mask)
                Data["mask"] = torch.as_tensor(Mask, dtype=torch.long)
            
        if "box_coordinates" in Data:
            Boxes = Data.pop("box_coordinates")
            Data["box_coordinates"] = torch.as_tensor(Boxes, dtype=torch.float)

        if "box_labels" in Data:
            BoxesLabels = Data.pop("box_labels")
            Data["box_labels"] = torch.as_tensor(BoxesLabels)

        if "instance_mask" in Data:
            assert "instance_coords" in Data
            InsMasks = Data.pop("instance_mask")
            Data["instance_mask"] = InsMasks.to(dtype=torch.long)

            InsCoords = Data.pop("instance_coords")
            Data["instance_coords"] = torch.as_tensor(InsCoords, dtype=torch.float)
        return Data