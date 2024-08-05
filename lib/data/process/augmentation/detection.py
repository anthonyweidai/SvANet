import numpy as np
from typing import Dict

import torchvision.transforms.functional as F


class BoxPercentCoords(object):
    """
    This class converts the box coordinates to percent
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __call__(self, Data: Dict) -> Dict:
        if "box_coordinates" in Data:
            Boxes = Data.pop("box_coordinates")
            W, H = F.get_image_size(Data["image"])

            Boxes = Boxes.astype(np.float32)

            Boxes[..., 0::2] /= W
            Boxes[..., 1::2] /= H
            Data["box_coordinates"] = Boxes

        return Data
