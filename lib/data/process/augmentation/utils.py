import numbers
import numpy as np
from typing import Sequence, Union, Dict, List

import torch
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision import transforms as T
import torchvision.transforms.functional as F

from ....utils import pair


def padFn(Data: Dict, padding: List[int], fill: int=0, padding_mode: str="constant") -> Tensor:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(padFn)
        
    Img = Data.pop("image")
    Data["image"] = F.pad(Img, padding=padding, fill=fill, padding_mode=padding_mode)

    if "mask" in Data:
        if Data["mask"] is not None:
            Mask = Data.pop("mask")
            Data["mask"] = F.pad(Mask, padding=padding, fill=fill, padding_mode=padding_mode)
        
    return Data


def cropFn(Data: Dict, top: int, left: int, height: int, width: int) -> Dict:
    """Helper function for cropping"""
    Img = Data.pop("image")
    Data["image"] = F.crop(Img, top=top, left=left, height=height, width=width)

    if "mask" in Data:
        if Data["mask"] is not None:
            Mask = Data.pop("mask")
            Data["mask"] = F.crop(Mask, top=top, left=left, height=height, width=width)
    
    if "box_coordinates" in Data:
        boxes = Data.pop("box_coordinates")

        area_before_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )

        boxes[..., 0::2] = np.clip(boxes[..., 0::2] - left, a_min=0, a_max=left + width)
        boxes[..., 1::2] = np.clip(boxes[..., 1::2] - top, a_min=0, a_max=top + height)

        area_after_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )
        area_ratio = area_after_cropping / (area_before_cropping + 1)

        # keep the boxes whose area is atleast 20% of the area before cropping
        keep = area_ratio >= 0.2

        box_labels = Data.pop("box_labels")

        Data["box_coordinates"] = boxes[keep]
        Data["box_labels"] = box_labels[keep]

    if "instance_mask" in Data:
        assert "instance_coords" in Data

        instance_masks = Data.pop("instance_mask")
        Data["instance_mask"] = F.crop(
            instance_masks, top=top, left=left, height=height, width=width
        )

        instance_coords = Data.pop("instance_coords")
        instance_coords[..., 0::2] = np.clip(
            instance_coords[..., 0::2] - left, a_min=0, a_max=left + width
        )
        instance_coords[..., 1::2] = np.clip(
            instance_coords[..., 1::2] - top, a_min=0, a_max=top + height
        )
        Data["instance_coords"] = instance_coords

    return Data


def resizeFn(
    Data: Dict,
    Size: Union[Sequence, int],
    Interpolation: Union[T.InterpolationMode, str] = T.InterpolationMode.BILINEAR,
) -> Dict:
    Img = Data.pop("image")
    
    W, H = F.get_image_size(Img) # W, H for Pillow image
    SizeH, SizeW = pair(Size) # B C H W for pytorch tensor
    
    # return if does not resize
    if (W, H) == (SizeW, SizeH):
        Data["image"] = Img
        return Data
    
    Data["image"] = F.resize(Img, pair(Size), Interpolation, antialias=True)
    
    if "mask" in Data:
        if Data["mask"] is not None:
            Mask = Data.pop("mask")
            ResizedMask = F.resize(Mask, pair(Size), T.InterpolationMode.NEAREST, antialias=True)
            Data["mask"] = ResizedMask

    if "box_coordinates" in Data:
        Boxes = Data.pop("box_coordinates")
        Boxes[:, 0::2] *= 1.0 * SizeW / W
        Boxes[:, 1::2] *= 1.0 * SizeH / H
        Data["box_coordinates"] = Boxes

    if "instance_mask" in Data:
        assert "instance_coords" in Data

        InsMasks = Data.pop("instance_mask")

        ResizedInsMasks = F.resize(InsMasks, pair(Size), T.InterpolationMode.NEAREST, antialias=True)
        Data["instance_mask"] = ResizedInsMasks

        InsCoords = Data.pop("instance_coords")
        InsCoords = InsCoords.astype(np.float)
        InsCoords[..., 0::2] *= 1.0 * SizeW / W
        InsCoords[..., 1::2] *= 1.0 * SizeH / H
        Data["instance_coords"] = InsCoords
    return Data


def centerCropFn(Data: Dict, output_size: List[int]) -> Dict:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(centerCropFn)
        
    Img = Data.pop("image")
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    W, H = F.get_image_size(Img)
    crop_height, crop_width = output_size

    if crop_width > W or crop_height > H:
        padding_ltrb = [
            (crop_width - W) // 2 if crop_width > W else 0,
            (crop_height - H) // 2 if crop_height > H else 0,
            (crop_width - W + 1) // 2 if crop_width > W else 0,
            (crop_height - H + 1) // 2 if crop_height > H else 0,
        ]
        Img = F.pad(Img, padding_ltrb, fill=0)  # PIL uses fill value 0
        W, H = F.get_image_size(Img)
        if crop_width == W and crop_height == H:
            Data["image"] = Img
            return Data

    crop_top = int(round((H - crop_height) / 2.0))
    crop_left = int(round((W - crop_width) / 2.0))
    
    Data["image"] = Img
    return cropFn(Data, crop_top, crop_left, crop_height, crop_width)


def resizedCropFn(Data: Dict, top: int, left: int, height: int, width: int, size: List[int], 
                  Interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR) -> Dict:
    """Crop the given image and resize it to desired size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(resizedCropFn)
    Data = cropFn(Data, top, left, height, width)
    Data = resizeFn(Data, size, Interpolation)
    return Data
