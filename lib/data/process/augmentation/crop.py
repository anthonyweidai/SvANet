import random
import numpy as np
from typing import Dict

from torchvision import transforms as T
import torchvision.transforms.functional as F

from .utils import padFn, cropFn, centerCropFn


class RandomCrop(T.RandomCrop):
    # not in detection (with bounding box)
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
    
    def forward(self, Data: Dict) -> Dict:
        if self.padding is not None:
            Data = padFn(Data, self.padding, self.fill, self.padding_mode)

        # _, height, width = F.get_dimensions(Data["image"])
        W, H = F.get_image_size(Data["image"])
        # pad the width if needed
        if self.pad_if_needed and W < self.size[1]:
            padding = [self.size[1] - W, 0]
            Data = padFn(Data, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and H < self.size[0]:
            padding = [0, self.size[0] - H]
            Data = padFn(Data, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(Data["image"], self.size)

        return cropFn(Data, i, j, h, w)
   

class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, Data: Dict) -> Dict:
        return centerCropFn(Data, self.size)


def intersect(box_a, box_b):
    """Computes the intersection between box_a and box_b"""
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccardNumpy(box_a: np.ndarray, box_b: np.ndarray):
    """
    Computes the intersection of two boxes.
    Args:
        box_a (np.ndarray): Boxes of shape [Num_boxes_A, 4]
        box_b (np.ndarray): Box osf shape [Num_boxes_B, 4]

    Returns:
        intersection over union scores. Shape is [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class SSDCroping(object):
    """
    This class implements cropping method for `Single shot object detector <https://arxiv.org/abs/1512.02325>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.iou_sample_opts = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.trials = 40
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 2.0

    def __call__(self, Data: Dict) -> Dict:
        if "box_coordinates" in Data:
            boxes = Data["box_coordinates"]

            # guard against no boxes
            if boxes.shape[0] == 0:
                return Data

            image = Data["image"]
            labels = Data["box_labels"]
            width, height = F.get_image_size(image)

            while True:
                # randomly choose a mode
                min_jaccard_overalp = random.choice(self.iou_sample_opts)
                if min_jaccard_overalp == 0.0:
                    return Data

                for _ in range(self.trials):
                    new_w = int(random.uniform(0.3 * width, width))
                    new_h = int(random.uniform(0.3 * height, height))

                    aspect_ratio = new_h / new_w
                    if not (
                        self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
                    ):
                        continue

                    left = int(random.uniform(0, width - new_w))
                    top = int(random.uniform(0, height - new_h))

                    # convert to integer rect x1,y1,x2,y2
                    rect = np.array([left, top, left + new_w, top + new_h])

                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    ious = jaccardNumpy(boxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if ious.max() < min_jaccard_overalp:
                        continue

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # if image size is too small, try again
                    if (rect[3] - rect[1]) < 100 or (rect[2] - rect[0]) < 100:
                        continue

                    # cut the crop from the image
                    image = F.crop(image, top=top, left=left, width=new_w, height=new_h)

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # take only matching gt labels
                    current_labels = labels[mask]

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    Data["image"] = image
                    Data["box_labels"] = current_labels
                    Data["box_coordinates"] = current_boxes

                    if "mask" in Data:
                        if Data["mask"] is not None:
                            Mask = Data.pop("mask")
                            Data["mask"] = F.crop(
                                Mask, top=top, left=left, height=new_h, width=new_w,
                            )

                    if "instance_mask" in Data:
                        assert "instance_coords" in Data
                        instance_masks = Data.pop("instance_mask")
                        Data["instance_mask"] = F.crop(
                            instance_masks,
                            top=top,
                            left=left,
                            width=new_w,
                            height=new_h,
                        )

                        instance_coords = Data.pop("instance_coords")
                        # should we use the box left and top corner or the crop's
                        instance_coords[..., :2] = np.maximum(
                            instance_coords[..., :2], rect[:2]
                        )
                        # adjust to crop (by substracting crop's left,top)
                        instance_coords[..., :2] -= rect[:2]

                        instance_coords[..., 2:] = np.minimum(
                            instance_coords[..., 2:], rect[2:]
                        )
                        # adjust to crop (by substracting crop's left,top)
                        instance_coords[..., 2:] -= rect[:2]
                        Data["instance_coords"] = instance_coords

                    return Data
        return Data
        