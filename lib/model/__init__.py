from typing import Any
from .segmentation import buildSegmentationModel

SUPPORTED_TASKS = ["segmentation"]

def getModel(opt, **kwargs: Any):
    return buildSegmentationModel(opt, **kwargs)