from typing import Any, Dict

from ..mydatasetReg import registerDataset
from ..baseDataset import SegBaseDataset


@registerDataset("common", "segmentation")
class SegmentationDataset(SegBaseDataset):
    """
    Dataset class for the PASCAL VOC 2012 dataset
    The structure of PASCAL VOC dataset should be something like this:
        PASCALVOC/mask
        PASCALVOC/train
        PASCALVOC/val
    """
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
    
    def __getitem__(self, Index) -> Dict:
        ImgPath = self.ImgPaths[Index]
        Img = self.readImagePil(ImgPath)
        Mask = self.readMaskPil(self.getMask(ImgPath))
        
        Data = {"image": Img, "mask": Mask, "sample_id": Index}
        Data = self.Transform(Data)
        
        Mask = Data.pop("mask")
        Data["label"] = Mask if Mask is not None else 0
        return Data