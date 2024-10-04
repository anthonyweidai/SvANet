import copy
from typing import Union, Tuple, Dict, Any

from torch import Tensor

from . import registerSegModels
from .baseSeg import BaseSegmentation
from .heads import buildSegmentationHead
from ..layers import computeMaxStage


@registerSegModels("encoder_decoder")
class SegEncoderDecoder(BaseSegmentation):
    """
    This class defines a encoder-decoder architecture for the task of semantic segmentation. 
    Different segmentation heads (e.g., PSPNet and DeepLabv3) can be used

    Args:
        opt: command-line arguments
        Encoder (BaseEncoder): Backbone network (e.g., ResNext)
    """

    def __init__(self, opt, Encoder, **kwargs: Any) -> None:
        super().__init__(opt, Encoder, **kwargs)
        # delete layers that are not required in segmentation network
        del self.Encoder.Classifier

        ModelConfigDict = copy.deepcopy(self.Encoder.ModelConfigDict) # no change in seghead
        if opt.seg_feature_guide and opt.fg_nostage5:
            MaxStage = computeMaxStage(self.Encoder.ModelConfigDict)
            for k, v in self.Encoder.ModelConfigDict.items():
                if v['stage'] == MaxStage:
                    del ModelConfigDict[k]
                    delattr(self.Encoder, k.capitalize()) # delete method
            self.Encoder.ModelConfigDict = ModelConfigDict
        
        self.SegHead = buildSegmentationHead(opt, ModelConfigDict=ModelConfigDict)

    def forward(self, x: Tensor, **kwargs) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if isinstance(x, Dict):
            Input = x["image"]
        elif isinstance(x, Tensor):
            Input = x
        else:
            raise NotImplementedError(
                "Input to segmentation should be either a Tensor or a Dict of Tensors"
            )
        
        FeaturesTuple = self.Encoder.forwardTuple(Input)
        return self.SegHead(FeaturesTuple, **kwargs)