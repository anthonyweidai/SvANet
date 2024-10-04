from typing import Any

from torch import nn, Tensor

from . import registerSegmentationHead
from .module import ASPP
from .utils import SegHeadClassifier
from .baseSegHead import BaseSegHead
from ...layers import initWeight


@registerSegmentationHead("deeplabv3")
class DeeplabV3(BaseSegHead):
    """A sementic segmentation system
    This class defines the segmentation head in 
    `DeepLabv3 architecture <https://arxiv.org/abs/1706.05587>`_
    """
    def __init__(self, opt, **kwargs: Any) -> None:
        # as the sampling rate becomes larger, the number of valid filter weights
        AtrousRates =  (12, 24, 36) # (6, 12, 18)
        IsSepConv = opt.use_sep_conv
        DropRate = 0.1
        super().__init__(opt, **kwargs)
        self.Aspp = nn.Sequential()
        self.Aspp.add_module(
            name="aspp_layer",
            module=ASPP(
                self.getChannelsbyStage(self.ModelConfigDict, -1),
                self.OutChannels,
                AtrousRates,
                IsSepConv,
                DropRate,
            ),
        )
        
        ClsInChannels = 0 if opt.fg_for_head else self.OutChannels
        self.Classifier = SegHeadClassifier(self.FMGChannels, ClsInChannels, self.NumClasses)

        if opt.init_weight:
            self.apply(initWeight)
        
        self.OutChannels = self.Classifier.ClsInChannels
            
    def forwardDecode(self, FeaturesTuple: list) -> Tensor:
        # low resolution features
        x = FeaturesTuple[-1]
        # ASPP featues
        return self.Aspp(x)
