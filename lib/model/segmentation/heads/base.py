from typing import Any, Tuple, Union

from torch import nn, Tensor
from torch.nn import functional as F

from .utils import HEAD_OUT_CHANNELS
from ...layers import (
    BaseConv2d, Dropout2d,
    checkExp, getLastIdxFromStage, 
    getChannelsbyLayer, getChannelsbyStage, 
    computeMaxStage, getAllLayerIndex
)
from ....utils import pair


'''separate base modules and fg modules'''


class BaseModule(nn.Module):
    def __init__(self, opt, ModelConfigDict, **kwargs: Any):
        super().__init__()
        self.opt = opt
        self.NumClasses = opt.seg_num_classes
        self.ModelConfigDict = ModelConfigDict
            
        self.AuxHead = None
        if opt.use_aux_head:
            self.StageIdx = self.getLastIdxFromStage(ModelConfigDict, -2)
            EncLast2Channels = ModelConfigDict[list(ModelConfigDict)[self.StageIdx]]['out']
            InnerChannels = max(int(EncLast2Channels // 4), 128)
            self.AuxHead = nn.Sequential(
                BaseConv2d(EncLast2Channels, InnerChannels, 3, 1, BNorm=True, ActLayer=nn.ReLU),
                Dropout2d(0.1),
                BaseConv2d(InnerChannels, self.NumClasses, 1, 1, ActLayer=None)
                )
        
        self.checkExp = checkExp
        self.getLastIdxFromStage = getLastIdxFromStage
        self.getChannelsbyLayer = getChannelsbyLayer
        self.getChannelsbyStage = getChannelsbyStage
        self.computeMaxStage = computeMaxStage
        self.getAllLayerIndex = getAllLayerIndex
        
        self.Classifier = None
        # default out channels
        self.OutChannels = HEAD_OUT_CHANNELS.get(opt.seg_head_name, HEAD_OUT_CHANNELS['default'])
        
    def forwardAuxHead(self, FeaturesTuple) -> Tensor:
        AuxOut = self.AuxHead(FeaturesTuple[self.StageIdx])
        return AuxOut
    
    def forwardDecode(self, FeaturesTuple) -> Tensor:
        raise NotImplementedError

    def forwardSegHead(self, FeaturesTuple: list) -> Tensor: 
        x = self.forwardDecode(FeaturesTuple)
        x = self.Classifier(x)
        return F.interpolate(x, size=pair(self.opt.resize_shape), mode="bilinear", align_corners=True)

    def forward(self, FeaturesTuple, **kwargs) -> Union[Tensor, Tuple[Tensor]]:
        if self.opt.seg_feature_guide == 0:
            Out = self.forwardSegHead(FeaturesTuple)
        else:
            Out = self.forwardFMG(FeaturesTuple)

        if self.AuxHead is not None and self.training:
            AuxOut = self.forwardAuxHead(FeaturesTuple)
            return Out, AuxOut
        return Out

    def profileModule(self, x: Tensor) -> Tuple[Tensor, float, float]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def initFeatureMapGuide(self):
        raise NotImplementedError
    
    def forwardFMG(self, FeaturesTuple) -> Union[Tensor, Tuple[Tensor]]:
        raise NotImplementedError

    def initLiteFeatureMapGuide(self):
        raise NotImplementedError
    
    def forwardLiteFMG(self, FeaturesTuple) -> Union[Tensor, Tuple[Tensor]]:
        raise NotImplementedError