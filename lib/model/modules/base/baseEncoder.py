from typing import Any

import torch
from torch import nn, Tensor

from ...layers import AdaptiveAvgPool2d
from ....utils import callMethod


class BaseEncoder(nn.Module):
    def __init__(self, opt, **kwargs: Any):
        super(BaseEncoder, self).__init__()
        self.opt = opt
        self.NumClasses = opt.cls_num_classes
        self.Classifier = None
        self.AvgPooling = AdaptiveAvgPool2d(1)
        
        self.ModelConfigDict = dict()
        '''
        Key: conv1, layer*no.
        Values: 'in': inchannels, 'out': outchannels, 'stage': stride stage
        '''

    def checkModel(self):
        assert self.Classifier is not None, 'Please implement self.Classifier'
        
    def forwardByConfig(self, x: Tensor) -> list:
        # may not update self layers' weight
        pass
    
    def forwardTuple(self, x: Tensor) -> list:
        # feature tuple occupies the same memory wih pin_memory is ture
        FeaturesTuple = list()
        for Key in self.ModelConfigDict:
            Layer = callMethod(self, Key.capitalize())
            x = Layer(x)
            FeaturesTuple.append(x)
        return FeaturesTuple
    
    def forwardFeatures(self, x: Tensor) -> Tensor:
        return self.forwardTuple(x)[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.forwardFeatures(x)
        x = self.AvgPooling(x)
        x = torch.flatten(x, 1)
        x = self.Classifier(x)
        return x