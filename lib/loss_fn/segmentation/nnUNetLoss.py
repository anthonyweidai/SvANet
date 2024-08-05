from typing import Any, Tuple, Union

import torch
from torch import Tensor

from . import registerSegLossFn
from .diceLoss import SoftDiceLoss
from .segCrossEntropy import SegCrossEntropy
from ..baseCriteria import BaseCriteria


@registerSegLossFn("nnunet")
class CDWithCE(BaseCriteria):
    " Dice + cross-entropy loss "
    def __init__(self, opt, **kwargs: Any):
        super(CDWithCE, self).__init__(opt, **kwargs)
        self.weight_dice = 1
        self.weight_ce = 1
        self.IgnoreIdx = opt.ignore_idx
        
        self.CrossEntropyLoss = SegCrossEntropy(opt, **kwargs)
        self.DiceLoss = SoftDiceLoss(opt, **kwargs)
    
    def forward(self, Input: Tensor, Prediction: Union[Tensor, Tuple[Tensor, Tensor]], Target: Tensor) -> Tensor:
        if self.IgnoreIdx is not None:
            Mask = Target != self.IgnoreIdx
            # remove ignore label from Target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            TargetDice = torch.where(Mask, Target, 0)
        else:
            TargetDice = Target
            Mask = None
            
        return self.weight_ce * self.CrossEntropyLoss(Input, Prediction, Target) + \
            self.weight_dice * self.DiceLoss(Input, Prediction, TargetDice, loss_mask=Mask)