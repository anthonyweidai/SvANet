from typing import Any, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from . import registerSegLossFn
from ..baseCriteria import BaseCriteria


@registerSegLossFn("cross_entropy")
class SegCrossEntropy(BaseCriteria):
    """Cross entropy loss for the task of semantic segmentation"""

    def __init__(self, opt, **kwargs: Any):
        super(SegCrossEntropy, self).__init__(opt, **kwargs)
        self.AuxWt = opt.aux_weight

    def computeLoss(self, PredMask: Tensor, TargetMask: Tensor, Weight=None):
        if TargetMask.ndim == 4:
            TargetMask = torch.argmax(TargetMask, dim=1)
        
        _, _, x_h, x_w = PredMask.shape
        _, y_h, y_w = TargetMask.shape

        # use label smoothing only for training
        LabelSmoothing = self.LabelSmoothing if self.training else 0.0

        if x_h != y_h or x_w != y_w:
            PredMask = F.interpolate(
                PredMask, size=(y_h, y_w), mode="bilinear", align_corners=True
            )

        return F.cross_entropy(
            input=PredMask,
            target=TargetMask,
            weight=Weight,
            ignore_index=self.opt.ignore_idx,
            label_smoothing=LabelSmoothing,
        )

    def forward(self, Input: Tensor, Prediction: Union[Tensor, Tuple[Tensor, Tensor]], Target: Tensor) -> Tensor:
        AuxOut = None
        if isinstance(Prediction, Tuple) and len(Prediction) == 2:
            Mask, AuxOut = Prediction
            assert isinstance(Mask, Tensor)
            assert isinstance(AuxOut, Tensor)
        elif isinstance(Prediction, Tensor):
            Mask = Prediction
            assert isinstance(Mask, Tensor)
        else:
            raise NotImplementedError(
                "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor"
            )

        if self.training:
            Weight = self.weightForward(Target)
            TotalLoss = self.computeLoss(
                PredMask=Mask, TargetMask=Target, Weight=Weight
            )

            if AuxOut is not None:
                LossAux = self.computeLoss(
                    PredMask=AuxOut, TargetMask=Target, Weight=Weight
                )
                TotalLoss = TotalLoss + (self.AuxWt * LossAux)
            return TotalLoss
        else:
            return self.computeLoss(PredMask=Mask, TargetMask=Target)

    def __repr__(self):
        repr_str = (
            "{}(\n\tweighted_loss={}\n\tignore_idx={}\n\tlabel_smoothing={}".format(
                self.__class__.__name__,
                self.UseClsWts,
                self.opt.ignore_idx,
                self.LabelSmoothing,
            )
        )

        if self.AuxWt > 0:
            repr_str += "\n\taux_wt={}".format(self.AuxWt)
        return repr_str + "\n)"
