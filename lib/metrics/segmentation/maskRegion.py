import numpy as np
from typing import Optional, Tuple, Union, Dict, List

import torch
from torch import Tensor


def computeMaskRegion(
    Prediction: Union[Tuple[Tensor, Tensor], Tensor],
    Target: Tensor,
    Eps: Optional[float] = 1e-7,
):
    # for Jaccard/iou, dice
    # manage mask data structure
    if isinstance(Prediction, Dict):
        Prediction = Prediction['mask']
    if isinstance(Target, Dict):
        Target = Target['mask']
    
    if isinstance(Prediction, (Tuple, List)) and len(Prediction) == 2:
        Mask = Prediction[0]
        assert isinstance(Mask, Tensor)
    elif isinstance(Prediction, Tensor):
        Mask = Prediction
        assert isinstance(Mask, Tensor)

    NumClasses = Mask.shape[1]
    # PredMask = torch.max(Mask, dim=1)[1]
    PredMask = torch.argmax(Mask, dim=1)
    if Target.ndim == 4:
        Target = torch.argmax(Target, dim=1)
    assert (
        PredMask.dim() == 3
    ), "Predicted Mask tensor should be 3-dimensional (B x H x W)"

    PredMask = PredMask.byte()
    Target = Target.byte()
    
    # shift by 1 so that 255 is 0
    PredMask += 1
    Target += 1

    PredMask = PredMask * (Target > 0)

    # calculate mask regions
    Inter = PredMask * (PredMask == Target)
    AreaInter = Inter.float().histc(bins=NumClasses, min=1, max=NumClasses)
    # torch.histc(Inter.float(), bins=NumClasses, min=1, max=NumClasses)
    AreaGT = Target.float().histc(bins=NumClasses, min=1, max=NumClasses)
    AreaPred = PredMask.float().histc(bins=NumClasses, min=1, max=NumClasses)
    AreaUnion = AreaGT + AreaPred - AreaInter + Eps
    
    # calculate unioncount (semantic level)
    UnionCout = np.zeros((NumClasses, 1), dtype=int)
    NpTarget = Target.cpu().numpy()
    Areas = np.apply_along_axis(lambda a: np.histogram(a, bins=NumClasses, range=(1, NumClasses))[0], 
                                1, NpTarget.reshape(*NpTarget.shape[:-2], -1))
    UnionIdx = np.where(Areas > 1e-5, 1, 0)
    UnionCout += np.expand_dims(np.sum(UnionIdx, axis=0), axis=-1)
    
    return (
        AreaPred.cpu().numpy(), 
        AreaInter.cpu().numpy(), 
        AreaUnion.cpu().numpy(), 
        UnionCout
    )