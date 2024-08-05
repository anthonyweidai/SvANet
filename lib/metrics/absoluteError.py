import torch
from torch import Tensor

from .utils import makeConsistentDim


# implementation follows https://pytorch.org/ignite/_modules/ignite/metrics/mean_absolute_error.html#MeanAbsoluteError
def computeAE(Prediction: Tensor, Target: Tensor):
    # compute absolute errors
    # difficult to use pure micro, cause it relates to batch size
    Prediction, Target = makeConsistentDim(Prediction, Target)
    
    AbsoluteErrors = torch.abs(Prediction - Target.view_as(Prediction))
    return AbsoluteErrors.float().mean().cpu().numpy()