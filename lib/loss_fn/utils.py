import torch
from torch import Tensor


def mixCriterion(opt, LossFn, Input, Prediction, Target, Epoch):
    if isinstance(Prediction, Tensor) or not opt.loss_coeff:
        return LossFn(Input, Prediction, Target), Prediction
    elif isinstance(Prediction, list) or isinstance(Prediction, tuple):
        LossList = [Factor * LossFn(Input[i], Prediction[i], Target[i]) for i, Factor in enumerate(opt.loss_coeff)]
        return torch.stack(LossList, dim=0).sum(dim=0), Prediction[-1]
    else:
        raise NotImplementedError("Not supporting prediction data type {}".format(type(Prediction)))