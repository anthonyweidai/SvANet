from torch import Tensor

from .utils import makeConsistentDim


def measureEAM(Prediction: Tensor, Target: Tensor, Eps=1e-7):
    ''' paper: https://arxiv.org/abs/1805.10421
    adapted from https://github.com/DengPingFan/PraNet
    abbreviated by EAM, enhanced alignment metric
    '''
    Prediction, Target = makeConsistentDim(Prediction, Target)
    
    Prediction[Prediction.float().abs() > Eps] = 1.
    Target[Target.float().abs() > Eps] = 1.
    
    if Target.float().max() == 0:
        # the GT is completely black
        # only calculate the black area of intersection
        EMatrix = 1.0 - Prediction
    elif Target.float().min() == 255:
        # the GT is completely white
        # only calcualte the white area of intersection
        EMatrix = Prediction
    else:
        AlignGT = Target - Target.float().mean()
        AlignPre = Prediction - Prediction.float().mean()
        
        AlignMatrix = 2 * AlignGT * AlignPre / (AlignGT ** 2 + AlignPre ** 2 + Eps)
        EMatrix = (AlignMatrix + 1) ** 2 / 4
    
    H, W = Target.shape[1:]
    Em = EMatrix.sum() / (W * H - 1 + Eps)
    return Em.cpu().numpy()
