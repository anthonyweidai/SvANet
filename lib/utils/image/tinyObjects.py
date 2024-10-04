import numpy as np

from .utils import getAllMaskArea
from ..path_manage.utils import replacedWithMask
from ..mathematics.utils import inhomogeneousArithmetic


def getKeepTinyIds(ImgPaths, TinyRatios, KeepEmpty=True, Eps=1e-7):
    # get the ratio of kept tiny objects
    ## replace multiple substrings to a string
    MaskPaths = replacedWithMask(ImgPaths)
    
    ## remove out of distribution mask data
    print("getting all mask area")
    AreaRatios = getAllMaskArea(MaskPaths=MaskPaths)
    AreaRatios = inhomogeneousArithmetic(AreaRatios, "max")
    if KeepEmpty:
        KeepLogic = AreaRatios < max(TinyRatios)
    else:
        KeepLogic = (AreaRatios < max(TinyRatios)) & (AreaRatios > Eps)
    KeepIds = np.where(KeepLogic)[0]
    
    return AreaRatios, KeepIds


def getTinyInstances(TinyRatios, AreaRatios, Eps=1e-7):
    # get the tiny instances sorted by area ratios
    AreaRatios = AreaRatios[AreaRatios > Eps]
    
    TinyRatios = np.asarray(TinyRatios)
    TinyInstances = dict.fromkeys(TinyRatios.astype('str'))
    for r in TinyRatios:
        TinyInstances[str(r)] = np.where(AreaRatios <= r)[0].size
        
    return TinyInstances