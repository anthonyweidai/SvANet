from typing import Any

from ...utils import colourText, importModule


LossFnRegisty = {}


def registerSegLossFn(Name):
    def registerLossFnCls(Fn):
        if Name in LossFnRegisty:
            raise ValueError("Cannot register duplicate segmentation loss function ({})".format(Name))
        LossFnRegisty[Name] = Fn
        return Fn
    return registerLossFnCls


def buildSegmentationLossFn(opt, **kwargs: Any):
    LossFn = None
    if opt.seg_loss_name in LossFnRegisty:
        LossFn = LossFnRegisty[opt.seg_loss_name](opt=opt, **kwargs)
    else:
        TempList = list(LossFnRegisty.keys())
        TempStr = "Supported loss functions are:"
        for i, Name in enumerate(TempList):
            TempStr += "\n\t {}: {}".format(i, colourText(Name))
    return LossFn


# automatically import different loss functions
importModule(__file__)