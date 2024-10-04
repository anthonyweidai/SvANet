from typing import Any

from .baseSegHead import BaseModule
from ....utils import colourText, importModule


SegHeadRegistry = {}


def registerSegmentationHead(Name):
    def registerModelCls(Cls):
        if Name in SegHeadRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))

        if not issubclass(Cls, BaseModule):
            raise ValueError(
                "Model ({}: {}) must extend BaseSegHead".format(Name, Cls.__Name__)
            )

        SegHeadRegistry[Name] = Cls
        return Cls
    return registerModelCls


def buildSegmentationHead(opt, **kwargs: Any) -> BaseModule:
    SegHead = None
    if opt.seg_head_name in SegHeadRegistry:
        SegHead = SegHeadRegistry[opt.seg_head_name](opt, **kwargs)
    else:
        SupportedHeads = list(SegHeadRegistry.keys())
        SuppModelStr = "Supported segmentation heads are:"
        for i, Name in enumerate(SupportedHeads):
            SuppModelStr += "\n\t {}: {}".format(i, colourText(Name))
    return SegHead


# automatically import the segmentation heads
importModule(__file__, PurePathMode=True)