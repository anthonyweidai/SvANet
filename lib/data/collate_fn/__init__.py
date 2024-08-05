from typing import Any
from ...utils import colourText, importModule


CollateFnRegistry = {}


def registerCollateFn(Name):
    def registerCollateFnMethod(f):
        if Name in CollateFnRegistry:
            raise ValueError(
                "Cannot register duplicate collate function ({})".format(Name)
            )
        CollateFnRegistry[Name] = f
        return f

    return registerCollateFnMethod


def buildCollateFn(opt, **kwargs: Any):
    CollateFn = None
    if opt.collate_fn_name in CollateFnRegistry:
        CollateFn = CollateFnRegistry[opt.collate_fn_name]
    else:
        SupportedCollateFn = list(CollateFnRegistry.keys())
        SuppModelStr = "Supported collate function are:"
        for i, Name in enumerate(SupportedCollateFn):
            SuppModelStr += "\n\t {}: {}".format(i, colourText(Name))

    return CollateFn


# automatically import collate function
importModule(__file__)