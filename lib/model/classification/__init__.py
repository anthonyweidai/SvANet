from typing import Any
from ..modules import BaseEncoder
from ...utils import colourText, importModule


ModelRegistry = {}


def registerClsModels(Name):
    def registerModelClass(Cls):
        if Name in ModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        ModelRegistry[Name] = Cls
        return Cls
    return registerModelClass


def buildClassificationModel(opt, **kwargs: Any):
    Model: BaseEncoder = None
    if opt.model_name in ModelRegistry:
        Model = ModelRegistry[opt.model_name](opt=opt, **kwargs)
    else:
        SupportedModels = list(ModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colourText(Name))
    
    return Model


# automatically import the models
importModule(__file__)