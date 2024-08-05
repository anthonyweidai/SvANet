from typing import Any

from ..classification import buildClassificationModel
from ...utils import colourText, importModule


ModelRegistry = {}


def registerSegModels(Name):
    def registerModelClass(Cls):
        if Name in ModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))

        ModelRegistry[Name] = Cls
        return Cls
    return registerModelClass


def buildSegmentationModel(opt, **kwargs: Any):
    Model = None
    if opt.seg_model_name in ModelRegistry:
        if 'encoder_decoder' in opt.seg_model_name:
            Encoder = buildClassificationModel(opt)
            Model = ModelRegistry[opt.seg_model_name](opt=opt, Encoder=Encoder, **kwargs)
        else:
            Model = ModelRegistry[opt.seg_model_name](opt=opt, **kwargs)
    else:
        SupportedModels = list(ModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colourText(Name))
    return Model


# automatically import the segmentation models
importModule(__file__, PurePathMode=True)