from ..utils import importModule, colourText


ParamsInitRegistry = {}


def registerParams(Name):
    def registerParamsClass(Cls):
        if Name in ParamsInitRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        ParamsInitRegistry[Name] = Cls
        return Cls
    return registerParamsClass


def singleInit(opt):
    if opt.task in ParamsInitRegistry:
        opt = ParamsInitRegistry[opt.task](opt)
    else:
        SupportedMethods = list(ParamsInitRegistry.keys())
        SuppStr = "Supported models are:"
        for i, Name in enumerate(SupportedMethods):
            SuppStr += "\n\t {}: {}".format(i, colourText(Name))
    return opt


importModule(__file__)