from typing import Any

from ..utils import colourText, importModule


SEPARATOR = ":"
DatasetRegistry = {}


def registerDataset(Name, Task):
    def registerMethodClass(Cls):
        if Name in DatasetRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        DatasetRegistry[Name + SEPARATOR + Task] = Cls
        return Cls
    return registerMethodClass


def getMyDataset(opt, ImgPaths, TargetSet=None, **kwargs: Any):
    DatasetMethod = None
    if opt.reg_by_name:
        RegName = opt.setname.lower() + SEPARATOR + opt.task
    else:
        RegName = opt.sup_method + SEPARATOR + opt.task
    if RegName in DatasetRegistry:
        DatasetMethod = DatasetRegistry[RegName](opt, ImgPaths, TargetSet=TargetSet, **kwargs)
    else:
        Supported = list(DatasetRegistry.keys())
        SuppStr = "Supported datasets are:"
        for i, Name in enumerate(Supported):
            SuppStr += "\n\t {}: {}".format(i, colourText(Name))
    return DatasetMethod


# Automatically import the dataset classes
importModule(__file__, SkipFolder=['collate_fn', 'process', 'sampler', 'tools'])