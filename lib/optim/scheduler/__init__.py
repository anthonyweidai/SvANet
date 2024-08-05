from typing import Any
from ...utils import colourText, importModule


SchedularRegistry = {}


def registerScheduler(Name: str):
    def registerSchedulerClass(Cls):
        if Name in SchedularRegistry:
            raise ValueError("Cannot register duplicate scheduler ({})".format(Name))
        SchedularRegistry[Name] = Cls
        return Cls
    return registerSchedulerClass


def buildScheduler(opt, **kwargs: Any):
    LrScheduler = None
    if opt.schedular in SchedularRegistry:
        LrScheduler = SchedularRegistry[opt.schedular](opt, **kwargs)
    else:
        SuppList = list(SchedularRegistry.keys())
        SuppStr = "LR Scheduler ({}) not yet supported. \n Supported schedulers are:".format(opt.schedular)
        for i, m_name in enumerate(SuppList):
            SuppStr += "\n\t {}: {}".format(i, colourText(m_name))

    return LrScheduler


# Automatically import the LR schedulers
importModule(__file__, PurePathMode=True)