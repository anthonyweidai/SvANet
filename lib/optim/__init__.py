from .baseOptim import BaseOptim
from ..utils import colourText, importModule


OptimRegistry = {}


def registerOptimiser(Name: str):
    def registerOptimiserCls(Cls):
        if Name in OptimRegistry:
            raise ValueError("Cannot register duplicate optimiser ({})".format(Name))

        if not issubclass(Cls, BaseOptim):
            raise ValueError(
                "Optimiser ({}: {}) must extend BaseOptim".format(Name, Cls.__name__)
            )

        OptimRegistry[Name] = Cls
        return Cls

    return registerOptimiserCls


def buildOptimiser(NetParam, opt) -> BaseOptim:
    optimiser = None
    
    if opt.optim in OptimRegistry:
        optimiser = OptimRegistry[opt.optim](opt, NetParam)
    else:
        SupList = list(OptimRegistry.keys())
        SupStr = (
            "Optimiser ({}) not yet supported. \n Supported optimisers are:".format(
                opt.optim
            )
        )
        for i, Name in enumerate(SupList):
            SupStr += "\n\t {}: {}".format(i, colourText(Name))

    return optimiser


# automatically import the optimisers
importModule(__file__, PurePathMode=True)
