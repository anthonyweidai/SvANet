from typing import Union
from torch import nn, Tensor


def moduleProfile(module, x: Tensor) -> Union[Tensor, float, float]:
    # Note: Module profiling is for reference only and may contain errors.
    # Relies on user to implement these functions accurately.

    if isinstance(module, nn.Sequential):
        NumMACs = NumParams = 0.0
        for l in module:
            try:
                x, l_p, l_macs = l.profileModule(x)
                NumMACs += l_macs
                NumParams += l_p
            except Exception as e:
                pass     
    else:
        x, NumParams, NumMACs = module.profileModule(x)

    return x, NumParams, NumMACs