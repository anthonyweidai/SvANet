from typing import Union, Dict, Optional
from torch import Tensor


def moveToDevice(
    x: Union[Dict, Tensor],
    Device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    **kwargs
) -> Union[Dict, Tensor]:

    if isinstance(x, Dict):
        # return the tensor because if its already on Device
        if "on_gpu" in x and x["on_gpu"]:
            return x

        for k, v in x.items():
            if isinstance(v, Dict):
                x[k] = moveToDevice(x=v, Device=Device)
            elif isinstance(v, Tensor):
                x[k] = v.to(Device, non_blocking=non_blocking)

    elif isinstance(x, Tensor):
        x = x.to(Device, non_blocking=non_blocking)
    else:
        print(
            "Inputs of type Tensor or Dict of Tensors are only supported right now"
        )
    return x