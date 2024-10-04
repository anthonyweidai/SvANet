from torch import nn, Tensor
from typing import Optional, Tuple, Union


class UpSample(nn.Upsample):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[float] = None,
        mode: Optional[str] = "nearest",
        align_corners: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        input = self.forward(input)
        return input, 0.0, 0.0
