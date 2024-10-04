from torch import nn, Tensor
from typing import Optional, Tuple, Any

from ..layers import NormLayerTuple
from ..modules import BaseEncoder


class BaseSegmentation(nn.Module):
    """Base class for segmentation networks"""

    def __init__(
        self, opt, 
        Encoder: Optional[BaseEncoder],
        **kwargs: Any,
    ) -> None:
        super(BaseSegmentation, self).__init__()
        assert isinstance(
            Encoder, BaseEncoder
        ), "Encoder should be an instance of BaseEncoder or BaseViTEncoder"
        self.opt = opt
        self.Encoder: BaseEncoder = Encoder
    
    def profileModel(self, Input: Tensor) -> Optional[Tuple[Tensor, float, float]]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def freezeNormLayers(self) -> None:
        for m in self.modules():
            if isinstance(m, NormLayerTuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False