from torch import nn, Tensor
from torchvision.ops import StochasticDepth as StochasticDepthTorch


class StochasticDepth(StochasticDepthTorch):
    def __init__(self, p: float, Mode: str="row") -> None:
        super().__init__(p, Mode)

    def profileModule(self, Input: Tensor):
        _, in_channels, in_h, in_w = Input.size()
        MACs = in_channels * in_h * in_w # one multiplication for each element
        return Input, 0.0, MACs
    
    
class Dropout(nn.Dropout):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0


class Dropout2d(nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor):
        return input, 0.0, 0.0
