import torch
from torch import nn, Tensor


class Linearlayer(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def profileModule(self, Input: Tensor):
        out_size = list(Input.shape)
        out_size[-1] = self.out_features
        Params = sum([p.numel() for p in self.parameters()])
        MACs = Params
        output = torch.zeros(size=out_size, dtype=Input.dtype, device=Input.device)
        return output, Params, MACs
    