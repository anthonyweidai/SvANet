from typing import Any
from torch.optim import Adam
from . import registerOptimiser
from .baseOptim import BaseOptim


@registerOptimiser("adam")
class AdamOptimiser(BaseOptim, Adam):
    """
    `Adam <https://arxiv.org/abs/1412.6980>`_ optimiser
    """
    def __init__(self, opt, NetParam, **kwargs: Any) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        Adam.__init__(
            self,
            params=NetParam,
            lr=self.Lr,
            betas=(opt.beta1, opt.beta2),
            eps=self.Eps,
            weight_decay=opt.weight_decay,
            amsgrad=opt.amsgrad,
        )
