import math
from typing import Any
from . import registerScheduler


@registerScheduler("mycosine")
class MyCosineScheduler(object):
    def __init__(self, opt, Optimiser, **kwargs: Any) -> None:
        super(MyCosineScheduler, self).__init__()
        self.opt = opt
        
        self.EpochTemp = 0
        self.Optimiser = Optimiser

        self.WarmupEpoches = max(opt.milestones // 2, 0) if opt.pretrained else opt.milestones
        if self.WarmupEpoches > 0:
            self.WarmupStep = (self.opt.max_lr - opt.warmup_init_lr) / self.WarmupEpoches

        self.Period = opt.milestones if opt.pretrained else opt.epochs

    def getLr(self, Epoch: int) -> float:
        if Epoch == self.opt.milestones and self.opt.pretrained:
            self.EpochTemp = Epoch - 1
            self.WarmupEpoches += Epoch
            self.Period = self.opt.epochs
        
        if Epoch < self.WarmupEpoches:
            CurrLr = self.opt.warmup_init_lr + (Epoch - self.EpochTemp) * self.WarmupStep
        elif Epoch + 1 < self.Period:
            CurrLr = self.opt.lr + 0.5 * (self.opt.max_lr - self.opt.lr) * (1. + math.cos(math.pi * Epoch / self.Period))
        else:
            CurrLr = self.opt.lr
        return max(0.0, CurrLr)
    
    def step(self, Epoch: int):
        Values = self.getLr(Epoch)
        self.Optimiser.param_groups[0]['lr'] = Values

        self._last_lr = [group['lr'] for group in self.Optimiser.param_groups]