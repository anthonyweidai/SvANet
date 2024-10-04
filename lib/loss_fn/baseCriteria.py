import argparse
from typing import Any
import torch
from torch import nn, Tensor


class BaseCriteria(nn.Module):
    def __init__(self, opt, Task=None, **kwargs: Any):
        super(BaseCriteria, self).__init__()
        if 'classification' in Task:
            self.NumClasses = opt.cls_num_classes
        elif 'segmentation' in Task:
            self.NumClasses = opt.seg_num_classes
        else:
            self.NumClasses = opt.num_classes
            
        self.UseClsWts = opt.class_weights
        self.LabelSmoothing = opt.label_smoothing
        self.Reduction = opt.loss_reduction
        
        self.eps = 1e-7
        self.opt = opt

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(self, Input: Any, Prediction: Any, Target: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def classWeights(Target: Tensor, NumClasses: int, NormVal: float = 1.1) -> Tensor:
        ClassHist = torch.histc(Target.float(), bins=NumClasses, min=0, max=NumClasses - 1)
        MaskIndices = ClassHist == 0

        # normalize between 0 and 1 by dividing by the sum
        NormHist = torch.div(ClassHist, ClassHist.sum())
        NormHist = torch.add(NormHist, NormVal)

        # compute class weights.
        # samples with more frequency will have less weight and vice-versa
        ClassWts = torch.div(torch.ones_like(ClassHist), torch.log(NormHist))

        # mask the classes which do not have samples in the current batch
        ClassWts[MaskIndices] = 0.0

        return ClassWts.to(Target.device)

    def weightForward(self, Target: Tensor):
        # manual rescaling weight for each class, passed to binary Cross-Entropy loss
        Weight = None
        if self.UseClsWts and self.training:
            Weight = self.classWeights(Target, self.NumClasses)
        return Weight
    
    def lossRedManager(self, Loss: Tensor):
        if self.Reduction == 'mean':
            Loss = Loss.mean()
        elif self.Reduction == 'sum':
            Loss = Loss.sum()  
        return Loss

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
