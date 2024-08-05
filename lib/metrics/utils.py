import numpy as np

import torch
from torch import Tensor


def makeConsistentDim(Prediction: Tensor, Target: Tensor):
    # for segmentation
    if Prediction.ndim != Target.ndim:
        if Prediction.ndim == 4:
            Prediction = torch.argmax(Prediction, dim=1)
        else:
            Target = torch.argmax(Target, dim=1)
    
    return Prediction, Target


def outDataTypeManage(Metric):
    # not used
    if isinstance(Metric, Tensor):
        Metric = Metric.cpu().numpy()
    return Metric


def manageMetricMM(MetricVal, UnionCout, MetricsType="micro", Scale=1.0):
    # manage the micro and macro types of metric
    if "macro" in MetricsType: # 'macro'
        AvgVal = np.mean(MetricVal)
    else: # 'micro'
        AvgVal = np.sum(MetricVal * UnionCout.T) / np.sum(UnionCout)
    return AvgVal * Scale
