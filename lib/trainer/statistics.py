import torch
from torch import Tensor

import numpy as np
import pandas as pd

from ..metrics import (
    manageMetricMM, computeMaskRegion, measureEAM, computeAE,
)


class Statistics(object):
    def __init__(self, opt, State='train') -> None:
        self.opt = opt
        self.State = State
        
        # Converting iou from [0, 1] to [0, 100]
        self.Scale = 100.0
        
        self.paramsInit()
        
    @torch.inference_mode()    
    def paramsInit(self):
        # accuracy for classification and mixupmask
        self.LossList, self.Top1AccuracyList, self.Top5AccuracyList \
            = [[] for _ in range(3)]
        self.ComMetric = 0
        
        # create multiple empty lists or pandas frame
        # Jaccard/IoU, F1 score, dice score
        self.IoUDF, self.DiceDF = [pd.DataFrame() for _ in range(2)]
        self.mIoUList, self.mDiceList = [[] for _ in range(2)]
        
        if 'val' in self.State:
            # save time in training
            # E-measure, absolute errors, hausdorff distance
            (
                self.EmDF, self.AeDF, 
                self.RecallDF, self.PrecisionDF, self.F2ScoreDF
            ) = [pd.DataFrame() for _ in range(5)]
            (
                self.mEMList, self.mAEList, self.AvgRecallList, 
                self.AvgPrecisionList, self.AvgF2ScoreList
            ) = [[] for _ in range(5)]

    @torch.inference_mode()          
    def batchInit(self):
        self.Loss = 0
        self.Epoch = 0
        self.NumTotal = 0
        self.RunningLoss = 0
        self.ComMetric = 0 # for save the best model
        self.NumTop1Correct = 0 # for classification and mixupmask
        self.NumTop5Correct = 0 # for classification

        if 'segmentation' in self.opt.task:
            self.UnionCout = np.zeros((self.opt.seg_num_classes, 1), dtype=int)
            self.AreaPred, self.AreaInter, self.AreaUnion, self.IoU, self.Dice = \
                np.zeros((5, self.opt.seg_num_classes), dtype=float)
            
            if 'val' in self.State:
                self.Em, self.Ae = np.zeros((2, 1), dtype=float) # without class restriction
                self.Recall, self.Precision, self.F2Score \
                    = np.zeros((3, self.opt.seg_num_classes), dtype=float) 

    @torch.inference_mode()
    def batchUpdate(self, PredLabel: Tensor, TargetLabel: Tensor, Loss: Tensor, Epoch: int):
        if isinstance(PredLabel, Tensor):
            BatchSize = PredLabel.shape[0] # Could be unequal to the default batchsize
        elif isinstance(PredLabel, dict):
            # Could be unequal to the default batchsize,
            # simsiam takes 2 view of same image as one pair in the batch
            BatchSize = list(PredLabel.values())[0].shape[0] 
        elif isinstance(PredLabel, list):
            # for barlow
            BatchSize = PredLabel[0].shape[0] 
        
        self.Epoch = Epoch
        self.NumTotal += BatchSize
        self.RunningLoss += Loss.item() if Loss is not None else 0
        
        AreaPred, AreaInter, AreaUnion, UnionCout = computeMaskRegion(PredLabel, TargetLabel)
        
        # assign values
        self.AreaPred += AreaPred
        self.AreaInter += AreaInter
        self.AreaUnion += AreaUnion
        self.UnionCout += UnionCout
        
        if 'val' in self.State:
            self.Em += measureEAM(PredLabel, TargetLabel)
            self.Ae += computeAE(PredLabel, TargetLabel)
                
    @torch.inference_mode()
    def update(self):
        Eps = 1e-7
        self.Loss = self.RunningLoss / max(self.NumTotal, 1) # for val start point
        self.LossList.append(self.Loss)
                     
        self.IoU = self.AreaInter / self.AreaUnion # summation wihtin an epoch
        # self.IoU += np.expand_dims(self.AreaInter / self.AreaUnion, axis=1)
        self.Dice = (2 * self.AreaInter) / (self.AreaUnion + self.AreaInter)
        
        self.IoUDF = pd.concat([self.IoUDF, pd.DataFrame(self.IoU * self.Scale).T])
        self.DiceDF = pd.concat([self.DiceDF, pd.DataFrame(self.Dice * self.Scale).T])
        
        self.mIoU = manageMetricMM(self.IoU, self.UnionCout, self.opt.metric_type, self.Scale)
        self.mDice = manageMetricMM(self.Dice, self.UnionCout, self.opt.metric_type, self.Scale)
            
        self.ComMetric = self.mIoU
        
        self.mIoUList.append(self.mIoU)
        self.mDiceList.append(self.mDice)
        
        if 'val' in self.State:
            self.EmDF = pd.concat([self.EmDF, pd.DataFrame(self.Em).T])
            self.AeDF = pd.concat([self.AeDF, pd.DataFrame(self.Ae).T])
            
            # AreaUnion - AreaPred + AreaInter = AreaGT
            # FN is true in ground truth: assume negative is false, then it is ture
            self.Recall = self.AreaInter / (self.AreaUnion - self.AreaPred + self.AreaInter + Eps) # TP / (TP + FN)
            self.Precision = self.AreaInter / (self.AreaPred + Eps) # TP / (TP + FP)
            self.F2Score = 5 *(self.Precision * self.Recall) / (4 * self.Precision + self.Recall + Eps)
            self.RecallDF = pd.concat([self.RecallDF, pd.DataFrame(self.Recall).T])
            self.PrecisionDF = pd.concat([self.PrecisionDF, pd.DataFrame(self.Precision).T])
            self.F2ScoreDF = pd.concat([self.F2ScoreDF, pd.DataFrame(self.F2Score).T])
            
            # only macro version
            self.mEM = self.Em[0] / self.NumTotal * self.Scale
            # no micro/macro difference
            self.mAE = self.Ae[0] / (self.Epoch + 1)
            # micro/macro version
            self.AvgRecall = manageMetricMM(self.Recall, self.UnionCout, self.opt.metric_type, self.Scale)
            self.AvgPrecision = manageMetricMM(self.Precision, self.UnionCout, self.opt.metric_type, self.Scale)

            self.AvgF2Score = manageMetricMM(self.F2Score, self.UnionCout, self.opt.metric_type, self.Scale)
            
            self.mEMList.append(self.mEM)
            self.mAEList.append(self.mAE)
            self.AvgRecallList.append(self.AvgRecall)
            self.AvgPrecisionList.append(self.AvgPrecision)
            self.AvgF2ScoreList.append(self.AvgF2Score)