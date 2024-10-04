import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from opts import Opts
from lib.data import getMyDataset
from lib.trainer import Inferencer
from lib.params_init import segOptInit, initSegModelbyCSV
from lib.metrics import manageMetricMM, computeMaskRegion, measureEAM, computeAE
from lib.utils import (
    getImgPath, getSetPath, getExpPath, getOnlyFileDirs,
    getKeepTinyIds, averageUniqueLog, writeCsv,
)


@torch.inference_mode()
def initAndRunTinyValer(opt, FirstKeepIds, FolderPath, Split, ValSplit):
    opt = initSegModelbyCSV(opt, FolderPath, ValSplit)
        
    opt = segOptInit(opt)
    Engine = Inferencer(opt, Split)
    
    # dataset attribute should not be set after DataLoader is initialized
    KeepTestSet = [Engine.TestSet[i] for i in FirstKeepIds]
    TestImgs = getMyDataset(opt, KeepTestSet, IsTraining=False)
    Engine.InferDL = DataLoader(
        TestImgs, Engine.LoaderBatch, num_workers=Engine.NumWorkers, shuffle=False, 
        pin_memory=Engine.PinMemory, collate_fn=Engine.CollateFn, drop_last=opt.drop_last
    )
    
    Target, Prediction = Engine.run()
    
    return Target, Prediction, KeepTestSet
    

@torch.inference_mode()
def computeMetrics(Ratio, Prediction: Tensor, Target: Tensor, KeepIdx, Eps):
    """
    Compute the calculated metrics restricted by in a size ratio.
    It can also be used to inference metrics by giving ratio 100.
    """
    Scale = 100.0
    
    TinyPrediction = Prediction[KeepIdx]
    TinyTarget = Target[KeepIdx]
    
    AreaPred, AreaInter, AreaUnion, UnionCout = computeMaskRegion(TinyPrediction, TinyTarget)
    
    IoU = AreaInter / AreaUnion
    Dice = (2 * AreaInter) / (AreaUnion + AreaInter)
    Precision = AreaInter / (AreaPred + Eps)
    Recall = AreaInter / (AreaUnion - AreaPred + AreaInter + Eps)
    F2Score = 5 *(Precision * Recall) / (4 * Precision + Recall + Eps)
    print("Tiny_%s IoU: " % str(Ratio), IoU) # r can be float number
    # print("Tiny_%s Dice: " % str(Ratio), Dice)
    
    mIoU = manageMetricMM(IoU, UnionCout, Scale=Scale)
    mDice = manageMetricMM(Dice, UnionCout, Scale=Scale)
    AvgPrecision = manageMetricMM(Precision, UnionCout, Scale=Scale)
    AvgRecall = manageMetricMM(Recall, UnionCout, Scale=Scale)
    AvgF2Score = manageMetricMM(F2Score, UnionCout, Scale=Scale)
    
    mEM = measureEAM(TinyPrediction, TinyTarget) / len(KeepIdx) * Scale
    mAE = computeAE(TinyPrediction, TinyTarget) / len(KeepIdx)
    
    return mIoU, mDice, AvgPrecision, AvgRecall, AvgF2Score, mEM, mAE


@torch.inference_mode()
def traversalTinyRatio(
    opt, TinyRatios, AreaRatios, 
    Exp, Prediction, Target, KeepTestSet, 
    OutFilePath, Eps, KeepThred=False,
):
    for r in TinyRatios:
        # keep only tiny object and those with 0 area
        KeepIdx = np.where(AreaRatios < r)[0] if r < 100 or not KeepThred \
            else np.arange(len(AreaRatios))
        if not KeepIdx.any():
            continue
        
        mIoU, mDice, AvgPrecision, AvgRecall, AvgF2Score, mEM, mAE = \
            computeMetrics(r, Prediction, Target, KeepIdx, Eps)
           
        # write into csv
        LogField = [
            "exp", "SegModel", "SegHead", "Weight", "TinyRatio",
            "FeatureGuide", "SvAttn", "FGViT", "FGNoStage5", "FGLink",
            "mIoU", "mDice", "mEM", "mAE", "AvgRecall", "AvgPrecision", "AvgF2Score",
        ]
        LogInfo = [
            re.findall(r"\d+", Exp)[0], opt.seg_model_name, opt.seg_head_name, Path(opt.pretrained_weight).stem, r,
            opt.seg_feature_guide, opt.fg_svattn, opt.fg_vit, opt.fg_nostage5, opt.fg_link,
            mIoU, mDice, mEM, mAE, AvgRecall, AvgPrecision, AvgF2Score,
        ]
        writeCsv(OutFilePath, LogField, LogInfo)
        
        # break loop if reaches maximum length of dataset 
        if len(KeepTestSet) == len(KeepIdx):
            break


@torch.inference_mode()
def computeRatioRangeMetrics(
    opt, TinyRatios, AreaRatios, 
    FirstKeepIds, Exp, ExpPath, Split=0, 
    OutFilePath="tiny_metrics.csv",  Eps=1e-7,
):
    """
    Compute the calculated metrics only in tiny objects (use ratio).
    It can also be used to inference metrics by giving 100 ratio.
    """
    if not FirstKeepIds.any():
        return
    AreaRatios = AreaRatios[FirstKeepIds]
    
    # get all valrpts
    FolderPath = "%s/%s" % (ExpPath, Exp)
    CSVPath = FolderPath + "/best_metrics.csv"
    MetricCsv = pd.read_csv(CSVPath) # read best_metrics.csv
    KFolds = MetricCsv["K-Fold"].tolist()
    ValSplits = KFolds.copy()
    if "Average" in ValSplits:
        ValSplits.remove("Average")
    ValSplits = [int(re.findall(r"valrpt_(\d+)", s)[0]) for s in ValSplits]
    
    # append tiny metric values in metric csv file
    for s in ValSplits:
        Target, Prediction, KeepTestSet = \
            initAndRunTinyValer(opt, FirstKeepIds, FolderPath, Split, s)
        traversalTinyRatio(
            opt, TinyRatios, AreaRatios, 
            Exp, Prediction, Target, KeepTestSet, 
            OutFilePath.replace(".csv", "_valrpt%d.csv" % s), Eps,
        )
        
    return ValSplits


if __name__ == "__main__":
    opt = Opts().parse()
    opt.gpus = "0" # "-1"
        
    SetNames = [
        "FIVES", "ISIC2018T1", "PolypGen", 
        "ATLAS", "KiTS23", "TissueNet",  
        "SpermHealth", 
    ]
    TinyRatios = [0.5, 1, 2, 3, 5, 10, 100]
    
    RootPath = "exp/segmentation"
    for n in SetNames:
        opt.setname = n
        
        # start printing
        ExpPath, Exps = getExpPath(opt, RootPath)
        if ExpPath is None:
            continue
        
        # remove exsited tiny_metrics
        CurrentFiles = getOnlyFileDirs(ExpPath)
        RemoveFiles = [f for f in CurrentFiles if "tiny_metrics" in f]
        for f in RemoveFiles:
            print("Remove %s" % f)
            os.remove(f)
        
        # output file
        OutFilePath = "%s/tiny_metrics.csv" % ExpPath
        if os.path.isfile(OutFilePath):
            os.remove(OutFilePath)
        
        # dataset
        opt = segOptInit(opt) # init to get set path
        TrainSet, TestSet = getImgPath(opt.dataset_path, opt.num_split, Mode=opt.get_path_mode)
        TrainSet, TestSet = getSetPath(TrainSet, TestSet, Split=0) # train set can also be inferenced
        
        # get only val/test mask ratios
        AreaRatios, FirstKeepIds = getKeepTinyIds(TestSet, TinyRatios)
        
        # inferencing
        for i, e in enumerate(Exps):
            ValSplits = computeRatioRangeMetrics(
                opt, TinyRatios, AreaRatios, FirstKeepIds, e, 
                ExpPath=ExpPath, OutFilePath=OutFilePath, 
            )
        
        if len(ValSplits) > 0:
            # average all models' tiny metric into the same csv
            MetadataGroup = []
            for s in ValSplits:
                Metadata = OutFilePath.replace(".csv", "_valrpt%d.csv" % s)
                MetaCsv = pd.read_csv(Metadata)
                MetadataGroup.append(MetaCsv.drop(columns=["Weight"]))
            averageUniqueLog(MetadataGroup, ExpPath, "tiny_metrics_avg")