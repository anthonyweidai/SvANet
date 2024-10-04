## Import module
 # path manager
import os
from pathlib import Path
 # data processing
import csv
import pandas as pd
 # torch module
import torch
from torch.utils.data import DataLoader
 # my module
from .statistics import Statistics
from ..loss_fn import getLossFn
from ..data import  getMyDataset, buildCollateFn
from ..utils import (
    WeightMetric, getImgPath, getSetPath, 
    workerManager, getLoaderBatch, saveModel, 
    listPrinter, createExcelDict, averageBestMetrics, writeCsv,
)


class BaseTrainer(object):
    def __init__(self, opt, DataParallel) -> None:
        opt = self.paramsInit(opt)
        self.filePathInit(opt)
        self.TrainState = Statistics(opt, State="train")
        self.ValState = Statistics(opt, State="val")
        
        self.opt = opt
        
        self.DataParallel = DataParallel
        
        LossFn = getLossFn(opt)
        self.LossFn = LossFn.to(opt.device)
      
    def paramsInit(self, opt):
        self.Split = None
        self.Rnd = 0 # for best model saver
        
        self.TrainDL = None
        self.ValDL = None
        
        self.LrList = None
        self.CurrentLrRate = None
        
        self.PinMemory, self.NumWorkers = workerManager(opt)
        
        self.TrainSet, self.TestSet = getImgPath(opt.dataset_path, opt.num_split, Mode=opt.get_path_mode)
        
        # drop last management, for batch normalisation
        NumOriTrainImgs = self.TrainSet.__len__()
        if NumOriTrainImgs % opt.batch_size == 1:
            opt.drop_last = True
        
        return opt

    def splitInit(self, opt):
        self.BestEpoch = 0
        self.BestLoss = 9e5
        self.BestComMetric = 0
        self.LastMinLossPath = ""
        self.LastMetricPath = ""
        self.BestStopIndicator = self.BestLoss if opt.saved_metric == "loss" \
            else self.BestComMetric # Best metric indicator
    
    def filePathInit(self, opt):
        if opt.dest_path is not None:
            self.ModelSavePath =  opt.dest_path + "/model"  # Save weight
            self.TrainRecordPath = opt.dest_path + "/metrics/record"  # Save indicators during training
            self.TrainMetricsPath =  opt.dest_path + "/best_metrics.csv"  # Save metrics
            self.TrainRecordPathSingle = opt.dest_path + "/metrics/"
            Path(self.TrainRecordPathSingle).mkdir(parents=True, exist_ok=True)
    
    def dataLoaderSetting(self, opt):
        TrainSet, TestSet = getSetPath(self.TrainSet, self.TestSet, self.Split)
        TargetTrainSet, TargetTestSet = None, None
        
        opt.split = self.Split # for mean std init         
        self.TestImgs = None
        self.TrainImgs = getMyDataset(opt, TrainSet, TargetTrainSet, IsTraining=True)
        self.TestImgs = getMyDataset(opt, TestSet, TargetTestSet, IsTraining=False)
        
        CollateFn = buildCollateFn(opt)
        LoaderBatch = getLoaderBatch(opt)
        
        # You don’t need to shuffle the validation and test datasets, since no training is done, 
        # the model is used in model.eval() and thus the order of samples won’t change the results
        # https://discuss.pytorch.org/t/shuffle-true-or-shuffle-false-for-val-and-test-dataloaders/143720/5
        self.TrainDL = DataLoader(self.TrainImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=opt.loader_shuffle, 
                                    pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=opt.drop_last)
        self.ValDL = DataLoader(self.TestImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=False, 
                                pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=opt.drop_last)
        
        self.opt.max_train_iters = len(self.TrainDL.dataset) // LoaderBatch + (0 if opt.drop_last else 1)
        self.opt.max_train_iters *= opt.epochs
            
    @torch.no_grad()
    def modelSaver(self, opt, Epoch: int, Model):
        if self.ValDL:
            Loss = self.ValState.Loss
            ComMetric = self.ValState.ComMetric
        else:
            Loss = self.TrainState.Loss
            ComMetric = self.TrainState.ComMetric
        
        SavedMode = False
        SaveModelStr = ""
        SaveModelFlag = 0
        if Loss < self.BestLoss:
            SaveModelFlag = 2
            self.BestLoss = Loss
            SaveModelStr += "_minloss"
            if "loss" in opt.saved_metric:
                SavedMode = True
                self.BestStopIndicator = self.BestLoss
                self.BestEpoch = Epoch
        if ComMetric > self.BestComMetric:
            SaveModelFlag += 3
            self.BestComMetric = ComMetric
            Temp = WeightMetric.get(opt.task, "")
            if Temp:
                SaveModelStr += ("_" + Temp)
            if "loss" not in opt.saved_metric:
                SavedMode = True
                self.BestStopIndicator = self.BestComMetric
                self.BestEpoch = Epoch
        
        if "None" not in opt.save_point:
            # segmentation and detection models are super large
            # save only one model for segmentation and detection tasks
            if Epoch in opt.save_point:
                SaveModelFlag = 1
        
        if not SavedMode:
            # storage-friendly for tasks
            if SaveModelFlag == 1 or (opt.task in "classification" and 
                                      SaveModelFlag > 0 and Epoch >= opt.milestones):
                SavedMode = True
            
        if SavedMode:
            if "segmentation" in opt.task:
                ModelName = (
                    opt.seg_head_name if "encoder_decoder" in opt.seg_model_name else opt.seg_model_name
                )
            else:
                ModelName = opt.model_name
            
            SaveModelStr = "%s_epoch%d%s%s_f%s.pth" \
                % (ModelName, Epoch, SaveModelStr, 
                    ("_valrpt%d" % (self.SaveSign)) if "common" in opt.sup_method else "", SaveModelFlag)
            SavePath = "%s/%s" % (self.ModelSavePath, SaveModelStr)
            saveModel(SavePath, Model)
            
            if SaveModelFlag == 2:
                if os.path.exists(self.LastMinLossPath):
                    if "_f2" in self.LastMinLossPath or \
                        ("loss" in opt.saved_metric and "_f5" in self.LastMinLossPath):
                        os.remove(self.LastMinLossPath)
                        # print("The file has been deleted successfully")
                self.LastMinLossPath = SavePath
            elif SaveModelFlag == 3:
                if os.path.exists(self.LastMetricPath):
                    if "_f3" in self.LastMetricPath or \
                        ("loss" not in opt.saved_metric and "_f5" in self.LastMetricPath):
                        os.remove(self.LastMetricPath)
                        # print("The file has been deleted successfully")
                self.LastMetricPath = SavePath
            elif SaveModelFlag == 5:
                if os.path.exists(self.LastMinLossPath):
                    os.remove(self.LastMinLossPath)
                if os.path.exists(self.LastMetricPath):
                    os.remove(self.LastMetricPath)
                self.LastMinLossPath = SavePath
                self.LastMetricPath = SavePath

    @torch.no_grad()
    def bestManager(self, opt, Epoch: int, Model):
        Epoch = Epoch + 1 # for number convenience
        self.SaveSign = max(self.Split, self.Rnd) + 1
        self.modelSaver(opt, Epoch, Model)
        
        if self.ValDL:
            print("Epoch: [%d/%d] \tCrossValid: [%d/%d], \tLearning rate: %.6f" \
                % (Epoch, opt.epochs, self.Split + 1, opt.num_split, self.CurrentLrRate))
            
            ListName = [
                "Best", "BEpoch", "mIoU", "ValmIoU", "mDice", "ValmDice", "Loss", "ValLoss",
                "ValmEM", "ValmAE", "AvgRecall", "AvgPrecis", "AvgF2Score", # "mAE"
            ] 
            ListValue = [
                self.BestStopIndicator, self.BestEpoch, self.TrainState.mIoU, self.ValState.mIoU, 
                self.TrainState.mDice, self.ValState.mDice, self.TrainState.Loss, self.ValState.Loss,
                self.ValState.mEM, self.ValState.mAE, # self.TrainState.mAE, 
                self.ValState.AvgRecall, self.ValState.AvgPrecision, self.ValState.AvgF2Score, 
            ]
            listPrinter(ListName, ListValue, Mode=2, LineNum=6)
        else:
            print("Epoch: [%d/%d], \tLearning rate: %.6f" % (Epoch, opt.epochs, self.CurrentLrRate))
            
            ListName = ["Best", "BEpoch", "Loss"]
            ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.Loss]
            
            if "smooth" not in opt.sup_method:
                ListName.extend(["mIoU", "mDice"])
                ListValue.extend([self.TrainState.mIoU, self.TrainState.mDice])

            listPrinter(ListName, ListValue, Mode=2)
        
           
    def writeRunningMetrics(self):
        # Write avrage training metrics record
        OutputFieldNames = ["LrRate", "BatchSize", "ResizeShape", "TrainLoss"]
        OutValues = [self.LrList, self.opt.batch_size, self.opt.resize_shape, self.TrainState.LossList]

        OutputExcel = createExcelDict(OutputFieldNames, OutValues)
        
        Output = pd.DataFrame(OutputExcel)
        Output.to_csv(self.TrainRecordPath + "_valrpt{}.csv".format(self.SaveSign), 
                      columns=OutputFieldNames, encoding="utf-8")
    
    def writeMetricsRecord(self):
        # Write training record for each class
        if self.ValDL:
            self.ValState.IoUDF.to_csv(self.TrainRecordPathSingle + "iou_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            self.ValState.DiceDF.to_csv(self.TrainRecordPathSingle + "dice_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            
            self.ValState.EmDF.to_csv(self.TrainRecordPathSingle + "em_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            self.ValState.AeDF.to_csv(self.TrainRecordPathSingle + "ae_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            self.ValState.RecallDF.to_csv(self.TrainRecordPathSingle + "recall_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            self.ValState.PrecisionDF.to_csv(self.TrainRecordPathSingle + "precision_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")
            self.ValState.F2ScoreDF.to_csv(self.TrainRecordPathSingle + "f2score_valrpt{}.csv".format(self.SaveSign), 
                                index=False, encoding="utf-8")

    def writeBestMetrics(self):
        # Export and write the best result
        if self.ValDL:
            if self.opt.saved_metric == "loss":
                Idx = self.ValState.LossList.index(min(self.ValState.LossList))
            elif self.opt.saved_metric == "iou":
                Idx = self.ValState.mIoUList.index(max(self.ValState.mIoUList))
            elif self.opt.saved_metric == "ap":
                Idx = self.ValState.mAPList.index(max(self.ValState.mAPList))
            else: # accuracy
                Idx = self.ValState.Top1AccuracyList.index(max(self.ValState.Top1AccuracyList))
            
            self.MetricsFieldNames = ["K-Fold", "BestEpoch"]
            DfBest = ["valrpt_{}".format(self.SaveSign), self.BestEpoch]
            
            self.MetricsFieldNames.extend([
                    "BestmIoU", "BestmDice", "BestmEM", "BestmAE", 
                    "BestAvgRecall", "BestAvgPrecis", "BestAvgF2Score",
                    ])
            DfBest.extend([
                self.ValState.mIoUList[Idx], self.ValState.mDiceList[Idx], 
                self.ValState.mEMList[Idx], self.ValState.mAEList[Idx],
                self.ValState.AvgRecallList[Idx], self.ValState.AvgPrecisionList[Idx], 
                self.ValState.AvgF2ScoreList[Idx],
            ])
            writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, DfBest)
        
    def writeAvgBestMetrics(self):
        # Compute and write mean values of metrics in k-fold validation
        MetricsReader = csv.reader(open(self.TrainMetricsPath, "r"))
        BestMetrics = []
        for Row in MetricsReader:
            BestMetrics.append(Row)
        BestMetrics.pop(0) # remove the header/title in the first row
        
        self.AvgBestMetric = averageBestMetrics(BestMetrics)
        writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, self.AvgBestMetric)