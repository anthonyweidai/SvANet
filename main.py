## Import module
 # path manager
import os
 # data processing
import time
from datetime import datetime
 # torch module
import torch
 # my module
from opts import Opts
from lib.trainer import Trainer
from lib.params_init import paramsInit
from lib.utils import (
    seedSetting, deviceInit, 
    expFolderCreator, writeCsv, optimiseLogVal, listPrinter,
)


class MainTrain(object):
    def __init__(self, opt) -> None:
        seedSetting(RPMode=False)
        
        self.paramsInit(opt)
        self.filePathInit(opt)
        self.logFieldInit(opt)
        
        self.opt = opt
        
    def paramsInit(self, opt):
        # device
        self.DataParallel, self.DeviceStr = deviceInit(opt)
        opt.device = torch.device(self.DeviceStr)
    
    def filePathInit(self, opt):
        if opt.target_supplement == 0:
            self.Sup = False
            self.TargetExp = None
        else:
            self.Sup = True
            self.TargetExp = opt.target_exp
            
        TaskType = opt.task
        if opt.lincls:
            TaskType += "_" + "lincls"
        elif "common" not in opt.sup_method:
            TaskType += "_" + opt.sup_method
        
        # input_param.csv
        DestPath, self.ExpCount = expFolderCreator(opt.exp_base, TaskType, opt.exp_level, self.TargetExp)
        self.InputLogPath = DestPath + "/input_param.csv"
        opt.dest_path = DestPath
        
        self.ExpLogPath = "./%s/%s/%s/log_%s.csv" % (
            opt.exp_base, TaskType, opt.exp_level, opt.sup_method)
        
    def logFieldInit(self, opt):
        LogField = [
            "Task", "Supervision", "Dataset", "NumClasses", 
            "BatchSize", "ResizeShape", "MeanStd",
            "Model", "Optimiser", "Schedular", "WeightDecay", 
            "Loss", "SavedMetric", "MetricType", "CollateFn", 
        ] # Define header
        LogInfo = [
            opt.task, opt.sup_method, opt.setname, opt.num_classes, 
            opt.batch_size, opt.resize_shape, opt.use_meanstd,
            opt.model_name, opt.optim, opt.schedular, opt.weight_decay, 
            opt.loss_name, opt.saved_metric, opt.metric_type, opt.collate_fn_name, 
        ]
        listPrinter(["Device"] + LogField, [self.DeviceStr] + LogInfo)
        
        FirstLogField = ["exp", "date"]
        EndLogField = [
            "LrDecay", "PreTrained", "FreezeWeight", "NumEpochs", 
            "NumberofSplit", "NumRepeat", "Sup", "LrRate", 
        ]
        FirstLogInfo = [self.ExpCount, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        EndLogInfo = [
            opt.lr_decay, opt.pretrained, opt.freeze_weight, opt.epochs,
            opt.num_split, opt.num_repeat, self.Sup, opt.lr, 
        ]
        self.LogField = FirstLogField + LogField + EndLogField
        self.LogInfo = FirstLogInfo + LogInfo + EndLogInfo
        
        if "common" in opt.sup_method:
            WeightName = opt.weight_name
            if WeightName:
                WeightName = os.path.splitext(os.path.basename(WeightName))[0] # get weight name
                # remove model name from weight name
                WeightName = WeightName.replace(opt.model_name.lower() + "_", "")
            self.LogField.extend(["WeightName"])
            self.LogInfo.extend([WeightName])
            
            
        ListName = []
        ListValues = []
        
        ListName.extend(["SegModel", "SegHead", "UseSepConv"])
        ListValues.extend([opt.seg_model_name, opt.seg_head_name, opt.use_sep_conv])
        
        listPrinter(ListName, ListValues)
        
        # feature map guide
        FGListName = [
            "FeatureGuide", "FGUseGuide", "FGStartStage", "FGResizeStage", 
            "FGForHead", "SvAttn", "SvAttnDivisor", "FGBottle", "FGBottleSE", 
            "FGViT", "FGViTSE", "FGNoStage5", "FGLink", 
            "FGLinkViT", "FGViTLinkSE",
        ]
        FGListValues = [
            opt.seg_feature_guide, opt.fg_use_guide, opt.fg_start_stage, opt.fg_resize_stage, 
            opt.fg_for_head, opt.fg_svattn, opt.fg_svattn_divisor, opt.fg_bottle, opt.fg_bottle_se, 
            opt.fg_vit, opt.fg_vit_se, opt.fg_nostage5, opt.fg_link, 
            opt.fg_link_vit, opt.fg_link_vit_se, 
        ]
        FGListValues = optimiseLogVal(FGListValues)
        if opt.seg_feature_guide:
            listPrinter(FGListName, FGListValues)
            
        ListName.extend(FGListName)
        ListValues.extend(FGListValues)
            
        self.LogField.extend(ListName)
        self.LogInfo.extend(ListValues)
        
        writeCsv(self.InputLogPath, self.LogField, self.LogInfo)
       
    def training(self):
        if self.opt.num_split == 1 and self.opt.num_repeat > self.opt.num_split:
            SplitLoop = [0] * self.opt.num_repeat
        else:
            SplitLoop = range(self.opt.num_split)

        for i, Split in enumerate(SplitLoop):
            Round = i + self.opt.target_supplement
            Split += self.opt.target_supplement
            
            self.MyTrainer = Trainer(self.opt, self.DataParallel) # training class
            self.MyTrainer.run(Round, Split)
            
            ## Writing results
            self.MyTrainer.writeRunningMetrics()
            if self.MyTrainer.ValDL:
                self.MyTrainer.writeMetricsRecord()
                self.MyTrainer.writeBestMetrics()
            if Round >= self.opt.num_repeat - 1 or Split >= self.opt.num_repeat - 1:
                # general, split limit, supplement limit
                if self.MyTrainer.ValDL:
                    self.MyTrainer.writeAvgBestMetrics()
                break
        
    def writeLogFile(self, TimeCost):
        # Write input and output param in log file
        ## field
        self.LogInfo[1] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.LogInfo.append(round(TimeCost, 2))
        NewFieldNames = ["TimeCost", "BEpoch"]
        
        ## value
        if self.MyTrainer.ValDL:
            if "segmentation" in opt.task:
                self.LogInfo.extend(self.MyTrainer.AvgBestMetric[1:])
                NewFieldNames.extend([
                    "mIoU", "mDice", "mEM", "mAE", "AvgRecall", "AvgPrecis", "AvgF2Score"
                ])
        else:
            self.LogInfo.append(self.MyTrainer.BestLoss)
            NewFieldNames.append("Train loss")
            if "segmentation" in self.opt.task:
                self.LogInfo.append(self.MyTrainer.BestComMetric)
                NewFieldNames.extend("Train mIoU")
                
        writeCsv(self.ExpLogPath, self.LogField, self.LogInfo, NewFieldNames)


if __name__ == "__main__":
    Tick0 = time.perf_counter() # unit: second
    
    opt = Opts().parse()
    opt = paramsInit(opt)
    MyTrain = MainTrain(opt)

    MyTrain.training()
    
    TimeCost = time.perf_counter() - Tick0
    print("Finish training using: %.2f minutes" % (TimeCost / 60))
    
    MyTrain.writeLogFile(TimeCost)