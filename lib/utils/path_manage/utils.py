import os
import re
import shutil
from glob import glob
from pathlib import Path

import numpy as np
from typing import Dict
from copy import deepcopy

from ..system.winCols import WinCols
from ..weight.utils import getBestValprt
from ..variables.defaultVar import WeightMetric


def keepBestValRpt(FolderPath, SaveModels, ValSplit: int=None):
    # keep the best th validation/repetition model name
    KeepValRpt = None
    if ValSplit is not None:
        KeepValRpt = "valrpt%d" % ValSplit
    else:
        CurrentExpFolder = str(Path(FolderPath).parents[0])
        KeepValRpt = getBestValprt(CurrentExpFolder)
    KeepModels = [m for m in SaveModels if KeepValRpt in m] \
        if KeepValRpt is not None else SaveModels
    return KeepModels

    
def getBestModelPath(FolderPath, ValSplit=None):
    if not os.path.isdir(FolderPath):
        return ""
    else:
        SaveModels = os.listdir(FolderPath)
        SaveModels = keepBestValRpt(FolderPath, SaveModels, ValSplit)
        if len(SaveModels) < 1:
            return ""
        Idx = [re.findall(r"%s(\d+)" % "epoch", SaveModels[i])[0] for i in range(len(SaveModels))]
        Idx = list(map(int, Idx))
        SortIdx = np.argsort(Idx) # needs sorting, or best index may be larger than others
        SaveModels = np.asarray(SaveModels)[SortIdx]
        
        IndexTemp = SortIdx[-1]
        SaveModel = SaveModels[-1]
        while not any([m in SaveModel for m in list(WeightMetric.values())]):
            if IndexTemp > 0:
                IndexTemp -= 1
                SaveModel = SaveModels[IndexTemp]
            else:
                # if weight does not has best metric, check loss
                IndexTemp = SortIdx[-1]
                SaveModel = SaveModels[SortIdx[-1]]
                while "minloss" not in SaveModel:
                    if IndexTemp > 0:
                        IndexTemp -= 1
                        SaveModel = SaveModels[IndexTemp]
                    else:
                        SaveModel = SaveModels[SortIdx[-1]]
                        break
                break
        
        return FolderPath + SaveModel


def getExpPath(opt, RootPath):
    # get exp folder path and folder name list
    ## state printing
    ExpPath = "%s/%s/%s" % (RootPath, opt.task, opt.setname.lower())
    if not os.path.isdir(ExpPath):
        print("Experiment %s on dataset %s does not exist" % (ExpPath, opt.setname.lower()))
        ExpPath = "%s/%s" % (RootPath, opt.setname.lower())
        if not os.path.isdir(ExpPath):
            print("Experiment %s on dataset %s does not exist" % (ExpPath, opt.setname.lower()))
            if opt.setname.lower() in RootPath:
                ExpPath = RootPath
            else:
                return None, None
            
    print(("Validating in %s dataset" % opt.setname.lower()).center(WinCols, "^"))
    
    ## return folder only list, sort exp in order
    Exps = next(os.walk(ExpPath))[1]
    Exps = [e for e in Exps if "exp" in e] # keep only exp folder
    ExpNums = [int(re.findall(r"\d+", e)[0]) for e in Exps]
    Exps = [Exps[i] for i in np.argsort(ExpNums)] # string maybe not in order
    return ExpPath, Exps


def getFilePathsFromSubFolders(WalkPath):
    return [os.path.join(Root, File) \
        for Root, Dirs, Files in os.walk(WalkPath) for File in Files]


def getSubdirectories(Dir):
    return [SubDir for SubDir in os.listdir(Dir)
            if os.path.isdir(os.path.join(Dir, SubDir))]


def expFolderCreator(BaseFolder, TaskType, ExpLevel="", TargetExp=None, Mode=0):
    # Count the number of exsited experiments
    FolderPath = "./%s/%s/%s" % (BaseFolder, TaskType, ExpLevel) 
    Path(FolderPath).mkdir(parents=True, exist_ok=True)
    
    ExpList = getSubdirectories(FolderPath)
    if TargetExp:
        ExpCount = TargetExp
    else:
        if len(ExpList) == 0:
            ExpCount = 1
        else:
            MaxNum = 0
            for idx in range(len(ExpList)):
                NumStr = re.findall("\d+", ExpList[idx])
                if NumStr: # should not be empty
                    temp = int(NumStr[0]) + 1
                    if MaxNum < temp:
                        MaxNum = temp
            ExpCount = MaxNum if Mode == 0 else MaxNum - 1
    
    DestPath = "%s/exp%s/" % (FolderPath, str(ExpCount))
    Path(DestPath).mkdir(parents=True, exist_ok=True)
    Path(DestPath + "/model").mkdir(parents=True, exist_ok=True)
    
    return DestPath, ExpCount


def getOnlyFileNames(Path):
    # get only file names
    # # equivalent code:
    # [f for f in os.listdir(Path) if os.path.isfile(f)]
    return next(os.walk(Path))[2] if os.path.isdir(Path) else []


def getOnlyFolderNames(Path):
    # get only folder names
    # # equivalent code:
    # FolderNames = [Name for Name in os.listdir(Path) \
    #     if os.path.isdir(os.path.join(Path, Name))]
    return next(os.walk(Path))[1] if os.path.isdir(Path) else []


def getOnlyFileDirs(Path):
    FileNames = getOnlyFileNames(Path)
    return [os.path.join(Path, n) for n in FileNames]


def getOnlyFolderDirs(Path):
    FolderNames = getOnlyFolderNames(Path)
    return [os.path.join(Path, n) for n in FolderNames]


def moveFiles(Files, DestPath):
    # move list of files to a destination
    if isinstance(Files, str):
        Files = [Files]
    
    for f in Files:
        if os.path.isfile(f):
            # overwrite if file is existed
            shutil.copy(f, DestPath)
    return


def adaptValTest(DatasetPath, ValidStr=["test", "val"]):
    for Name in ValidStr:
        SetPath = "%s/%s/" % (DatasetPath, Name)
        if os.path.isdir(SetPath):
            break
    return Name


def getImgPath(DatasetPath, NumSplit, Mode=1, Shuffle=True):
    # Put images into train set or test set
    from sklearn.model_selection import KFold
    
    TrainSet, TestSet = [], [] # init
    if Mode == 1:
        """ Cross-validation split without class folder
        root/split1/dog_1.png
        root/split1/dog_2.png
        root/split2/cat_1.png
        root/split2/cat_2.png
        """
        for i in range(1, NumSplit + 1):
            TestSet.append(glob("%s/split%d/*" % (DatasetPath, i)))
            
            TrainImgs = []
            for j in range(1, NumSplit + 1):
                if j != i:
                    TrainImgs.extend(glob("%s/split%d/*" % (DatasetPath, j)))
            TrainSet.append(TrainImgs)
        
    elif Mode == 2:
        """ No split with only class folder
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
        """
        TrainSet, TestSet = [[] for _ in range(NumSplit)], [[] for _ in range(NumSplit)]
        ClassNames = os.listdir(DatasetPath)
        Kf = KFold(n_splits=NumSplit, shuffle=Shuffle)
        
        for ClassName in ClassNames:
            ImagePath = glob("%s/%s/*" % (DatasetPath, ClassName))
            IndexList = range(0, len(ImagePath))

            Kf.get_n_splits(IndexList)
            
            for idx, (TrainIndexes, TestIdexes) in enumerate(Kf.split(IndexList)):
                [TrainSet[idx].append(ImagePath[i]) for i in TrainIndexes]
                [TestSet[idx].append(ImagePath[j]) for j in TestIdexes]
                
    elif Mode == 3:
        """ Train and val/test split with class folder
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/val/dog/xxx.png
        root/val/dog/xxy.png
        root/val/dog/[...]/xxz.png
        or,
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/test/dog/xxx.png
        root/test/dog/xxy.png
        root/test/dog/[...]/xxz.png
        """
        TrainSet = glob("%s/train/*/*" % DatasetPath)
        
        Level = adaptValTest(DatasetPath, ValidStr=["test", "val"])
        TestSet = glob("%s/%s/*/*" % (DatasetPath, Level))
        
    elif Mode == 4:
        """  Train and val/test split without class folder
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/val/xxx.png
        root/val/xxy.png
        root/val/[...]/xxz.png
        or,
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/test/xxx.png
        root/test/xxy.png
        root/test/[...]/xxz.png
        """
        TrainSet = glob("%s/train/*" % DatasetPath)
        
        Level = adaptValTest(DatasetPath, ValidStr=["test", "val"])
        TestSet = glob("%s/%s/*" % (DatasetPath, Level))
        
    return TrainSet, TestSet


def getSetPath(TrainSet, TestSet, Split):
    # for source and target domain set path init
    SamplePath = TrainSet[0] if TrainSet else TestSet[0]
    if isinstance(SamplePath, str):
        TrainSet = TrainSet
        TestSet = TestSet
    else:
        TrainSet = TrainSet[Split]
        TestSet = TestSet[Split]
    
    return TrainSet, TestSet


def replacedWithMask(ImgPaths):
    import functools
    
    ReplaceStrs = {
        ".jpg": ".png", 
        **dict.fromkeys(
            ["/train", "/val", "/test", "\\train", "\\val", "\\test"], "/mask"
        ),
    }
    MaskPaths = [
        functools.reduce(lambda a, kv: a.replace(*kv), ReplaceStrs.items(), p)
        for p in ImgPaths
    ]
    return MaskPaths