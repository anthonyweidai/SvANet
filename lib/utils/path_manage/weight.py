import os
from pathlib import PurePath, Path


# default, supplement, auto-dl, kaggle
DEFAULT_WEIGHT_ROOT = ['./savemodel/', '../savemodel/', '../autodl-tmp/savemodel/', '../input/']
# WeightPath = '../input/pad-model/' # for kaggle


def correctWeightPath(WeightPath, BestMode=False):
    # get the last level folder
    FinalLevelFolder = PurePath(WeightPath).name  
    
    # get suffix folder name
    if 'best' not in FinalLevelFolder:
        FlagFolder = FinalLevelFolder
    else:
        # last second level folder
        FlagFolder = Path(WeightPath).parent.name
    SetName = '' if 'savemodel' in FlagFolder else FlagFolder
    SuffixFolder = '%s/best/' % SetName if BestMode else SetName

    # get existed weight path
    Count = 0
    while not os.path.isdir(WeightPath):
        WeightPath = '%s/%s/' % (DEFAULT_WEIGHT_ROOT[Count], SuffixFolder)
        Count += 1
        if Count == len(DEFAULT_WEIGHT_ROOT):
            break
    
    return WeightPath