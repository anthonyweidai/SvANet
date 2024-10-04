import os


# default, supplement, auto-dl, kaggle
DEFAULT_DATA_ROOT = ['./dataset/', '../dataset/', '../autodl-tmp/', '../input/']


def correctDatasetPath(DataPath, SetName: str=None):
    Count = 0
    while not os.path.isdir(DataPath):
        DataPath = DEFAULT_DATA_ROOT[Count]
        Count += 1
        
    if SetName is not None:
        if '../input/' in DataPath:
            FolderPath = DataPath + SetName.lower() + '/' + SetName # for kaggle
        else:
            FolderPath = DataPath + SetName
        return DataPath, FolderPath
    else:    
        return DataPath
