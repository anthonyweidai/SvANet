import os
import re
import importlib
import numpy as np


def importModule(CurrentPath, PurePathMode: bool=False, SkipFolder: list=[]):
    # Automatically import the modules
    SubFolds = [''] # relative pure directory
    RegisterPath = os.path.dirname(CurrentPath.replace(os.getcwd(), ''))
    RegisterPath = RegisterPath.removeprefix('\\').removeprefix('/')
    if not PurePathMode:
        SubFolds.extend(next(os.walk(RegisterPath))[1]) # return folder only list
        SkipFolder.extend(['__pycache__', 'config'])
        SubFolds = [f for f in SubFolds if f not in SkipFolder]
    
    RegisterPath = RegisterPath.replace('\\', '.').replace('/', '.')
    for f1 in SubFolds:
        ModulesDir = '%s/%s' % (os.path.dirname(CurrentPath), f1)
        RelativePath = '%s.%s' % (RegisterPath, ('%s.' % f1) if len(f1) > 0 else '')
        
        if os.path.isdir(ModulesDir):
            for f2 in os.listdir(ModulesDir): # [f for f in os.listdir(ModulesDir) if f not in SkipFolder]
                JoinPath = os.path.join(ModulesDir, f2)
                if (
                        not f2.startswith("_") 
                        and not f2.startswith(".") 
                        and (f2.endswith(".py") or os.path.isdir(JoinPath))
                ):
                    ModuleName = f2[: f2.find(".py")] if f2.endswith(".py") else f2
                    _ = importlib.import_module(RelativePath + ModuleName)


def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)


def callMethod(self, ElementName):
    return getattr(self, ElementName)


def pair(Val):
    return Val if isinstance(Val, (tuple, list)) else (Val, Val)


def indicesSameEle(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def splitChecker(SetName: str, RSplit=False):
    IsSplit = False
    
    if 'split' in SetName:
        # for test mode
        NumSplit = 2
        IsSplit = True
    else:
        NumCand = re.findall(r'S(\d+)', SetName)
        if NumCand:
            NumSplit = int(NumCand[-1])
            if NumSplit <= 10: # no more than 10 split
                IsSplit = SetName.endswith('S%d' % NumSplit)
    
    if RSplit: # return the number of splits
        if not IsSplit:
            NumSplit = 1
                
        return IsSplit, NumSplit
    else:
        return IsSplit


def groupSort(ListGroup, KeyRul=lambda s: s.split('\\')[-1].split('_')[0]):
    import itertools
    from itertools import groupby
    # group elements in list with same substring 
    # images with same source -> similar features
    
    ListGroup = [list(i) for _, i in groupby(ListGroup, KeyRul)]
    np.random.shuffle(ListGroup)
    return list(itertools.chain(*ListGroup))
