import numpy as np
from typing import Optional


def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.Py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def removeOutliers(Data, OutlierConstant=1.5):
    if isinstance(Data[0], np.bool_):
        # remove outliers in bool array
        NewArr = np.arange(len(Data))
        NewArr = NewArr[Data]
        NewArr, KeepIndex = removeOutliers(NewArr)
        
        NewBool = np.asarray(Data)
        for i in range(len(Data)):
            if i not in NewArr:
                NewBool[i] = False
        return NewBool, KeepIndex
    else:
        # remove outliers in numpy array
        Temp = np.asarray(Data)
        Upper = np.percentile(Temp, 75)
        Lower = np.percentile(Temp, 25)
        IQR = (Upper - Lower) * OutlierConstant
        QuartileSet = (Lower - IQR, Upper + IQR)
        
        KeepIndex = np.where((Temp >= QuartileSet[0]) & (Temp <= QuartileSet[1]))
        return Temp[KeepIndex].tolist(), KeepIndex[0]


def removeOutliersV2(Data, m=2.):
    if isinstance(Data[0], np.bool_):
        # remove outliers in bool array
        NewArr = np.arange(len(Data))
        NewArr = NewArr[Data]
        NewArr, KeepIndex = removeOutliersV2(NewArr)
        
        NewBool = np.asarray(Data)
        for i in range(len(Data)):
            if i not in NewArr:
                NewBool[i] = False
        return NewBool, KeepIndex
    else:
        Data = np.asarray(Data)
        Dist = np.abs(Data - np.median(Data))
        MeanDev = np.median(Dist)
        s = Dist / (MeanDev if MeanDev else 1.)
        
        KeepIndex = np.where(s < m)
        return Data[KeepIndex].tolist(), KeepIndex


def normaliseVetor(Vector):
    ''' adapted from https://stackoverflow.com/a/40360416/15329637 '''
    if not isinstance(Vector, np.ndarray):
        Vector = np.asarray(Vector)
    
    NormV = np.linalg.norm(Vector)
    if NormV == 0:
        NormV = np.finfo(Vector.dtype).eps
    return Vector / NormV


def keepIndex2Bool(KeepIndex, Len):
    ''' Invert keepindex to remove index
    # _, KeepIndex = removeOutliersV2(RemainedSumDist)
    # BoolIndex = keepIndex2Bool(KeepIndex, len(RemainedSumDist))W
    # BoolIndex = np.invert(BoolIndex) # get outlier index
    '''
    if isinstance(KeepIndex, tuple):
        # for numpy where format
        KeepIndex = KeepIndex[0]
        
    BoolIndex = np.full((Len,), False)
    for i in KeepIndex:
        BoolIndex[i] = True
    return BoolIndex


def averageBestMetrics(BestMetrics, AvgSupMode=True):
    if isinstance(BestMetrics, list):
        Flag1 = 1
        RowNum = len(BestMetrics)
        ColNum = len(BestMetrics[0])
    else:
        # pd.series
        RowNum, ColNum = BestMetrics.shape
        Flag1 = 1 if 'BestEpoch' in BestMetrics.columns.values.tolist() else 0
    
    Values = np.zeros((ColNum - 1, 1))
    
    Flag2 = 0
    BestMetrics = np.asarray(BestMetrics)
    for i in range(RowNum):
        if 'Average' in BestMetrics[i][0]:
            Flag2 = 1
            continue
        for j in range(ColNum - 1):
            Values[j] += float(BestMetrics[i][j + 1])
    if Flag2 == 1:
        RowNum -= 1
    Values /= RowNum
    if Flag1:
        # round epoch to be integer, 0 decimal
        Values[0] = int(np.rint(Values[0])) 
    
    AvgBestMetric = ['Average'] if Flag2 == 0 or not AvgSupMode else ['Average_Sup']
    AvgBestMetric.extend(list(Values.flatten()))
    return AvgBestMetric


def inhomogeneousArithmetic(Data, Type='mean'):
    # np.mean(Data, axis=1), np.max(Data, axis=1), np.min(Data, axis=1)
    # not working for inhomogeneous part
    if 'mean' in Type:
        return np.asarray([np.mean(d) if d.any() else 0 for d in Data])
    elif 'max' in Type:
        return np.asarray([np.max(d) if d.any() else 0 for d in Data])
    elif 'min' in Type:
        return np.asarray([np.min(d) if d.any() else 0 for d in Data])
    else:
        NotImplementedError


def bincount2DVectorized(Array):
    # binary count of generic n-dims
    # https://stackoverflow.com/a/46256361/15329637 
    N = Array.max() + 1
    ArrayOffset = Array + np.arange(Array.shape[0])[:, None] * N
    return np.bincount(ArrayOffset.ravel(), minlength=Array.shape[0] * N).reshape(-1, N)
