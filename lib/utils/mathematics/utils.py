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