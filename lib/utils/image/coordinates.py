from typing import List

import cv2
import numpy as np

from ..mathematics.utils import normaliseVetor


def correctCoordinates(Origin, Points):
    ''' move the coordinates axis reference to origin ''' 
    # findcontours return the contour values [y, x] 
    # in image coordinates system
    if not isinstance(Points[0], (list, np.ndarray)):
        Points = [Points]
    
    for i, p in enumerate(Points):
        Points[i][0] = p[0] - Origin[0]
        Points[i][1] = p[1] - Origin[1]
    return Points


def getPixelCoordinates(Img):
    # get the location of the point with pixel for binary images
    if not isinstance(Img, np.ndarray):
        Img = np.asarray(Img)
    return np.asarray(list(zip(*np.where(Img > 0))))


def getPointGroupBoundary(Points):
    # get the boundary of a point group
    Points = np.asarray(Points)
    x1, y1 = Points.min(axis=0)
    x2, y2 = Points.max(axis=0)
    return np.asarray([x1, y1, x2, y2])


def resizePointGroup(Ratio, Points):
    # may be inhomogeneous
    return [list(np.asarray(p) / Ratio) for p in Points] 


def correctBoundary(BoundingBox, Points):
    if not isinstance(Points[0], list):
        Points = [Points]
        
    for i, p in enumerate(Points):
        if p[0] < BoundingBox[0]:
            Points[i][0] = BoundingBox[0]
        elif p[0] > BoundingBox[2]:
            Points[i][0] = BoundingBox[2]
        
        if p[1] < BoundingBox[1]:
            Points[i][1] = BoundingBox[1]
        elif p[1] > BoundingBox[3]:
            Points[i][1] = BoundingBox[3]
            
    return Points


def interpolate2DPoints(Points, NumPoints=200, k=2, BoundingBox=None):
    from scipy import interpolate
    
    NpPoints = np.asarray(Points)
    Tck, U = interpolate.splprep([NpPoints[:, 0], NpPoints[:, 1]], k=k, s=0)
    
    if len(U) >= NumPoints:
        return Points
    else:
        UNew = np.arange(0, 1.001, 1. / NumPoints)
        OutPoints = interpolate.splev(UNew, Tck)
        OutPoints = np.vstack((OutPoints[0], OutPoints[1])).T.tolist()
        if BoundingBox is not None:
            return correctBoundary(BoundingBox, OutPoints)
        else:
            return OutPoints


def supplementPoints(x, NumPoints):
    # supplement point on the inner or outter left and right
    import warnings
    from sklearn.cluster import KMeans
    
    # remove unwanted kmeans warning
    warnings.filterwarnings("ignore")
    
    NumPointsLeft = round(NumPoints / 2)
    NumPointsRight = NumPoints - NumPointsLeft
    
    KeepIdx = np.where(x == True)[0]
    if len(KeepIdx) <= 1:
        return x
    else:
        if KeepIdx[0] == 0:
            kMeanValues = KMeans(n_clusters=2)
            kMeanValues.fit(KeepIdx.reshape(-1, 1))
            YKMeans = kMeanValues.predict(KeepIdx.reshape(-1, 1))
            
            IdxLeft = KeepIdx[np.where(YKMeans == YKMeans[-1])[0][0]]
            IdxRight = KeepIdx[np.where(YKMeans == YKMeans[0])[0][-1]]
        else:
            IdxLeft = KeepIdx[0]
            IdxRight = KeepIdx[-1]
        
        # left
        IdxKeepLeft = IdxLeft - NumPointsLeft
        if IdxKeepLeft < 0:
            x[0:IdxLeft] = True
            x[IdxKeepLeft::] = True
        else:
            x[IdxKeepLeft:IdxLeft] = True
        
        # right
        IdxKeepRight = IdxRight + NumPointsRight + 1
        if IdxKeepRight > len(x):
            x[IdxRight + 1::] = True
            
            LEndIdx = IdxKeepRight - len(x)
            x[0:LEndIdx] = True
        else:
            x[IdxRight + 1:IdxKeepRight] = True
        
        return x


def getPointBreaks(Data):
    # get the break of 1D points
    if isinstance(Data[0], bool):
        # adapted from https://stackoverflow.com/a/69213725/15329637
        from itertools import groupby, accumulate
        
        Indices = list(accumulate(len(list(g)) for i, g in groupby(Data)))
        Starts = Indices[:len(Indices) // 2 * 2:2]
        Stops = [i - 1 for i in Indices[1::2]]
    else:
        # array
        NeighbourGap = np.zeros((len(Data) - 1, 1))
        for i in range(len(NeighbourGap)):
            NeighbourGap[i] = Data[i + 1] - Data[i]
        Breaks = np.where(NeighbourGap != 1)[0]

        # breaks considering boundary condition
        Starts = np.hstack((Data[0], Data[Breaks + 1]))
        Stops = np.hstack((Data[Breaks], Data[-1]))
        
    return Starts, Stops
    

def continuePoints(IDMap, NumTotal):
    # continue 1D points (give and drop) with only one point in each break
    # cycling [0, end]
    IDMap = np.sort(IDMap)
    Starts, Stops = getPointBreaks(IDMap)
    
    SupIDs = []
    for i in range(len(Starts) - 1):
        s1 = Starts[i]
        s2 = Starts[i + 1]
        
        if s2 - s1 == 2:
            SupIDs.append(s1 + 1)
        elif s1 == 1 and Stops[-1] == NumTotal - 1:
            SupIDs.append(0)
        elif s1 == 0 and Stops[-1] == NumTotal - 2:
            SupIDs.append(NumTotal - 1)

    if SupIDs:
        IDMap = np.concatenate((IDMap, SupIDs), axis=0)
        Starts, Stops = getPointBreaks(np.sort(IDMap))
    
    # remove not continous
    if len(Starts) > 1:
        if Starts[0] > 0 or len(Starts) > 2:
            Gaps = np.asarray(Stops) - np.asarray(Starts)
            Flag = 0
            if Starts[0] == 0 and Stops[-1] == NumTotal - 1:
                Gaps, EndGap = Gaps[:-1], Gaps[-1]
                Gaps[0] += EndGap
                Flag = 1
                
            MaxId = np.where(Gaps == max(Gaps))[0][0] # should have only one longest
            if MaxId == 0 and Flag:
                IDMap1 = np.arange(0, Stops[0] + 1)
                IDMap2 = np.arange(Starts[-1], NumTotal)
                IDMap = np.concatenate((IDMap1, IDMap2), axis=0) 
                
            else:
                IDMap = np.arange(Starts[MaxId], Stops[MaxId] + 1)

    return IDMap


def xywh2xyxy(Box, ImgWidth=None, ImgHeight=None) -> List:
    # Box = self.scaleBoundingBox(ImgIdx, Box)
    x1, y1, w, h = Box
    return [
        max(0, x1),
        max(0, y1),
        min(x1 + w, ImgWidth if ImgWidth else x1 + w), # min(x1 + w, self.ImgWidth),
        min(y1 + h, ImgHeight if ImgHeight else y1 + h), # min(y1 + h, self.ImgHeight),
    ]


def swapArrayImageCoor2D(Points):
    # trasnform array coordinates system to image coordinates system
    if not isinstance(Points, np.ndarray):
        Points = np.asarray(Points)
    Points[:, [0, 1]] = Points[:, [1, 0]]
    return Points


def swapCartesianImageCoor2D(Points):
    # trasnform cartesian coordinates system to image coordinates system
    if not isinstance(Points, np.ndarray):
        Points = np.asarray(Points)
    Points[:, 1] = - Points[:, 1]
    return Points


def pointOnLine(LP1, LP2, Point):
    # adapted from https://stackoverflow.com/a/61342198/15329637
    Pto1 = Point - LP1
    LineLen = LP2 - LP1
    t = np.dot(Pto1, LineLen) / np.dot(LineLen, LineLen)
    # # if you need the the closest point belonging to the segment
    # t = max(0, min(1, t))
    return LP1 + t * LineLen
