import cv2
import math
import random
import numpy as np
from scipy.spatial.distance import cdist

from ..utils import pair


# position initailisation
def posInitWithRes(opt, SupRatio, GenMode, ImgSize=None, HWEqual=True):
    if ImgSize:
        W, H = ImgSize
    else:
        W, H = pair(opt.oriresize_shape)
    MinLen = min(W, H)
    
    CropRatio = opt.crop_ratio
    if not GenMode:
        # for pretext mask generation
        MinCropRatio = 0.8
    elif not opt.mincrop_ratio:
        MinCropRatio = 0.6
    else:
        MinCropRatio = opt.mincrop_ratio
    MaxCropRatio = 1.0
    
    if not CropRatio:
        CropRatio = random.uniform(MinCropRatio, MaxCropRatio - SupRatio)
        if HWEqual:
            CropRatioH = CropRatioW = CropRatio
        else:
            CropRatioH = CropRatio
            CropRatioW = random.uniform(MinCropRatio, MaxCropRatio - SupRatio)
    else:
        CropRatioH = CropRatioW = CropRatio
        
    
    def computePosBBox(opt, Len, CropRatio, MinCropRatio, MinLen, Delta=None):
        RandomPos = opt.random_pos
        if CropRatio >= 1:
            CropRatio = 1
            RandomPos = False
        
        CropLen = MinLen * CropRatio
        
        PosMin = (Len - CropLen) / 2
        PosMax = PosMin + CropLen
        SupLen = SupRatio * MinLen / 2 # Min using sublen prevent zero pixel near the boundary of image
        
        # Random rotation centre
        if RandomPos:
            # Delta = random.uniform(- Len / 2 + CropLen / 2 + SupLen, Len / 2 - CropLen / 2 - SupLen)
            if Delta is None:
                if not opt.centre_rand:
                    ## uniformly random
                    Delta = random.uniform(- PosMin, PosMin)
                else:
                    ## random in the centre
                    MaxOffset = max(MinLen * (CropRatio - MinCropRatio) / 2 - SupLen, 0) # - 0.1
                    MaxOffset = min(MaxOffset, PosMin)
                    Delta = random.uniform(- MaxOffset, MaxOffset)
            
            PosMin = PosMin + Delta
            PosMax = PosMax + Delta
        
        return PosMin, PosMax, CropLen, SupLen, Delta
    
    if HWEqual:
        Left, Right, _, _, Delta = computePosBBox(opt, W, CropRatioW, MinCropRatio, MinLen=MinLen)
        Upper, Lower, CropLen, SupLen, _ = computePosBBox(opt, H, CropRatioH, MinCropRatio, MinLen=MinLen, Delta=Delta)
    else:
        Left, Right, CropLenW, SupLenW, _ = computePosBBox(opt, W, CropRatioW, MinCropRatio, MinLen=W)
        Upper, Lower, CropLenH, SupLenH, _ = computePosBBox(opt, H, CropRatioH, MinCropRatio, MinLen=H)
        CropLen = [CropLenH, CropLenW]
        SupLen = [SupLenH, SupLenW]
    
    return Left, Upper, Right, Lower, MinLen, CropLen, SupLen


def pointCentreRotation(Centre, Point, Angle):
    # Rotate a point clockwise by a given Angle around a given origin.
    Angle = math.radians(Angle)
    # [x, y, w, h] format
    Ox, Oy = Centre
    Px, Py = Point
        
    Qx = Ox + math.cos(Angle) * (Px - Ox) - math.sin(Angle) * (Py - Oy)
    Qy = Oy + math.sin(Angle) * (Px - Ox) + math.cos(Angle) * (Py - Oy)
    
    return [round(Qx, 3), round(Qy, 3)]


def boxPointRotation(Box, Angle):
    '''MaskBox = boxPointRotation(MaskBox, RotAngle2)'''
    # [Left, Upper, Right, Lower]
    Cx = (Box[2] - Box[0]) / 2
    Cy = (Box[3] - Box[1]) / 2
    Centre = [Cx, Cy]
    
    UpperLeft = pointCentreRotation(Centre, [Box[0], Box[1]], Angle)
    UpperRight = pointCentreRotation(Centre, [Box[2], Box[1]], Angle)
    LowerLeft = pointCentreRotation(Centre, [Box[0], Box[3]], Angle)
    LowerRight = pointCentreRotation(Centre, [Box[2], Box[3]], Angle)
    
    XList= [UpperLeft[0], UpperRight[0], LowerLeft[0], LowerRight[0]]
    YList= [UpperLeft[1], UpperRight[1], LowerLeft[1], LowerRight[1]]
    XList.sort()
    YList.sort()
    
    return [XList[0], YList[0], XList[-1], YList[-1]]


def getBoxDimension(Box):
    Length, Width = [0] * 2
    
    Distance = cdist(np.expand_dims(Box[0], axis=0), Box[1:])
    DistInOrder = np.unique(Distance)
    if len(DistInOrder) > 1: # box points are identical overlap
        Length = np.unique(Distance)[-2]
        Width = np.min(Distance)
        # print('length: %4f' % Length, 'width: %4f' % Width)
        
    return Length, Width


def correctBoxPointOrder(BoundingBox, OriSize):
    # correct order
    Left = min(BoundingBox[0::2])
    Upper = min(BoundingBox[1::2])
    Right = max(BoundingBox[0::2])
    Lower = max(BoundingBox[1::2])
    
    # correct threshold
    Diff = 1e-7
    Left = 0 if Left <= Diff else round(Left)
    Upper = 0 if Upper <= Diff else round(Upper)
    
    W, H = OriSize
    Right = W if abs(Right - W) <= Diff else round(Right)
    Lower = H if abs(Lower - H) <= Diff else round(Lower)
    
    return [Left, Upper, Right, Lower]


def computeTwoPointsHorizonAngle(Points):
    # compute the anlge between an edge (two points) and horizontal axis
    DeltaY = Points[1][1] - Points[0][1]
    DeltaX = Points[1][0] - Points[0][0]
    return math.degrees(math.atan2(DeltaY, DeltaX))
    

def getBoxAngle(Box):
    # get box angle between long axis and horizontal plane
    Distance = cdist(np.expand_dims(Box[0], axis=0), Box[1:])
    _, IdxMap = np.unique(Distance, return_index=True)
    NumIdx = len(IdxMap)
    if NumIdx == 1:
        return 0
    elif NumIdx == 2:
        Idx = IdxMap[0]
    else:   
        Idx = IdxMap[-2]
    LongEdge = [Box[0], Box[Idx]]
    return computeTwoPointsHorizonAngle(LongEdge)


def getMinimumBoxFromContour(Contour):
    Rect = cv2.minAreaRect(Contour)
    Box = cv2.boxPoints(Rect)
    return np.intp(Box), getBoxAngle(Box) # Rect[-1]


def computeMalDistance(S1, S2):
    # https://stackoverflow.com/a/27691752/15329637
    # two state, S1 includes all situation, S2 only has one
    Delta = S2 - S1
    
    X = np.vstack([S2, S1])
    V = np.cov(X.T)
    CoVariance = np.linalg.inv(V)
    
    return np.sqrt(np.einsum('ij, jk, ik->i', Delta, CoVariance, Delta))


def polynomialFitting(Points, FitNum=100):
    # use non-linear trendline from points 
    XCoor = [p[0] for p in Points]
    YCoor = [p[1] for p in Points]
    
    # sort values
    SortIndexes  = sorted(range(len(XCoor)), key=lambda k: XCoor[k])
    SortedX = sorted(XCoor)
    SortedY = [] 
    for i in SortIndexes:
        SortedY.append(YCoor[i])
    
    if SortedX[0] == SortedX[-1] or SortedY[0] == SortedY[-1]:
        return Points

    # fit with polynomial of second orders
    Coeff = np.polyfit(SortedX, SortedY, 2)
    XPoly = np.linspace(SortedX[0], SortedX[-1], num=FitNum)
    YPoly = np.polyval(Coeff, XPoly)
    
    return [np.asarray([x, y]) for x, y in zip(XPoly, YPoly)]
