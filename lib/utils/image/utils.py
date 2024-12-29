import numpy as np
from glob import glob
from tqdm import tqdm
from typing import List, Optional

import cv2
from PIL import Image


def readImagePil(ImgPath) -> Image.Image:
    try:
        Img = Image.open(ImgPath).convert("RGB")
    except:
        Img = None
    return Img


def readMaskPil(MaskPath):
    try:
        Mask = Image.open(MaskPath)
        if Mask.mode != "L" and Mask.mode != "P":
            print("Mask mode should be L or P. Got: {}".format(Mask.mode))
        return Mask
    except:
        return None
    

def pil2OpenCV(Img: Image.Image) -> np.ndarray:
    # adapted from https://gist.github.com/panzi/1ceac1cb30bb6b3450aa5227c02eedd3
    ImgMode = Img.mode
    if ImgMode == "1":
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg *= 255
    elif ImgMode == "L":
        CVImg = np.array(Img, dtype=np.uint8)
    elif ImgMode == "LA" or ImgMode == "La":
        CVImg = np.array(Img.convert("RGBA"), dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_RGBA2BGRA)
    elif ImgMode == "RGB":
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_RGB2BGR)
    elif ImgMode == "RGBA":
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_RGBA2BGRA)
    elif ImgMode == "LAB":
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_LAB2BGR)
    elif ImgMode == "HSV":
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_HSV2BGR)
    elif ImgMode == "YCbCr":
        # XXX: not sure if YCbCr == YCrCb
        CVImg = np.array(Img, dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_YCrCb2BGR)
    elif ImgMode == "P" or ImgMode == "CMYK":
        CVImg = np.array(Img.convert("RGB"), dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_RGB2BGR)
    elif ImgMode == "PA" or ImgMode == "Pa":
        CVImg = np.array(Img.convert("RGBA"), dtype=np.uint8)
        CVImg = cv2.cvtColor(CVImg, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"unhandled image color mode: {ImgMode}")

    return CVImg, ImgMode
    

class Colormap(object):
    """
    Generate colormap for visualizing segmentation masks or bounding boxes.

    This is based on the MATLab code in the PASCAL VOC repository:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def __init__(self, n: Optional[int] = 256, normalized: Optional[bool] = False):
        super(Colormap, self).__init__()
        self.n = n
        self.normalized = normalized

    @staticmethod
    def getBitAtIdx(val, idx):
        return (val & (1 << idx)) != 0

    def getColourMap(self) -> np.ndarray:
        dtype = "float32" if self.normalized else "uint8"
        color_map = np.zeros((self.n, 3), dtype=dtype)
        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.getBitAtIdx(c, 0) << 7 - j)
                g = g | (self.getBitAtIdx(c, 1) << 7 - j)
                b = b | (self.getBitAtIdx(c, 2) << 7 - j)
                c = c >> 3

            color_map[i] = np.array([r, g, b])
        color_map = color_map / 255 if self.normalized else color_map
        return color_map

    def getBoxColourCodes(self) -> List:
        box_codes = []

        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.getBitAtIdx(c, 0) << 7 - j)
                g = g | (self.getBitAtIdx(c, 1) << 7 - j)
                b = b | (self.getBitAtIdx(c, 2) << 7 - j)
                c = c >> 3
            box_codes.append((int(r), int(g), int(b)))
        return box_codes

    def getColorMapList(self) -> List:
        cmap = self.getColourMap()
        cmap = np.asarray(cmap).flatten()
        return list(cmap)

    
def measureMaskArea(Path):
    # measure the semantic object area within an image
    Img = readMaskPil(Path) if ".png" in Path else readImagePil(Path)
    Img, _ = pil2OpenCV(Img)
    GrayImg = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    
    ContourAreas = []
    for u in np.unique(GrayImg):
        # skip black background
        if u == 0:
            continue
        
        GrayImgCls = np.copy(GrayImg)
        GrayImgCls[GrayImgCls != u] = 0
        
        # binarisation, (0, 0, 0) is background and the others are interested objects
        _, Thred = cv2.threshold(GrayImgCls, 1, 255, cv2.THRESH_BINARY)
        
        # open operation
        Kernal = np.ones((2,2), np.uint8)
        OpeningEdges = cv2.morphologyEx(Thred, cv2.MORPH_CLOSE, Kernal, iterations=2)
        
        Contours, _ = cv2.findContours(OpeningEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ContourArea = 0
        for i in range(len(Contours)):
            ContourTemp = Contours[i]
            AreaTemp = cv2.contourArea(ContourTemp)
            if ContourArea < AreaTemp:
                ContourArea = AreaTemp
        ContourAreas.append(ContourArea)
    return np.asarray(ContourAreas) / GrayImg.size


def getAllMaskArea(RootPath=None, MaskPaths=None):
    # measure the mask largest object (within a folder) area ratio and plot
    if MaskPaths is None:
        MaskPaths = glob("%s/*.png" % RootPath) + glob("%s/*.jpg" % RootPath) # could have png/jpg mask
    
    AreaRatios = []
    for m in tqdm(MaskPaths, colour="blue", ncols=50):
        AreaRatio = measureMaskArea(m)
        AreaRatios.append(AreaRatio * 100)
    return np.asarray(AreaRatios, dtype="object")


def largeGrey2RGB(Img, MaxPixVal=None):
    # convert large grey scale image to RGB image
    # supported format: I;16
    if not isinstance(Img, np.ndarray):
        Img = np.asarray(Img)
    MaxPixValTemp = np.max(Img) # cannot remove outlier, or it will damage image info
    if MaxPixVal is None or MaxPixValTemp > MaxPixVal:
        MaxPixVal = MaxPixValTemp
        
    Img = Img.astype("float32") / MaxPixVal * 255
    # Img = cv2.normalize(Img, Img, 0, 255, cv2.NORM_MINMAX)
    return Img.astype("uint8")
