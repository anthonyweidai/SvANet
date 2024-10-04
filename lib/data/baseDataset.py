import os
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from typing import Optional, Any

import torch
from torch import nn, Tensor
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F

from .process import *
from .utils import initMeanStdByCsv
from ..utils import (
    COLOUR_CODES, pair, colourText, readImagePil, readMaskPil, makeDivisible
)


class CustomDataset(data.Dataset):
    def __init__(self, opt, ImgPaths, Transform=None, IsTraining: Optional[bool]=True, 
                 TargetSet=None, **kwargs: Any) -> None:
        super().__init__()
        self.opt = opt
        self.ImgPaths =  np.asarray(ImgPaths)
        self.TargetSet = np.asarray(TargetSet) if TargetSet is not None else TargetSet
        self.ClassNames = opt.class_names
        self.IsTraining = IsTraining
        
        """
        https://stats.stackexchange.com/questions/202287/why-standardization-of-the-testing-set-has-to-be-performed-with-the-mean-and-sd
        use only meanstd of train set, test meanstd vary from task
        Mean and std should be the same for train and test dataset
        """
        self.MeanStdType = "train" 
        self.MeanStdDP = 6 # DecimalPlaces
        
        self.readImagePil = readImagePil
        self.readMaskPil = readMaskPil
        
        self.initMeanStd()
        
        self.InterpolationMode = F.InterpolationMode.BICUBIC
        if not Transform:
            self.Transform = self.buildTransforms()

        print("The amount of original %s data: %s" % \
            ("train" if self.IsTraining else "test", 
             colourText(str(self.__len__()))))
    
    def __getitem__(self, Index):
        pass

    def __len__(self):
        return len(self.ImgPaths)
    
    def getNumEachClass(self):
        pass
    
    def getSenFactor(self):
        pass

    def callerInit(self):
        pass
    
    def buildTransforms(self):
        pass
    
    def initMeanStd(self):
        """ Normalization helps get data within a range and reduces 
        the skewness which helps learn faster and better.
        """
        self.MeanValues, self.StdValues = \
            initMeanStdByCsv(self.opt, self.IsTraining, self.MeanStdType, self.MeanStdDP)
            
        
class ClsBaseDataset(CustomDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.NumClasses = opt.cls_num_classes

    def buildTransforms(self):
        AugList, AugCommon = [], []
        if self.IsTraining:
            if "common" in self.opt.sup_method:
                AugResize = RandomResizedCrop(self.opt.resize_shape, Interpolation=self.InterpolationMode)
                AugCommon = [
                    RandomHorizontalFlip(),
                    RandomRotation(10, fill=0), # 180
                ]
                
            else:
                if "rotation" in self.opt.sup_method and self.opt.crop_mode != 0:
                    AugResize = RandomResizedCrop(self.opt.resize_shape, Interpolation=self.InterpolationMode)
                else:
                    AugResize = Resize(pair(self.opt.resize_shape), Interpolation=self.InterpolationMode)
                    
                if self.opt.auglikeclr:
                    s = 0.5
                    ColourJitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
                    AugCommon = [
                        RandomApply([ColourJitter], p=0.8),
                        RandomGrayscale(p=0.2),
                        RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    ]
                else:
                    pass
        else:
            ResizeRes = makeDivisible(self.opt.resize_shape * 1.15, 8)
            AugResize = Resize(pair(ResizeRes), Interpolation=self.InterpolationMode)
            """
            If size is a sequence like (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            """
            AugCommon = [CenterCrop(pair(self.opt.resize_shape))] # val tunes test process
        
        AugList.extend([AugResize] + AugCommon + [ToTensor()])
        if self.MeanValues:
            AugList.append(Normalisation(self.MeanValues, self.StdValues))
        
        return transforms.Compose(AugList)
    
    
class SegBaseDataset(CustomDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.NumClasses = opt.seg_num_classes
        # mapping instructions to map BGR mask to classes
        self.Mapping = COLOUR_CODES.get(opt.setname, COLOUR_CODES["default"])
    
    def getMask(self, ImgPath):
        MaksFolder = "mask"
        ImgName = Path(ImgPath).stem
        Ext = "png"
        
        MaskPath = "%s/%s/%s.%s" % (self.opt.dataset_path, MaksFolder, ImgName, Ext)
        return MaskPath
    
    @staticmethod
    def convertMask2Tensor(Mask):
        # convert to tensor
        Mask = np.array(Mask)
        if len(Mask.shape) > 2 and Mask.shape[-1] > 1:
            Mask = np.ascontiguousarray(Mask.transpose(2, 0, 1))
        return torch.as_tensor(Mask, dtype=torch.long)
    
    def mask2RGBClass(self, Mask: Tensor):
        """
        the lables of mask of sperm segementation dataset (semsperm) 
        are aleady in order, but has to consider mask fill in data augmentation
        """
        
        # check the present values in the mask, 0 and 255 in my case
        # print("unique values rgb    ", torch.unique(Mask)) 
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)
        
        MaskFillIdx = Mask == self.opt.ignore_idx # self.opt.mask_fill
        # assert torch.sum(Mask == 255) == torch.sum(Mask >= len(self.Mapping)), "Wrong index" 
        Mask[MaskFillIdx] = torch.tensor(0, dtype=torch.long) # background
        # for i, m in enumerate(self.Mapping):
        #     Validx = (Mask == torch.tensor(m, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))   
        #     Validx = (Validx.sum(0) == 3)  
        # MaskOut[Validx] = torch.tensor(i, dtype=torch.long)

        MaskOut = nn.functional.one_hot(Mask, self.NumClasses)
        MaskOut = MaskOut.permute(2, 0, 1).contiguous()
        
        # MaskOut[:, MaskFillIdx] = torch.tensor(self.opt.ignore_idx, dtype=torch.long)
        # check the present values after mapping, in my case 0, 1, 2, 3
        # print("unique values mapped ", torch.unique(MaskOut))
        # -> unique values mapped  tensor([0, 1, 2, 3])
        return MaskOut
    
    def buildTransforms(self):
        AugList = []
        if self.IsTraining:
            if "common" in self.opt.sup_method:
                AugResize = RandomShortSizeResize(
                    makeDivisible(self.opt.resize_shape // 2, 4), 
                    makeDivisible(self.opt.resize_shape * 1.5, 4), 
                    makeDivisible(self.opt.resize_shape * 2, 4), 
                    self.InterpolationMode
                ) # the params are also suitable for small objects
                # AugResize = RandomResizedCrop(self.opt.resize_shape, Interpolation=self.InterpolationMode)
                AugList.append(AugResize)
                
                AugCommon = [
                    RandomHorizontalFlip(),
                    RandomCrop(self.opt.resize_shape, pad_if_needed=True, fill=self.opt.ignore_idx),
                    RandomGaussianBlur(),
                    PhotometricDistort(),
                    RandomRotation(10, fill=self.opt.ignore_idx),
                ]
                if self.opt.random_aug_order:
                    AugList.append(RandomOrder(AugCommon))
                else:
                    AugList.extend(AugCommon)
                    
            else:
                AugList.extend([Resize(self.opt.resize_shape, self.InterpolationMode)])
        else:
            AugList.extend([Resize(self.opt.resize_shape, self.InterpolationMode)])
        
        AugList.append(ToTensor())
        if self.MeanValues:
            AugList.append(Normalisation(self.MeanValues, self.StdValues))
 
        return transforms.Compose(AugList)
    