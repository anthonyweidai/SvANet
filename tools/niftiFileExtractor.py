from glob import glob
from tqdm import tqdm
from pathlib import Path

import math
import numpy as np
from PIL import Image
import SimpleITK as sitk

from lib.utils import seedSetting, largeGrey2RGB, COLOUR_CODES


def itkMetaInfo(Img: sitk.Image):
    # Define function which extracts header information in a dictionary
    Header = {} 
    for k in Img.GetMetaDataKeys():
        Header[k] = Img.GetMetaData(k)
    return(Header)


def niftiExtractor(FilePath, TargetPath, Level, ColourCode, Gap=4, Mode='MRI', RandomIdxs=None):
    # Extract and save data in the created directory (in numpy format)
    ITKImg = sitk.ReadImage(FilePath)
    # Extract and save image data in numpy format
    CTMRIImg = sitk.GetArrayFromImage(ITKImg)  # multidimensional array
    # Extract and save header data in numpy format
    Header = itkMetaInfo(ITKImg)
    
    # encode and save
    NumSlices = CTMRIImg.shape[0]
    if Mode == 'CT':
        NumSlices = CTMRIImg.shape[-1]
        if RandomIdxs is None:
            RandomIdxs = np.random.choice(range(round(NumSlices * 0.15), round(NumSlices * 0.85)), 
                                        math.ceil(NumSlices * Gap), replace=False)
        
    for i in range(NumSlices):
        if RandomIdxs is not None:
            Flag =  i in RandomIdxs
        else:
            Flag = i % Gap == 0 or i == CTMRIImg.shape[0] - 1
        if Flag:
            if Mode == 'MRI':
                Img = CTMRIImg[i]
            else:
                # cannot reshape CTMRIImg
                Img = CTMRIImg[:, :, i]
                Img = Img.transpose((1, 0))
            
            if 'images' in Level:
                if Mode == 'MRI':
                    Img = largeGrey2RGB(Img) # , MaxPixVal=4095
                Img = Image.fromarray(Img)
                Img = Img.convert('RGB')
            else:
                Img = Image.fromarray(Img)
                Img = Img.convert('P') # for colour palette
                Img.putpalette(list(np.asarray(ColourCode).flatten()))
            
            Img.save('%s_%d.%s' % (TargetPath, i, 'jpg' if 'images' in Level else 'png'))
            
    return RandomIdxs


if __name__ == "__main__":
    """ Adapted from 
    https://gist.github.com/Azzedine-Touazi/0fdeace5dc3fddb134136e7cfffdca30#file-extract_nifti-py-L44
    Extract sequence of images from .nii.gz file
    ATLAS labels: 1 liver, 2 tumour
    KiTS23 labels: 1 kidney, 2 tumor, 3 cyst
    """
    seedSetting(RPMode=True)
    Mode = 1 # 1 ATLAS, 2 KiTS23
    
    if Mode == 1:
        RootPath = r'D:\dataset\MRI CT\atlas-train-dataset-1.0.1'
        ColourCode = COLOUR_CODES['atlas']
        
        Level = ['imagesTr', 'labelsTr']
        TargetLevel = ['images', 'masks']
        for l1, l2 in zip(Level, TargetLevel):
            LabelPaths = glob('%s/train/%s/*.nii.gz' % (RootPath, l1))
            
            TargetPath = '%s/%s' % (RootPath, l2)
            Path(TargetPath).mkdir(parents=True, exist_ok=True)
            
            for p in tqdm(LabelPaths, colour='cyan', ncols=60):
                ''' mode 'I;16', [C, H, W] ''' 
                FileName = Path(p).stem if 'imagesTr' in l1 else Path(p).stem.replace('lb', 'im')
                TargetP = '%s/%s' % (TargetPath, FileName)
                niftiExtractor(p, TargetP, l2, ColourCode, Gap=4, Mode='MRI')
    else:
        RootPath = r'D:\dataset\MRI CT\kits23 2023'
        ColourCode = COLOUR_CODES['kits23']
        
        RandomIdxsList = []
        Level = ['imaging', 'segmentation'] # 
        TargetLevel = ['images', 'masks'] # 
        for i, (l1, l2) in enumerate(zip(Level, TargetLevel)):
            LabelPaths = glob('%s/dataset/*/*.nii.gz' % RootPath)
            LabelPaths = [p for p in LabelPaths if l1 in p]
            
            TargetPath = '%s/%s' % (RootPath, l2)
            Path(TargetPath).mkdir(parents=True, exist_ok=True)
            
            for j, p in enumerate(tqdm(LabelPaths, colour='cyan', ncols=60)): 
                ''' mode 'F', [W, H, C] ''' 
                TargetP = '%s/%s' % (TargetPath, Path(p).parent.name) # get last level of folder name
                if i == 0:
                    RandomIdxs = niftiExtractor(p, TargetP, l2, ColourCode, 0.02, 'CT')
                    RandomIdxsList.append(RandomIdxs)
                else:
                    # keep the same extracting sequence
                    niftiExtractor(p, TargetP, l2, ColourCode, 0.02, 'CT', RandomIdxsList[j])
                