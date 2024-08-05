import numpy as np
from ...utils import COLOUR_CODES


# for visualisation
def colorPalettePASCAL():
    OriColourCodes = COLOUR_CODES['pascal']
    
    AnimalColourCodes = [
        [0, 0, 0], # background
        [128, 128, 0],
        [64, 0, 0],
        [64, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        # [192, 128, 128], # person
        [128, 64, 0],
        ]
    
    Animal2OriIdx = {i: OriColourCodes.index(Code) for i, Code in enumerate(AnimalColourCodes)}
    Ori2AnimalIdx = {i: AnimalColourCodes.index(Code) for i, Code in enumerate(OriColourCodes) if Code in AnimalColourCodes}

    AnimalColourCodes = np.asarray(AnimalColourCodes).flatten()
    return OriColourCodes, list(AnimalColourCodes), Animal2OriIdx, Ori2AnimalIdx


def switchPalatte(OldPlatte, SwithIdx: dict):
    NpPlatte = np.asarray(OldPlatte)
    NewPlatte = NpPlatte.reshape(-1 ,3)
    for Key in SwithIdx:
        Temp = NewPlatte[SwithIdx[Key]]
        NewPlatte[SwithIdx[Key]] = NewPlatte[Key]
        NewPlatte[Key] = Temp
        
    return list(NewPlatte.flatten())
