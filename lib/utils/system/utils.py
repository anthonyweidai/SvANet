import random
import numpy as np

import torch

from ..variables import TextColors
from ..mathematics import makeDivisible


def colourText(in_text: str, Mode=1, ColourName='') -> str:
    if ColourName:
        return TextColors[ColourName] + in_text + TextColors['end_colour']
    else:
        if Mode == 1:
            # cyan colour, params init/input params
            return TextColors['light_cyan'] + in_text + TextColors['end_colour']
        elif Mode == 2:
            # yellow colour, results output
            return TextColors['light_yellow'] + in_text + TextColors['end_colour']
        elif Mode == 3:
            # green colour, others
            return TextColors['light_green'] + in_text + TextColors['end_colour']


def seedSetting(RPMode, Seed=999):
    # Set random seed for reproducibility
    if RPMode:
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", Seed)
        np.random.seed(Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
    return


def deviceInit(opt):
    DataParallel = False
    CUDA_AVAI = torch.cuda.is_available()
    if CUDA_AVAI and opt.gpus != '-1':
        if len(opt.gpus) > 1:
            DeviceStr = 'cuda'
            DataParallel = True
        else:
            DeviceStr = 'cuda:' + str(opt.gpus[0])
    else:
        DeviceStr = 'cpu'
    return DataParallel, DeviceStr


def getLoaderBatch(opt):
    if opt.sup_method in ['common', 'autoencoder'] or 'shuffle' in opt.sup_method:
        LoaderBatch = opt.batch_size
    else:
        LoaderBatch = opt.batch_size / opt.views
        # self supervised dataloader provide different number of data
        if LoaderBatch.is_integer():
            # keep real batch size
            LoaderBatch = round(LoaderBatch)
        else:
            LoaderBatch = makeDivisible(LoaderBatch, 4)
    return LoaderBatch

