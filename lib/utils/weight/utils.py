import os
import csv
import numpy as np
from pathlib import Path

import torch

from ..variables.defaultVar import LogMetric


def pretrainedDictManager(PretrainedDict):
    # Get weight if pretrained weight has the partly same structure but diffetnent keys
    # initialize keys and values to keep the original order
    if len(PretrainedDict) == 1 or len(PretrainedDict) == 5:
        # 1 for convnext pth, 5 for beit2 vit pth
        PretrainedDict = PretrainedDict['model']
    elif len(PretrainedDict) == 2:
        PretrainedDict = PretrainedDict['state_dict'] # for triplet pth
    elif len(PretrainedDict) == 4:
        PretrainedDict = PretrainedDict['model_state'] # for SegNet pyth
    return PretrainedDict


def interpolatePosEmbed(Model, PretrainedDict):
    # --------------------------------------------------------
    # Interpolate position embeddings for high-resolution
    # References:
    # DeiT: https://github.com/facebookresearch/deit
    # --------------------------------------------------------
    EmbedKeys = [
        'pos_embed', # MAE, BEiT
        'PosEmbedding', # ours ViT
    ]
    for k in EmbedKeys:
        if k in PretrainedDict:
            PosEmbedCheckpoint = PretrainedDict[k]
            EmbeddingSize = PosEmbedCheckpoint.shape[-1]
            try:
                # for encoder-decoder
                NumPatches = Model.Encoder.patch_embed.num_patches
                NumExtraTokens = getattr(Model.Encoder, k).shape[-2] - NumPatches
            except:
                NumPatches = Model.patch_embed.num_patches
                NumExtraTokens = getattr(Model, k).shape[-2] - NumPatches
            # height (== width) for the checkpoint position embedding
            OrigSize = int((PosEmbedCheckpoint.shape[-2] - NumExtraTokens) ** 0.5)
            # height (== width) for the new position embedding
            NewSize = int(NumPatches ** 0.5)
            # class_token and dist_token are kept unchanged
            if OrigSize != NewSize:
                print("Position interpolate from %dx%d to %dx%d" % (OrigSize, OrigSize, NewSize, NewSize))
                ExtraTokens = PosEmbedCheckpoint[:, :NumExtraTokens]
                # only the position tokens are interpolated
                PosTokens = PosEmbedCheckpoint[:, NumExtraTokens:]
                PosTokens = PosTokens.reshape(-1, OrigSize, OrigSize, EmbeddingSize).permute(0, 3, 1, 2)
                PosTokens = torch.nn.functional.interpolate(
                    PosTokens, size=(NewSize, NewSize), mode='bicubic', align_corners=False)
                PosTokens = PosTokens.permute(0, 2, 3, 1).flatten(1, 2)
                NewPosEmbed = torch.cat((ExtraTokens, PosTokens), dim=1)
                PretrainedDict[k] = NewPosEmbed
    return PretrainedDict


def wightFrozen(Model, FreezeWeight, TransferFlag=0,  PreTrained=1):
    # ModelDict = Model.state_dict()
    if FreezeWeight == 0:
        print("Skip freezing layers")
        return Model
    else:
        Idx = 0
        for Name, Param in Model.named_parameters():
            if FreezeWeight == 1: 
                # if 'features' in Name:
                # Judger = 'classifier' not in Name.lower()
                # if PreTrained == 2:
                #     Judger = Judger and idx < len(ModelDict.keys()) - 7
                    
                if 'classifier' not in Name.lower():
                    Param.requires_grad = False
                else:
                    print(Name, Param.requires_grad)      
            elif FreezeWeight == 2:
                # Freeze the layers with transferred weight
                while 1:
                    if TransferFlag[Idx] == 1:
                        Param.requires_grad = False
                        break
                    elif TransferFlag[Idx] == 2:
                        Idx += 1
                    else:
                        print(Name, Param.requires_grad)
                        break
                Idx += 1
            elif FreezeWeight == 3:
                # For step weight freezing
                Param.requires_grad = True
            elif FreezeWeight == 4:
                Param.requires_grad = False
            else:
                print(Name, Param.requires_grad)
                
        if FreezeWeight == 3:
            print("Unfreeze all layers")
        elif FreezeWeight == 4:
            print("Freeze all layers")
            
        return Model


def ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, FalseFlag=0):
    if any(s in NewKey for s in IgnoreStrList):
        TransferFlag[IdxNew] = 2
    else:
        TransferFlag[IdxNew] = FalseFlag
    return TransferFlag


def loadModelWeight(Model, FreezeWeight, PreTrainedWeight, 
                    PreTrained=1, DistMode=False, DropLast=False, SkipSELayer=False):
    '''
    cannot load pkl file, "Invalid magic number; corrupt file?"
    '''
    # correct path with default file extension
    if not os.path.splitext(PreTrainedWeight)[1]:
        PreTrainedWeight += '.pth'
    if not os.path.isfile(PreTrainedWeight):
        print("Skip empty weight path: %s" % PreTrainedWeight)
        return Model
    
    IgnoreStrList = ["running_mean", "running_var", "num_batches_tracked"] # pytorch 1.10
    
    print('Knowledge transfer from: %s' % PreTrainedWeight)
    
    # load checkpoint to CPU to avoid CUDA memory leak
    PretrainedDict = torch.load(PreTrainedWeight, map_location=torch.device('cpu'))
    
    ModelDict = Model.state_dict()
    TransferFlag = np.zeros((len(ModelDict), 1))
    if PreTrained == 1 or PreTrained == 3:
        # Get weight if pretrained weight has the same dict
        if PreTrained == 1:
            # PretrainedDictNew = {}
            # for Idx, (k, v) in enumerate(PretrainedDict.items()):
            #     if k in ModelDict and 'classifier' not in k.lower():
            #         Temp = {k: v}
            #         PretrainedDictNew.append(Temp)
            #         TransferFlag[Idx] = 1
            # PretrainedDict = PretrainedDictNew
            PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and ('classifier' not in k.lower() or DistMode)}
        else:
            PretrainedDict = {k.replace('module.',''): v for k, v in PretrainedDict.items()}
            TransferFlag.fill(1)
        # PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and 'classifier' not in k and 'fc' not in k}
    elif PreTrained == 2 or PreTrained == 4:
        PretrainedDict = pretrainedDictManager(PretrainedDict)
        PreTrainedWeight = interpolatePosEmbed(Model, PretrainedDict)
            
        OldDictKeys = list(PretrainedDict.keys())
        OldValues = list(PretrainedDict.values())
        NewDictKeys = list(ModelDict.keys())
        NewValues = list(ModelDict.values())
        
        Len1 = len(PretrainedDict)
        Len2 = len(ModelDict)
        LenFlag =  Len1 > Len2
        MaxLen = max(Len1, Len2)
        
        Count = IdxNew = IdxOld = 0
        for _ in range(MaxLen):
            if IdxOld >= Len1 or IdxNew >= Len2:
                break
            
            OldKey = OldDictKeys[IdxOld]
            OldVal = OldValues[IdxOld]
            NewKey = NewDictKeys[IdxNew]
            NewVal = NewValues[IdxNew]
            
            TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew)
            
            if not (PreTrained == 4 or DistMode):
                Flag = 0
                if 'classifier.' in OldKey.lower():
                    PretrainedDict.pop(OldKey)
                    IdxOld += 1
                    Flag = 1
                if 'classifier.' in NewKey.lower():
                    IdxNew += 1
                    Flag = 1
                if Flag:
                    continue # for multi classifier
                    # break
            
            SEJudge = False # "selayer" in NewKey.lower() or "mclayer" in NewKey.lower()
            if OldVal.shape == NewVal.shape:
                if not (SkipSELayer and SEJudge): # comment this for other projects
                    PretrainedDict[NewKey] = PretrainedDict.pop(OldKey)
                    TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, 1)
                    Count += 1
            elif LenFlag:
                if SEJudge:
                    IdxOld -= 1
                else:
                    IdxNew -= 1
                # PretrainedDict.pop(OldKey)
            else:
                IdxOld -= 1
            IdxNew += 1
            IdxOld += 1

            if DropLast:
                if LenFlag and IdxOld == len(OldDictKeys) - 2:
                    break
                elif IdxNew == len(NewDictKeys) - 2:
                    break
    
        print('The number of transferred layers: %d' %(Count))
 
    ModelDict.update(PretrainedDict)
    Model.load_state_dict(ModelDict, strict=False)
    return wightFrozen(Model, FreezeWeight, TransferFlag, PreTrained)


def saveModel(SavePath, Model):
  if isinstance(Model, torch.nn.DataParallel):
    StateDict = Model.module.state_dict()
  else:
    StateDict = Model.state_dict()
  torch.save(StateDict, SavePath)


def getBestValprt(FolderPath):
    # keep the best th validation/repetition model name
    KeepValRpt = None
    MetricCSVPath = FolderPath + '/best_metrics.csv'
    if os.path.isfile(MetricCSVPath):
        with open(MetricCSVPath) as f:
            LogFileReader = csv.DictReader(f)
            HeadNames = LogFileReader.fieldnames
            for m in LogMetric:
                MetricName = [n for n in HeadNames if m in n.lower()]
                if MetricName:
                    MetricName = MetricName[0]
                    break
            LogList = list(LogFileReader)
            BestVals = [l[MetricName] for l in LogList]
            BestVal = max(BestVals)
            KeepValRpt = LogList[BestVals.index(BestVal)]['K-Fold'].replace('_', '')
    else:
        print("%s is not existed" % MetricCSVPath)
    
    return KeepValRpt
