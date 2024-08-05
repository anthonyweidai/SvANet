def checkExp(ModelConfigDict: dict):
    # check if there is expension layer on the finnal stage
    CfgDict1 = ModelConfigDict[list(ModelConfigDict)[-1]]
    CfgDict2 = ModelConfigDict[list(ModelConfigDict)[-2]]
    if CfgDict1['in'] != CfgDict1['out'] and \
        CfgDict1['stage'] == CfgDict2['stage']:
            return True
    else:
        return False
    
    
def computeMaxStage(ModelConfigDict: dict):
    # get the maximum stage of backbone
    return ModelConfigDict[list(ModelConfigDict)[-1]]['stage']


def getLastIdxFromStage(ModelConfigDict: dict, Stage: int): # , CheckExpMode: bool=False
        # get the last layer Index among those layers with the same stride stages
        StageOri = Stage
        if StageOri < 0:
            # MaxStage = computeMaxStage(ModelConfigDict)
            Stage = ModelConfigDict[list(ModelConfigDict)[-1]]['stage'] + Stage + 1
            # if CheckExpMode:
            #     if checkExp(ModelConfigDict) and StageOri != -1:
            #         Stage -= 1
                
        StageIdx = -1
        for Config in ModelConfigDict:
            if ModelConfigDict[Config]['stage'] <= Stage:
                StageIdx += 1
            else:
                break
            
        return StageIdx


def getAllLayerIndex(ModelConfigDict: dict):
    LayerIndexes = []
    MaxStage = computeMaxStage(ModelConfigDict)
    for i in range(1, MaxStage + 1):
        LayerIndexes.append(getLastIdxFromStage(ModelConfigDict, i))
    return LayerIndexes


def getChannelsbyLayer(ModelConfigDict: dict, LayerIdx: int):
    # get the out channels by using stage index
    return ModelConfigDict[list(ModelConfigDict)[LayerIdx]]['out']


def getChannelsbyStage(ModelConfigDict: dict, Stage: int):
    # get the out channels by using stage number
    LayerIdx = getLastIdxFromStage(ModelConfigDict, Stage)
    return getChannelsbyLayer(ModelConfigDict, LayerIdx)
