from torch import nn
from . import moduleProfile

def printProfLayer(Layer, LayerParam, LayerMACs):
    if isinstance(Layer, nn.Sequential):
        ModuleName = "\n+".join([l.__class__.__name__ for l in Layer])
    else:
        ModuleName = Layer.__class__.__name__
    print(
        '{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format(ModuleName,
                                                                    'Params',
                                                                    round(LayerParam / 1e6, 3),
                                                                    'MACs',
                                                                    round(LayerMACs / 1e6, 3)
                                                                    ))
    
    
def printProfOverall(OverallParams, OverallMacs, ModelParams):
    print('{:<20} = {:>8.3f} M'.format('Overall parameters', OverallParams / 1e6))
    # Counting Addition and Multiplication as 1 operation
    print('{:<20} = {:>8.3f} M'.format('Overall MACs', OverallMacs / 1e6))
    OverallParamsPy = sum([p.numel() for p in ModelParams])
    print('{:<20} = {:>8.3f} M'.format('Overall parameters (sanity check)', OverallParamsPy / 1e6))
    

def profFeatures(Input, OutDict, OverallParams, OverallMacs, FeaturesLayer):
    for Idx, Layer in enumerate(FeaturesLayer):
        if Layer is None:
            continue
        if isinstance(Layer, nn.MaxPool2d):
            Input = Layer(Input)
            LayerParam = LayerMACs = 0.0
        else:
            Input, LayerParam, LayerMACs = moduleProfile(module=Layer, x=Input)
            OutDict["out_f" + str(Idx)] = Input

        OverallParams += LayerParam
        OverallMacs += LayerMACs
        
        printProfLayer(Layer, LayerParam, LayerMACs)
        
    return Input, OverallParams, OverallMacs


def profClassifier(Input, OutDict, OverallParams, OverallMacs, ClassifierLayer, ModelParams):
    _, LayerParam, LayerMACs = moduleProfile(module=ClassifierLayer, x=Input)
    OutDict["out_classifier"] = Input
    OverallParams += LayerParam
    OverallMacs += LayerMACs
    
    printProfOverall(OverallParams, OverallMacs, ModelParams)
    return OutDict, OverallParams, OverallMacs