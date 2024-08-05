from torch import Tensor
from typing import Dict, Any, Union

from .baseEncoder import BaseEncoder
from ...misc import moduleProfile, printProfLayer, printProfOverall


def classifierCounter(Input, Classifier, OverallParams, OverallMacs):
    # Dropout and pooling doesn't have params and macs was ignored
    Classifier_params, Classifier_macs = 0.0, 0.0
    if Classifier is not None:
        Input, Classifier_params, Classifier_macs = moduleProfile(module=Classifier, x=Input)
        print('{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M'.format('Classifier',
                                                                        'Params',
                                                                        round(Classifier_params / 1e6, 3),
                                                                        'MACs',
                                                                        round(Classifier_macs / 1e6, 3)))
    OverallParams += Classifier_params
    OverallMacs += Classifier_macs
    
    return OverallParams, OverallMacs


class BaseMVit(BaseEncoder):
    def __init__(self, opt, *args, **kwargs: Any):
        super(BaseMVit, self).__init__(opt, *args, **kwargs)
        self.Conv1 = None
        self.Layer1 = None
        self.Layer2 = None
        self.Layer3 = None
        self.Layer4 = None
        self.Layer5 = None
        self.Conv1x1Exp = None

        self.BranchFolding = None # Hier attention
        self.Conv1x1Hier = None # Hier attention
        self.Conv1x1Exp1 = None # ML loss
        self.Classifier1 = None # ML loss
        self.Conv1x1Exp2 = None # ML loss
        self.Classifier2 = None # ML loss
        
        self.round_nearest = 8

        self.ModelConfigDict = dict()


    def check_model(self):
        assert self.ModelConfigDict, "Model configuration dictionary should not be empty"
        assert self.Conv1 is not None, 'Please implement self.Conv1'
        assert self.Layer1 is not None, 'Please implement self.Layer1'
        assert self.Layer2 is not None, 'Please implement self.Layer2'
        assert self.Layer3 is not None, 'Please implement self.Layer3'
        assert self.Layer4 is not None, 'Please implement self.Layer4'
        assert self.Layer5 is not None, 'Please implement self.Layer5'
        assert self.Conv1x1Exp is not None, 'Please implement self.Conv1x1Exp'
        assert self.Classifier is not None, 'Please implement self.Classifier'
        
    
    def branchFeatures(self, x: Tensor):
        x = self.Conv1(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        
        InB1 = self.Layer3(x)
        InB2 = self.Layer4(InB1)
        InB3 = self.Layer5(InB2)
        return InB1, InB2, InB3
    
    def forwardFeatures(self, x: Tensor) -> Tensor:
        x = self.Conv1(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)

        x = self.Layer4(x)
        x = self.Layer5(x)
        x = self.Conv1x1Exp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forwardFeatures(x)
        x = self.Classifier(x)
        return x

    @staticmethod
    def profileLayers(layers, Input, OverallParams, OverallMacs, MultiFlag=0, Multiple=1):
        if not isinstance(layers, list):
            layers = [layers]

        for Layer in layers:
            if Layer is None:
                continue
            Input, LayerParam, LayerMACs = moduleProfile(module=Layer, x=Input)
            
            if MultiFlag:
                LayerMACs *= Multiple

            OverallParams += LayerParam
            OverallMacs += LayerMACs
            
            printProfLayer(Layer, LayerParam, LayerMACs)
        return Input, OverallParams, OverallMacs
    

    def profileModel(self, Input: Tensor, IsClassification: bool = True) -> Union[Tensor or Dict[Tensor], float, float]:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.
        OverallParams = OverallMacs = 0.0

        if IsClassification:
            print('Model statistics for an Input of size {}'.format(Input.size()))
            print('{:>35} Summary'.format(self.__class__.__name__))

        Flag = 0
        OutDict = {}
        if self.BranchFolding:
            InB1, InB2, InB3 = self.branchFeatures(Input)
            EmbInput, OverallParams, OverallMacs = self.BranchFolding.profileModule(InB1, InB2, InB3)
            OutDict["branch_fold"] = EmbInput
            Flag = 0
        elif self.Classifier1 and self.Classifier2:
            InB1, InB2, _ = self.branchFeatures(Input)
            InB1, OverallParams, OverallMacs = self.profileLayers(self.Conv1x1Exp1, InB1, OverallParams, OverallMacs)
            InB1, OverallParams, OverallMacs = self.profileLayers(self.Classifier1, InB1, OverallParams, OverallMacs)
            OutDict["out_b1"] = InB1
            
            InB2, OverallParams, OverallMacs = self.profileLayers(self.Conv1x1Exp2, InB2, OverallParams, OverallMacs)
            InB2, OverallParams, OverallMacs = self.profileLayers(self.Classifier2, InB2, OverallParams, OverallMacs)
            OutDict["out_b2"] = InB2
            Flag = 0
            
        Input, OverallParams, OverallMacs = self.profileLayers([self.Conv1, self.Layer1], Input, OverallParams, OverallMacs, Flag, 3)
        OutDict["out_l1"] = Input

        Input, OverallParams, OverallMacs = self.profileLayers(self.Layer2, Input, OverallParams, OverallMacs, Flag, 3)
        OutDict["out_l2"] = Input

        Input, OverallParams, OverallMacs = self.profileLayers(self.Layer3, Input, OverallParams, OverallMacs, Flag, 3)
        OutDict["out_l3"] = Input
        
        Input, OverallParams, OverallMacs = self.profileLayers(self.Layer4, Input, OverallParams, OverallMacs, Flag, 2)
        OutDict["out_l4"] = Input
        
        Input, OverallParams, OverallMacs = self.profileLayers(self.Layer5, Input, OverallParams, OverallMacs)
        OutDict["out_l5"] = Input

        if self.Conv1x1Hier:
            Input, OverallParams, OverallMacs = self.profileLayers(self.Conv1x1Hier, EmbInput, OverallParams, OverallMacs)
            OutDict["out_l6_hier"] = Input
        elif self.Conv1x1Exp:
            Input, OverallParams, OverallMacs = self.profileLayers(self.Conv1x1Exp, Input, OverallParams, OverallMacs)
            OutDict["out_l5_exp"] = Input
            
        if IsClassification:
            OverallParams, OverallMacs = classifierCounter(Input, self.Classifier, OverallParams, OverallMacs)
            
        printProfOverall(OverallParams, OverallMacs, self.parameters())
            
        return OutDict, OverallParams, OverallMacs
