import random
from typing import Any, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base import BaseModule
from .utils import HEAD_OUT_CHANNELS
from .module import FGBottleneck, CSLayer, FGLink
from ...modules import AssembleFormer, SqueezeExcitation, MoCAttention
from ...layers import StochasticDepth
from ....utils import pair, setMethod, callMethod


class BaseSegHead(BaseModule):
    def __init__(self, opt, ModelConfigDict, **kwargs: Any):
        super().__init__(opt, ModelConfigDict, **kwargs)
        self.FMGChannels = None
        
        if opt.seg_feature_guide != 0:
            SELayerList = [SqueezeExcitation, MoCAttention]
            ViTBlockList = [AssembleFormer]
            
            # neck block module
            if opt.fg_bottle == 1:
                self.NeckBlock = FGBottleneck
            else:
                self.NeckBlock = nn.Identity
            
            self.NeckSELayer = None
            if opt.fg_bottle_se > 0:
                self.NeckSELayer = SELayerList[self.opt.fg_bottle_se - 1]
            
            # vit module
            ## cross-scale block module vit
            self.ViTBlock = None
            self.ViTSELayer = None
            if self.opt.fg_vit > 0:
                self.ViTBlock = ViTBlockList[self.opt.fg_vit - 1]
                if opt.fg_vit_se > 0:
                    self.ViTSELayer = SELayerList[self.opt.fg_vit_se - 1]
            ## link module vit (same as cross-scale block module vit in initialisation)
            self.LinkViTBlock = None
            self.LinkViTSELayer = None
            if self.opt.fg_link_vit > 0:
                self.LinkViTBlock = ViTBlockList[self.opt.fg_link_vit - 1]
                if opt.fg_link_vit_se > 0:
                    self.LinkViTSELayer = SELayerList[self.opt.fg_link_vit_se - 1]
            ## final branch (with cs guidance) vit
            self.SegHeadViTBlock = None
            self.HeadViTSELayer = None
            if opt.fg_seghead_vit > 0:
                self.SegHeadViTBlock = ViTBlockList[self.opt.fg_seghead_vit - 1]
                if opt.fg_seghead_vit_se > 0:
                    self.HeadViTSELayer = SELayerList[self.opt.fg_seghead_vit_se - 1]

            self.initFeatureMapGuide()
            
    def initFeatureMapGuide(self):
        # v2: bottleneck and basic block with vit
        StartStage = self.opt.fg_start_stage
        MaxStage = self.computeMaxStage(self.ModelConfigDict)
        self.NumBranches = MaxStage - StartStage + 1
        if not self.opt.fg_for_head:
            # the branch, final one use seg head without fg for head
            self.NumBranches -= 1
            NumAttnBlocksList = [2, 3]
        else:
            NumAttnBlocksList = [2, 4, 3]

        # Counter = 0
        FMGChannels = 0
        for b in range(self.NumBranches): 
            InChannels = self.getChannelsbyStage(self.ModelConfigDict, b + StartStage)
            FMGChannels += InChannels
            
            # use vit
            ViTBlock = None
            NumAttnBlocks = 0
            if self.opt.fg_vit:
                # skip the first two blocks
                if self.opt.fg_vit_all:
                    NumAttnBlocks = 2
                    ViTBlock = self.ViTBlock
                elif b > 2 - StartStage:
                    NumAttnBlocks = NumAttnBlocksList[b + StartStage - 3]
                    ViTBlock = self.ViTBlock
            
            # drop path
            DropRate = 0.2 * float(b) / self.NumBranches
            # branch dropout
            # BranchAttnDropRate = DropRate / 2 if self.opt.fg_branch_drop else 0
            ## cross dropout
            for t in range(b + 1, self.NumBranches): # target branch
                Dropout = StochasticDepth(DropRate)
                DCName = 'Dropr%d%d' % (b + 1, t + 1)
                setMethod(self, DCName, Dropout)

            # branch conv and cross conv
            if self.opt.seg_feature_guide == 1:
                self.NumCrossStages = self.NumBranches
                for c in range(self.NumCrossStages): # the cross stage, final one use only parallel conv
                    if c + 1 > b: # check if branch b has stage c 
                        Name = 'ConvB%dC%d' % (b + 1, c + 1)
                        setMethod(self, Name, self.NeckBlock(InChannels=InChannels, 
                                                             SELayer=self.NeckSELayer,
                                                             MoCOrder=self.opt.moc_order,
                                                             PoolRes=self.opt.moc_pool_res,
                                                             ViTBlock=ViTBlock, 
                                                             NumAttnBlocks=NumAttnBlocks, 
                                                             ViTSELayer=self.ViTSELayer, 
                                                            ))
                        # Counter += 1
                        if self.opt.fg_use_guide and 0 < c < self.NumCrossStages:
                            for t in range(b + 1, c + 1): # target branch
                                if t < self.NumBranches:
                                    OutChannels = self.getChannelsbyStage(self.ModelConfigDict, t + StartStage)
                                    CrossLayer = CSLayer(self.opt, InChannels, OutChannels, 2, t - b, b + StartStage)
                                    
                                    CName = 'ConvC%dr%d%d' % (c + 1, b + 1, t + 1)
                                    setMethod(self, CName, CrossLayer)
                                    # Counter += 1
                
            else:
                # v2: bottleneck and basic block with vit
                Name = 'ConvB%d' % (b + 1)
                setMethod(self, Name, self.NeckBlock(InChannels=InChannels, 
                                                     SELayer=self.NeckSELayer, 
                                                     MoCOrder=self.opt.moc_order,
                                                     PoolRes=self.opt.moc_pool_res,
                                                     ViTBlock=ViTBlock, 
                                                     NumAttnBlocks=NumAttnBlocks, 
                                                     ViTSELayer=self.ViTSELayer, 
                                                    ))
                # Counter += 1
                
                if self.opt.fg_use_guide:
                    for t in range(b + 1, self.NumBranches): # target branch
                        OutChannels = self.getChannelsbyStage(self.ModelConfigDict, t + StartStage)
                        CrossLayer = CSLayer(self.opt, InChannels, OutChannels, 2, t - b, b + StartStage) \
                        
                        CName = 'Convr%d%d' % (b + 1, t + 1)
                        setMethod(self, CName, CrossLayer)
                        # Counter += 1
        
        if self.opt.fg_link:
            self.LinkHead = FGLink(
                self.opt, 
                self.ModelConfigDict,
                ViTBlock=self.LinkViTBlock, 
                NumAttnBlocks=NumAttnBlocks, 
                ViTSELayer=self.LinkViTSELayer
            )
        # self.UpFlag = (
        #     self.opt.fg_link and 
        #     hasattr(self, 'forwardDecodeFeature') and 
        #     callable(self.forwardDecodeFeature)
        # )
        if self.SegHeadViTBlock is not None:
            HeadChannels = HEAD_OUT_CHANNELS.get(
                self.opt.seg_head_name, 
                HEAD_OUT_CHANNELS['default']
            )
            self.SegHeadViTBlock = self.SegHeadViTBlock(
                HeadChannels, 
                NumAttnBlocks=4, 
                ViTSELayer=self.HeadViTSELayer,
                MoCOrder=self.opt.moc_order,
            )
        else:
            self.SegHeadViTBlock = nn.Identity()
        
        # print(Counter)
        self.FMGChannels = FMGChannels
    
    def forwardFMG(self, FeaturesTuple) -> Union[Tensor, Tuple[Tensor]]:
        LayerIndexes = self.getAllLayerIndex(self.ModelConfigDict)
        
        LeftSupFeature = []
        if self.opt.fg_link:
            # controlled by the started stage
            LeftSupFeature = [FeaturesTuple[i] for i in LayerIndexes[0:self.opt.fg_start_stage - 1]]
            
        RightSupFeature = []
        if self.opt.fg_for_head:
            Output = [FeaturesTuple[i] for i in LayerIndexes[self.opt.fg_start_stage - 1:]]
        else:
            RightSupFeature = [FeaturesTuple[-1]]
            
            HeadOutput = self.forwardDecode(FeaturesTuple)
            HeadOutput = self.SegHeadViTBlock(HeadOutput)
            
            Output = [FeaturesTuple[i] for i in LayerIndexes[self.opt.fg_start_stage - 1:-1]]
        
        # Counter = 0
        if self.opt.seg_feature_guide == 1:
            for c in range(self.NumCrossStages): # the cross stage, final one use only parallel conv
                # finish the convolution on the current branch
                for b in range(self.NumBranches): # the branch
                    if c + 1 > b: # check if branch b has stage c 
                        Name = 'ConvB%dC%d' % (b + 1, c + 1)
                        Output[b] = callMethod(self, Name)(Output[b])
                        # Counter += 1
                
                if self.opt.fg_use_guide and 0 < c < self.NumCrossStages:
                    # guide feature map by diverse feature maps
                    for t in range(c + 1):
                        if t < self.NumBranches:
                            # up to down
                            for b in range(t):
                                CName = 'ConvC%dr%d%d' % (c + 1, b + 1, t + 1)
                                DCName = 'Dropr%d%d' % (b + 1, t + 1)
                                # + instead of += to prevent from backward errors
                                Output[t] = Output[t] + callMethod(self, DCName)(callMethod(self, CName)(Output[b]))
                                # Counter += 1
        else:
            # v2: bottleneck and basic block with vit
            for b in range(self.NumBranches): # the branch
                # finish the convolution on the current branch
                Name = 'ConvB%d' % (b + 1)
                Output[b] = callMethod(self, Name)(Output[b])
                # Counter += 1
            
            if self.opt.fg_use_guide:
                for b in range(self.NumBranches):
                    # guide feature map by diverse feature maps
                    for t in range(b + 1, self.NumBranches):
                        # up to down
                        CName = 'Convr%d%d' % (b + 1, t + 1)
                        DCName = 'Dropr%d%d' % (b + 1, t + 1)
                        Output[t] = Output[t] + callMethod(self, DCName)(callMethod(self, CName)(Output[b]))
                        # Counter += 1
        
        # print(Counter)
        
        if self.opt.fg_link:
            DecodeFeature = self.LinkHead.forwardFeature(LeftSupFeature + Output + RightSupFeature)
            if self.opt.fg_for_head:
                Output = [Output[-1]] + DecodeFeature
            else:
                Output = DecodeFeature
            Output.reverse()
        
        if not self.opt.fg_for_head:
            Output = Output + [HeadOutput]
        Output = [F.interpolate(o, size=list(Output[self.opt.fg_resize_stage].shape[-2:]), 
                                mode="bilinear", align_corners=True) for o in Output]
        
        if self.training and self.opt.fg_cat_shuffle:
            random.shuffle(Output)
        Output = torch.cat(Output, dim=1)
        Output = self.Classifier(Output)
        
        return F.interpolate(Output, size=pair(self.opt.resize_shape), mode="bilinear", align_corners=True)
