import torch.nn as nn
from torch import Tensor

from ...layers import BaseConv2d


# resize shape 512 x 512 only
HEAD_OUT_CHANNELS = {**dict.fromkeys(["default", "pspnet", "deeplabv3", "spatial_ocr"], 512), 
                         "deeplabv3+": 560, "asp_ocr": 256, "lraspp": 128, 
                         "caranet": 32, "pranet": 64, "transnetr": 64, "cfanet": 64,}


class SegHeadClassifier(nn.Module):
    def __init__(self, FMGChannels, ClsInChannels, NumClasses) -> None:
        super().__init__()
        # use expansion will dramatically increase the model size
        
        if FMGChannels is not None:
            ClsInChannels += FMGChannels
        #     self.Classifier = nn.Sequential(
        #     # MobileOneBlock(ClsInChannels, ClsInChannels, 1, 1, UseSE=True), # 25.824 M, 77048.134 M
        #     # BaseConv2d(ClsInChannels, ClsInChannels, 1, 1, BNorm=True, ActLayer=nn.ReLU), # 25.689 M, 76842.088 M
        #     BaseConv2d(ClsInChannels, NumClasses, 1, 1), # 24.639 M, 24023.218 M
        # )

        self.Classifier = BaseConv2d(ClsInChannels, NumClasses, 1, 1)
        self.ClsInChannels = ClsInChannels
    
    def forward(self, x: Tensor) -> Tensor:
        return self.Classifier(x)
    