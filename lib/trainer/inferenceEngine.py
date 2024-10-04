## Import module
 # path manager
from tqdm import tqdm
 # torch module
import torch
from torch.utils.data import DataLoader
 # my module
from ..model import getModel
from ..params_init import initPathMode
from ..data import getMyDataset, buildCollateFn
from ..utils import (
    WinCols, getImgPath, getSetPath, loadModelWeight, 
    moveToDevice, getLoaderBatch, workerManager
)


class Inferencer(object):
    def __init__(self, opt, Split=0, UseTrainSet=False) -> None:
        self.opt = opt
        
        if not opt.get_path_mode:
            opt.get_path_mode = initPathMode(opt.dataset_path, opt.setname.lower())
        
        self.TrainSet, self.TestSet = getImgPath(opt.dataset_path, opt.num_split, Mode=opt.get_path_mode)
        self.TrainSet, self.TestSet = getSetPath(self.TrainSet, self.TestSet, Split) # train set can also be inferenced
        TestImgs = getMyDataset(opt, self.TrainSet if UseTrainSet else self.TestSet, IsTraining=False)
        
        self.LoaderBatch = getLoaderBatch(opt)
        self.CollateFn = buildCollateFn(opt)
        
        self.PinMemory, self.NumWorkers = workerManager(opt)
        self.InferDL = DataLoader(TestImgs, self.LoaderBatch, num_workers=self.NumWorkers, shuffle=False, 
                                    pin_memory=self.PinMemory, collate_fn=self.CollateFn, drop_last=opt.drop_last)
        
        Model = getModel(opt=opt).to(opt.device)
        
        if opt.pretrained:
            Model = loadModelWeight(Model, opt.freeze_weight, opt.pretrained_weight, 
                                    opt.pretrained, opt.dist_mode)
        
        self.Model = Model
        
    def inference(self, GetTarget=True, GetPrediction=True):
        """ special mode with gradient """
        # Start inferencing
        print(('Inferencing').center(WinCols, '*'))
        
        self.Model.eval()
        Target, Prediction = [], []
        for Batch in tqdm(self.InferDL, ncols=WinCols - 9, colour='magenta'):
            Batch = moveToDevice(Batch, self.opt.device)
            Img = Batch['image']
            # for memory saving
            if GetTarget:
                Target.append(Batch['label'])
            if GetPrediction:
                Prediction.append(self.Model(Img))
        
        return (
            torch.cat(Target, dim=0) if GetTarget else Target, 
            torch.cat(Prediction, dim=0) if GetPrediction else Prediction
        )
        
    @torch.inference_mode()
    def run(self, GetTarget=True, GetPrediction=True):
        """ efficient mode without gradient 
        Attention: decorator cannot be replaced or updated
        """
        return self.inference(GetTarget, GetPrediction)