## Import module
 # path manager
import time
from tqdm import tqdm
 # torch module
import torch
from torch.optim.lr_scheduler import MultiStepLR
 # my module
from .utils import reparameteriseModel
from .baseTrainer import BaseTrainer
from ..model import getModel
from ..optim import buildOptimiser
from ..optim.scheduler import buildScheduler
from ..loss_fn.utils import mixCriterion
from ..utils import WinCols, colourText, loadModelWeight, wightFrozen, moveToDevice


class Trainer(BaseTrainer):
    def __init__(self, opt, DataParallel) -> None:
        super().__init__(opt, DataParallel)
        
    def runnerInit(self, Rnd, Split):
        # separate initialisation for inferFeature
        opt = self.opt
        self.splitInit(opt)
        
        self.Rnd = Rnd
        self.Split = Split
            
        Model = getModel(opt=opt).to(opt.device)
        
        if self.DataParallel:
            Model = torch.nn.DataParallel(Model)
            # Model = DDP(Model) # self.Optim, grad, and backward should also be changed
            
        if opt.pretrained:
            if opt.freeze_weight == 3:
                Model = loadModelWeight(Model, 2, opt.pretrained_weight, 
                                        opt.pretrained, opt.dist_mode)
            else:
                Model = loadModelWeight(Model, opt.freeze_weight, opt.pretrained_weight, 
                                        opt.pretrained, opt.dist_mode)

        self.Optim = buildOptimiser(Model.parameters(), opt)
        
        self.Model = Model
        
        if "cosine" in opt.schedular:
            self.Scheduler = buildScheduler(opt, Optimiser=self.Optim)
        else:
            # Scheduler = StepLR(self.Optim, step_size=20, gamma=0.9)
            # Scheduler = ReduceLROnPlateau(self.Optim, mode="min", factor=0.05, patience=5, min_lr=0.0001)
            self.Scheduler = MultiStepLR(self.Optim, milestones=[opt.milestones], gamma=0.1) 
        
        self.dataLoaderSetting(opt)
        
    def run(self, Rnd, Split):
        self.runnerInit(Rnd, Split)
        opt = self.opt # run() is used outside
        self.LrList = []
        
        # Start tranining
        print((" Training%s %s " % (" and Validating" if self.ValDL else "", 
                                    colourText(str(Split + 1)))).center(WinCols, "*"))
        for Epoch in range(opt.epochs):
            time.sleep(0.5)  # To prevent possible deadlock during epoch transition
            
            if Epoch == opt.milestones and opt.pretrained and opt.freeze_weight == 3:
                # Starting from Milestones + 1, stop all layers" weight freezing
                self.Model = wightFrozen(self.Model, opt.freeze_weight)
                if opt.empty_cache:
                    # with torch.cuda.device("cuda:1"):
                    torch.cuda.empty_cache()
            
            self.TrainDL.dataset.callerInit()
            self.training(Epoch)
                
            if opt.lr_decay:
                if "cosine" in opt.schedular:
                    self.Scheduler.step(Epoch)
                else:
                    self.Scheduler.step()
                self.CurrentLrRate = self.Scheduler._last_lr[0] 
            else:
                self.CurrentLrRate = opt.lr
            self.LrList.append(self.CurrentLrRate)
                
            self.bestManager(opt, Epoch, self.Model)
        
        self.Model = reparameteriseModel(self.Model)
    
    def training(self, Epoch):
        self.Model.train()  # Normalization is different in trainning and evaluation
        self.TrainState.batchInit()
        for Batch in tqdm(self.TrainDL, ncols=WinCols - 9, colour="magenta"):
            self.Optim.zero_grad()  # Initialize gradient, preventing accumulation
            
            Batch = moveToDevice(Batch, self.opt.device)
            Img = Batch["image"]
            Label = Batch.get("label", None)
            
            PredLabel = self.Model(Img)  # prediction
            Loss, FinalPredLabel = mixCriterion(self.opt, self.LossFn, 
                                                Img, PredLabel, Label, Epoch)

            Loss.backward()  # backpropagation
            self.Optim.step()  # optimise model"s weight
            
            self.TrainState.batchUpdate(FinalPredLabel, Label, Loss, Epoch)
        
        self.TrainState.update()

        if self.ValDL:
            if "5nn" in self.opt.clsval_mode:
                self.generateFeatureBank()
                
            self.validating(Epoch)

    @torch.inference_mode()
    def validating(self, Epoch):
        self.ValState.batchInit()
        if Epoch >= self.opt.val_start_epoch:
            self.Model.eval()
            for Batch in tqdm(self.ValDL, ncols=WinCols - 9, colour="magenta"):
                Batch = moveToDevice(Batch, self.opt.device)
                Img = Batch["image"]
                Label = Batch["label"]
                # Label = torch.squeeze(Label)
                
                PredLabel = self.Model(Img)  # validation
                Loss, FinalPredLabel = mixCriterion(self.opt, self.LossFn, 
                                                    Img, PredLabel, Label, Epoch)
        
                self.ValState.batchUpdate(FinalPredLabel, Label, Loss, Epoch)
                
        self.ValState.update()
