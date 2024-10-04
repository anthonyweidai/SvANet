import math

from .segInit import segmentationParamsInit
from .utils import resizeShapeInit, initPathMode
from ..utils import (
    correctDatasetPath, correctWeightPath,
    getOnlyFolderNames, getFilePathsFromSubFolders, splitChecker
)


def paramsInit(opt):
    '''Init before single task init'''
    if opt.lincls:
        opt.sup_method = 'common'
    
    
    # single init
    opt = segmentationParamsInit(opt)
    SetName = opt.setname.lower()
  
  
    '''Init after single task init'''
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    
    # dataset
    ## the number of splits
    _, opt.num_split = splitChecker(opt.setname, RSplit=True)
    
    ## path
    opt.custom_dataset_img_path, opt.dataset_path = correctDatasetPath(opt.custom_dataset_img_path, opt.setname)
        
    ## resize shape
    opt = resizeShapeInit(opt)

    ## get image path mode
    if not opt.get_path_mode:
        opt.get_path_mode = initPathMode(opt.dataset_path, SetName)
        
    ## class name
    if 'traversal' in opt.class_names:
        opt.class_names = getOnlyFolderNames(opt.dataset_path + '/train')
        
    ## collate function name
    if not opt.collate_fn_name:
        opt.collate_fn_name = "default"
            

    # train
    ## model
    opt.model_name = opt.model_name.lower()
    
    if 'None' not in opt.save_point:
        opt.save_point = [int(i) for i in opt.save_point.split(',')]
        opt.save_point.sort()
    
    if not opt.num_repeat:
        opt.num_repeat = opt.num_split
    
    ## exp level
    if opt.exp_level == '': # and not opt.lincls
        if 'common' in opt.sup_method:
            # replace for sample dataset
            opt.exp_level = opt.setname.lower().replace('/', '_')
    
    ## start epoch
    if opt.val_start_epoch is None:
        opt.val_start_epoch = 0
  
    ## optimiser
    if 'cosine' in opt.schedular:
        opt.stop_station = opt.epochs
        if opt.lr is None:
            if 'mae' == opt.sup_method:
                # The actual `lr` is computed by the 
                # [linear scaling rule](https://arxiv.org/abs/1706.02677)
                opt.lr = 1.5e-4 * opt.batch_size / 256
            elif 'sgd' in opt.optim:
                opt.lr = 0
                opt.max_lr = 0.05 * opt.batch_size / 256
            else:
                opt.lr = 2e-4
        
        if opt.max_lr is None:
            opt.max_lr = opt.lr * 10
        
        if opt.warmup_init_lr is None:
            opt.warmup_init_lr = opt.lr
    else:
        if opt.lr is None:
            opt.lr = 0.002
    
    if not opt.lr:
        if 'sgd' in opt.optim:
            opt.lr = 1e-7
            opt.max_lr = opt.lr_factor * opt.batch_size / 256
        else:
            opt.lr = 2e-4
            opt.max_lr = 2e-3
    elif not opt.max_lr:
        opt.max_lr = opt.lr * 10
    
    if opt.milestones is None:
        opt.milestones = math.ceil(0.1 * opt.epochs)
  
    # if opt.is_student:
    #   opt.freeze_weight = 0 if not opt.is_distillation else 0 # 0, 1, 2, 3
    
    if 'adamw' in opt.optim:
        if opt.weight_decay is None:
            opt.weight_decay = 1.e-2
  
    ## pretrained weight
    opt.load_model_path = correctWeightPath(opt.load_model_path)
    
    if opt.weight_name:
        WeightPool = getFilePathsFromSubFolders(opt.load_model_path)
        opt.pretrained_weight = [WeightPath for WeightPath in WeightPool \
            if opt.weight_name in WeightPath][0]
    else:
        opt.pretrained_weight = None

    ## number of views
    if 'common' in opt.sup_method:
        opt.views = 1
    # number of classes
    if any(m in opt.sup_method for m in ['common', 'smooth']):
        opt.num_classes = len(opt.class_names)
        opt.seg_num_classes = opt.num_classes
        opt.det_num_classes = opt.num_classes
      
    if not opt.cls_num_classes:
        opt.cls_num_classes = opt.num_classes
    
    
    # metrics
    if 'common' not in opt.sup_method and not opt.selfsup_valid:
        opt.saved_metric = 'loss'
        
    if opt.sup_metrics or opt.cls_num_classes < 10:
        opt.topk = (1, )
        opt.sup_metrics = True
        
    opt.cls_loss_name = opt.loss_name
    
    return opt