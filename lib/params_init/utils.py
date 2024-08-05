from ..utils import RESDICT, PRE_RESIZE, getOnlyFolderNames


def initPathMode(DatasetPath, SetName):
    # for param initialisation and norm counter
    FolderNames = getOnlyFolderNames(DatasetPath)
    if "split" in FolderNames or "split1" in FolderNames: # or use splitChecker in lib.utils
        PathMode = 1
    else:
        PathMode = 4
    return PathMode


def resizeShapeInit(opt):
    ## resize shape
    SetName = opt.setname.lower()
    if "resize" in SetName:
        DefaultResizeRes = RESDICT.get(SetName.replace("_resize", ""), RESDICT["default"][opt.task])
    else:
        DefaultResizeRes = RESDICT.get(SetName, RESDICT["default"][opt.task])
     
    if opt.resize_shape is None:
        opt.resize_shape = DefaultResizeRes
    
    if opt.resize_shape == DefaultResizeRes:
        opt.pre_resize = PRE_RESIZE.get(SetName, PRE_RESIZE["default"])
    else:
        opt.pre_resize = False
    return opt


def segOptInit(opt):
    import torch
    from ..utils import deviceInit, correctDatasetPath, CLASS_NAMES
    # device
    _, DeviceStr = deviceInit(opt)
    opt.device = torch.device(DeviceStr)
    
    # task and data loader
    opt.task = "segmentation"
    opt.pretrained = 4
    opt.batch_size = 1
    opt.loader_shuffle = False
    opt = resizeShapeInit(opt)
    
    opt.custom_dataset_img_path, opt.dataset_path = correctDatasetPath(opt.custom_dataset_img_path, opt.setname)
    if not opt.get_path_mode:
        opt.get_path_mode = initPathMode(opt.dataset_path, opt.setname.lower())
    
    opt.class_names = ["background"]
    opt.class_names.extend(CLASS_NAMES.get(opt.setname.lower(), CLASS_NAMES["pascalanimal"]))
    opt.num_classes = opt.cls_num_classes = opt.seg_num_classes = len(opt.class_names)
    
    return opt


def initSegModelbyCSV(opt, FolderPath, ValSplit):
    import pandas as pd
    from ..utils import getBestModelPath
    # init model by reading information of input_param.csv
    opt.pretrained_weight = getBestModelPath(FolderPath + "/model/", ValSplit)
    
    Df = pd.read_csv(FolderPath + "/input_param.csv", header=0)
    opt.model_name = Df["Model"][0]
    opt.seg_model_name = Df["SegModel"][0]
    opt.seg_head_name = Df["SegHead"][0]
    opt.seg_feature_guide = Df["FeatureGuide"][0]
    if opt.seg_feature_guide:
        FGListValues = [
            "fg_use_guide", "fg_start_stage", "fg_resize_stage", 
            "fg_for_head", "fg_svattn", "fg_svattn_divisor", "fg_bottle", "fg_bottle_se", 
            "fg_vit", "fg_vit_se", "fg_nostage5", "fg_link", 
            "fg_link_vit", "fg_link_vit_se", 
        ]
        FGListName = [
            "FGUseGuide", "FGStartStage", "FGResizeStage", 
            "FGForHead", "SvAttn", "SvAttnDivisor", "FGBottle", "FGBottleSE", 
            "FGViT", "FGViTSE", "FGNoStage5", "FGLink", 
            "FGLinkViT", "FGViTLinkSE",
        ]

        for n1, n2 in zip(FGListValues, FGListName):
            ValColumn = Df.get(n2, None)
            if ValColumn is not None:
                Val = ValColumn[0]
                setattr(opt, n1, Val)
            else:
                print("ValColumn: %s is None" % n2)
    return opt
