from ..utils import CLASS_NAMES


def segmentationParamsInit(opt):
    TestMode = not opt.setname
    
    opt.mask_fill = 255
    opt.ignore_idx = 255

    if not opt.class_names:
        SetName = opt.setname.lower()
        opt.class_names = ["background"]
        if SetName:
            opt.class_names.extend(CLASS_NAMES[SetName.lower()])
            
    if 'encoder_decoder' not in opt.seg_model_name:
        opt.seg_head_name = '-'
        opt.model_name = opt.seg_model_name
    elif opt.fg_nostage5:
        opt.seg_head_name = 'fg'
        opt.fg_for_head = True
        
    
    # feature map guide
    if opt.seg_feature_guide:
        if not opt.fg_use_guide:
            opt.fg_svattn = 0
            opt.fg_vit = 0
            
    if not opt.saved_metric:
      opt.saved_metric = 'iou'
    
    # test mode
    if TestMode:
        if not SetName:
            opt.class_names.extend(CLASS_NAMES['spermhealth'])
            
        if any(m in opt.sup_method for m in ['common', 'smooth']):
            opt.setname = 'sample_seg'

        opt.epochs = 2
        opt.target_supplement = 0
        opt.target_exp = None
        opt.is_student = False
    elif opt.epochs < 100 and 'common' in opt.sup_method:
        opt.num_repeat = 1
    
    if not opt.seg_loss_name:
      opt.seg_loss_name = opt.loss_name
    
    return opt