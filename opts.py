import argparse


class Opts(object):
    def __init__(self):
        self.Parser = argparse.ArgumentParser()
        # basic experiment setting
        self.Parser.add_argument("--task", default="segmentation",
                                help="segmentation")
        self.Parser.add_argument(
            "--sup_method", type=str, default="common",
            help="common")
        
        
        # system
        self.Parser.add_argument("--gpus", default="0", 
                                help="-1 for CPU, use comma for multiple gpus")
        self.Parser.add_argument("--num_workers", type=int, default=None,
                                help="dataloader threads. 0 for single-thread")
        self.Parser.add_argument("--pin_memory", action="store_false",
                                help="Use pin_memory in dataloader threads.")
        self.Parser.add_argument("--empty_cache", action="store_true",
                                help="Releases all unoccupied cached memory in milestone")


        # dataset
        self.Parser.add_argument("--custom_dataset_img_path", default="./dataset/",
                                help="custom dataset")
        self.Parser.add_argument("--setname", default="", # "sample/split",
                                help="see lib/dataset/ for available datasets")
        self.Parser.add_argument("--reg_by_name", action="store_true",
                                help="Get dataset by its name but not supervision method?")
        self.Parser.add_argument("--num_split", type=int, default=1,
                                help="The number of cross-validation folders") # k-folds validation
        self.Parser.add_argument("--get_path_mode", type=int, default=None,
                                help="The mode of calling images from their path")
        self.Parser.add_argument("--num_repeat", type=int, default=None,
                                help="The number of repeated split") # k-folds validation
        self.Parser.add_argument("--resize_shape", type=int, default=None,
                                help="The resize scale of transforms")
        self.Parser.add_argument("--pre_resize", action="store_true",
                                help="Is the dataset aleady pre-processed?")
        self.Parser.add_argument("--class_names", type=list, default=None,
                                help="The list of class name")
        self.Parser.add_argument("--collate_fn_name", type=str, default=None,
                                help="The collate function name")
        self.Parser.add_argument("--drop_last", action="store_true",
                                help="Drop last in data loader, due to batch normlisation, \
                                the value will be different for the same dataset for \
                                    different batch size")
        self.Parser.add_argument("--loader_shuffle", action="store_false",
                                help="Shuffle data order while loading?")
        self.Parser.add_argument("--random_aug_order", action="store_true", 
                            help="Use random order for augmentation methods?")
        self.Parser.add_argument("--mask_ratio", type=float, default=None,
                                help="The mask ratio of masking image in data loading.")
        self.Parser.add_argument("--use_meanstd", action="store_false",
                                help="Use mean std to process the image RGB channels?")


        # task
        self.Parser.add_argument("--num_classes", type=int, default=None,
                                help="The number of classes")
        self.Parser.add_argument("--init_weight", action="store_false", 
                                help="Apply weight initialisation?")
        ## classification
        self.Parser.add_argument("--model_name", default="resnet50",
                                help="model architecture")
        self.Parser.add_argument("--cls_num_classes", type=int, default=None,
                                help="The number of classes")
        self.Parser.add_argument("--dist_mode", action="store_true", 
                                help="Model disillation")
        self.Parser.add_argument("--is_student", action="store_true", 
                                help="Student network for model disillation")
        self.Parser.add_argument("--auglikeclr", action="store_true", 
                                help="Use contrastive learning augmentation in classification")
        self.Parser.add_argument("--reparamed", action="store_true", 
                                help="Model structure in inferencing for reparameterisation")


        ## segmentation
        self.Parser.add_argument("--seg_num_classes", type=int, default=None,
                                help="The number of classes")
        self.Parser.add_argument("--seg_model_name", type=str, default="encoder_decoder",
                                help="Segmentation models\" name")
        self.Parser.add_argument("--seg_head_name", type=str, default="deeplabv3",
                                help="Segmentation heads\" name")
        self.Parser.add_argument("--use_sep_conv", action="store_true",
                                help="True | False") # deeplabv3
        # self.Parser.add_argument("--output_stride", type=int, default=None,
        #                          help="Stride of the output layer")
        self.Parser.add_argument("--use_aux_head", action="store_true",
                                 help="Do you use auxiliary head during segmentation?")
        ### feature guidance module
        self.Parser.add_argument("--seg_feature_guide", type=int, default=0,
                                help="0 not used | 1 standard | 2 lightweight")
        self.Parser.add_argument("--fg_start_stage", type=int, default=1,
                                help="The starting stage of feature map guide")
        self.Parser.add_argument("--fg_resize_stage", type=int, default=0,
                                help=
                                "The stage for resizing feature maps \
                                before concatenation 0 | -1, \
                                0 has better result, but may be slower in some heads")
        self.Parser.add_argument("--fg_bottle", type=int, default=1,
                                help="0 None | 1 default")
        self.Parser.add_argument("--fg_bottle_se", type=int, default=2,
                                help="0 none | 1 squeeze and excitation | 2 monte carlo attention")
        self.Parser.add_argument("--fg_use_guide", action="store_false",
                                help="Use feature guide (cslayer)?")
        self.Parser.add_argument("--fg_svattn", type=int, default=1,
                                help="Use the scale variant attention in the segmentation head? \
                                    -3 vanilla max attention | -2 vanilla avg attention | -1 final avg attention \
                                    | 0 none attention | 1 scale variant attention")
        self.Parser.add_argument("--fg_svattn_divisor", type=int, default=4,
                                help="The divisor for pool size")
        self.Parser.add_argument("--moc_order", action="store_false",
                                help="The order of moc attention in fg bottleneck")
        self.Parser.add_argument("--fg_vit", type=int, default=1,
                                help="Use vit in FG bottleneck")
        self.Parser.add_argument("--fg_vit_se", type=int, default=0,
                                help="Use selayer in segmentation FG bottleneck vit?")
        self.Parser.add_argument("--fg_vit_all", action="store_false",
                                help="Use vit in all sub-branch")
        self.Parser.add_argument("--fg_link", type=int, default=2,
                                help="0 None | 1 link head | 2 catlink head")
        self.Parser.add_argument("--fg_link_vit", type=int, default=1,
                                help="Use vit in link module?")
        self.Parser.add_argument("--fg_link_vit_se", type=int, default=0,
                                help="Use selayer in link module vit?")
        ### tested modules
        self.Parser.add_argument("--fg_for_head", action="store_true",
                                help="Use the feature map guide the segmentation head? \
                                    Only useful in fg_nostage5")
        ### low-efficacy modules
        self.Parser.add_argument("--fg_seghead_vit", type=int, default=0,
                                help="Use vit after segmentation head?")
        self.Parser.add_argument("--fg_seghead_vit_se", type=int, default=2,
                                help="Use selayer in segmentation head vit?")
        self.Parser.add_argument("--fg_cat_shuffle", action="store_true",
                                help="shuffle the concatenated feature maps?")
        ### low-performance modules
        self.Parser.add_argument("--fg_nostage5", action="store_true",
                                help="Remove stage-5 layers in encoder?")


        # train
        self.Parser.add_argument("--epochs", type=int, default=70,
                                help="total training epochs.")
        self.Parser.add_argument("--batch_size", type=int, default=4,
                                help="batch size") # 8 multiple
        self.Parser.add_argument("--max_train_iters", type=int, default=None,
                                help="the total training iterations.")
        self.Parser.add_argument("--stop_station", type=int, default=100,
                                help="The stop epoch NO. for efficient training")
        self.Parser.add_argument("--exp_base", type=str, default="exp",
                                help="The base folder of exp output") 
        self.Parser.add_argument("--exp_level", type=str, default="",
                                help="The focus comparison name for a new level of folder") 
        self.Parser.add_argument("--target_exp", type=int, default=None,
                                help="The target exp folder location for supplement")
        self.Parser.add_argument("--target_supplement", type=int, default=0,
                                help="start training round, starting from 0")
        self.Parser.add_argument("--dest_path", type=str, default=None,
                                help="The full target exp folder path")
        self.Parser.add_argument("--save_point", type=str, default="80", # 30,60,90
                                help="when to save the model to disk.")
        self.Parser.add_argument("--clsval_mode", type=str, default="linear",
                                help="linear | 5nn") # only in classification task
        self.Parser.add_argument("--cpu_5nn", action="store_false",
                                help="use 5nn in CPU to save memory") # only in classification task
        self.Parser.add_argument("--knn_k", type=int, default=5,
                                help="The numer of nearest neighbor in kNN monitor")
        self.Parser.add_argument("--val_start_epoch", type=int, default=None,
                                help="The validation starting point")
        self.Parser.add_argument("--tsne_mode", action="store_true",
                                help="Extract only feature for t-SNE graph?")


        # optim
        self.Parser.add_argument("--optim", default="adamw")
        self.Parser.add_argument("--lr", type=float, default=None, 
                                 help="The minimum learning rate")
        self.Parser.add_argument("--warmup_init_lr", type=float, default=None, 
                                help="warming up learning rate for schedular")
        self.Parser.add_argument("--max_lr", type=float, default=None, 
                                help="maximum learning rate for schedular")
        self.Parser.add_argument("--lr_factor", type=float, default=0.05, 
                                help="The divior of divident max_lr and quotient lr, for sgd")
        self.Parser.add_argument("--lr_decay", action="store_false",
                                help="learning rate decay")
        self.Parser.add_argument("--schedular", type=str, default="mycosine",
                                help="mycosine | base")
        self.Parser.add_argument("--weight_decay", type=float, default=0,
                                help="mycosine")
        self.Parser.add_argument("--milestones", type=int, default=None,
                                help="milestones for learning rate decay")
        ## adam
        self.Parser.add_argument("--beta1", type=float, default=0.9)
        self.Parser.add_argument("--beta2", type=float, default=0.999)
        self.Parser.add_argument("--amsgrad", action="store_true",
                                help="whether to use the AMSGrad variant of \
                                this algorithm from the paper \
                                    `On the Convergence of Adam and Beyond`_(default: False)")

        # transfer
        self.Parser.add_argument("--lincls", action="store_true",
                                help="Use linear classification protocol?")
        self.Parser.add_argument("--load_model_path", default="./savemodel/",
                                help="path to pretrained model")
        self.Parser.add_argument("--pretrained", type=int, default=0, 
                                help="Use transfer learning?")
        self.Parser.add_argument("--freeze_weight", type=int, default=0, 
                                help="Freezing for partially transfer learning warm-up")
        self.Parser.add_argument("--weight_name", type=str, default=None, 
                                help="for partially transfer learning warm-up")


        # metrics, loss
        self.Parser.add_argument("--saved_metric", type=str, default=None,
                                help="loss | accuracy | iou | ap")
        self.Parser.add_argument("--loss_coeff", type=list, default=None) # [0.3, 0.3, 1]
        self.Parser.add_argument("--loss_name", type=str, default="cross_entropy")
        self.Parser.add_argument("--cls_loss_name", type=str, default=None)
        self.Parser.add_argument("--seg_loss_name", type=str, default=None)

        self.Parser.add_argument("--class_weights", action="store_true",
                                help="Use class sensitive loss?")
        self.Parser.add_argument("--label_smoothing", type=float, default=0, # 0.1
                                help="The label smoothing params for cross entropy.")
        self.Parser.add_argument("--loss_reduction", type=str, default="mean",
                                help="mean | sum")
        self.Parser.add_argument("--aux_weight", type=float, default=0.4,
                                help="The loss weight of segmentation auxiliary branch.")
        self.Parser.add_argument("--metric_type", type=str, default="micro",
                                help="macro | micro")
        self.Parser.add_argument("--ignore_idx", type=int, default=-100,
                                help="ignore background in segmentation loss calculation")
        ## multibox
        self.Parser.add_argument("--max_monitor_iter", type=int, default=-1,
                                help="the maximum monitor iteration for multibox loss")
        self.Parser.add_argument("--update_wt_freq", type=int, default=None,
                                help="frequency to update wegiht")
        ## ntxent
        self.Parser.add_argument("--temperature", type=float, default=0.5,
                                help="temperature for ntxent")
        ## mae
        self.Parser.add_argument("--norm_pix_loss", action="store_false",
                                help="the target for better representation learning")
        
        ## supplementary metrics for classification
        self.Parser.add_argument("--sup_metrics", action="store_true",
                                help="For small dataset. Supplementary metrics for classifcation, \
                                including recall, precision, specificity, F1Score")
        self.Parser.add_argument("--topk", type=tuple, default=(1, 5),
                                help="For small dataset. Supplementary metrics for classifcation, \
                                including recall, precision, specificity, F1Score")

        ## rotation
        self.Parser.add_argument("--rot_degree", type=int, default=None,
                                help="The degree of angle of rotation self-sup") # 0 < degree <= 120
        self.Parser.add_argument("--angle_shaking", action="store_true",
                                help="Random angle degrees?")
        self.Parser.add_argument("--range_angle_shaking", action="store_true",
                                help="Random angle degrees in range each iter?")
        self.Parser.add_argument("--angle_shaking_divisor", type=int, default=None,
                                help="What is proportion of divisor?")
        self.Parser.add_argument("--region_rot", action="store_false",
                                help="Rotate only center circular region? True | False")


        # module
        ## mocattn
        self.Parser.add_argument("--moc_pool_res", '--list', nargs='+',
                                 type=int, default=[1, 2, 3],
                                 help="The kept pooled tensor sizes.")
        
        self.Parser.add_argument("--crop_mode", type=int, default=None,
                                help="0 no crop (original rot) | 1 rectangle crop | 2 circle crop \
                                    | 3 circle crop and rotation around original image point  ")
        self.Parser.add_argument("--crop_ratio", type=float, default=None,
                                help="How large ratio of region do you want to crop? (0, 1)")
        self.Parser.add_argument("--mincrop_ratio", type=float, default=0.6,
                                help="How minimum ratio of region do you want to crop? (0, 1)")
        self.Parser.add_argument("--random_pos", action="store_false",
                                help="Do you use random position of rotated region?")
        self.Parser.add_argument("--centre_rand", action="store_true",
                                help="Random position in the centre or uniform pos?")
        self.Parser.add_argument("--crop_resize", action="store_false", # If false, could add additional len info (include swim mode)
                                help="Do you want to resize image before croping?")
        
    def parse(self, args=""):
        return (self.Parser.parse_args() if args == "" 
                else self.Parser.parse_args(args))