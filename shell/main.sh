CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname PolypGen --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname FIVES --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname ISIC2018T1 --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname ATLAS --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname KiTS23 --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname TissueNet --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth