# dataset: SpermHealth, PolypGen, ATLAS, ISIC2018T1, FIVES, KiTS23, and TissueNet
# module: SE in bottle
## full paper
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --moc_pool_res 1 2 3 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation
### SE in bottle
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 1 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 3 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 4 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation
### Moc Pool Res
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --moc_pool_res 1 2 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --moc_pool_res 2 3 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --moc_pool_res 1 2 3 4 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation
### batch size
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 1 --epochs 100 --save_point None --batch_size 2 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 2 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation
### fg link
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 0 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 1 --fg_link_vit 1 --fg_link_vit_se 0 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation
### No stage 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --task segmentation --gpus 3 --setname SpermHealth --num_repeat 3 --num_workers 2 --epochs 100 --save_point None --batch_size 4 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adamw --weight_decay 0.03 --seg_head_name deeplabv3 --seg_feature_guide 2 --fg_start_stage 1 --fg_bottle 1 --fg_bottle_se 2 --moc_pool_res 1 2 3 --fg_svattn 1 --fg_vit 1 --fg_vit_se 0 --fg_link 2 --fg_link_vit 1 --fg_link_vit_se 0 --fg_nostage5 --pretrained 2 --freeze_weight 3 --weight_name resnet50-2.pth --exp_base exp_full_ablation