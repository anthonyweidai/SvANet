# Exploiting Scale-Variant Attention for Segmenting Small Medical Objects
Welcome to the official implementation of ``Exploiting Scale-Variant Attention for Segmenting Small Medical Objects''. This repository offers a robust toolkit designed for advanced tasks in deep learning and computer vision, specifically tailored for semantic segmentation. It supports features such as training progress visualization, logging, and calculation of standard metrics.

[**Exploiting Scale-Variant Attention for Segmenting Small Medical Objects**](https://doi.org/10.1109/TNNLS.2025.3645355)  
Wei Dai, Rui Liu, Zixuan Wu, Tianyi Wu, Min Wang, Junxian Zhou, Yixuan Yuan, Jun Liu        
Accepted in IEEE Transactions on Neural Networks and Learning Systems, 2026. [[Paper](https://doi.org/10.1109/TNNLS.2025.3645355)][[arXiv](https://arxiv.org/pdf/2407.07720v4)]


![](./readme/architecture_animation.gif)


## Installation

To install the SvANet implementation, please follow the detailed instructions in [INSTALL.md](readme/INSTALL.md).

## Benchmark and Evaluation

Please refer to [DATA.md](readme/DATA.md) for guidelines on preparing the datasets for benchmarking and training.

To initiate the training and evaluation processes, utilize the configuration settings provided in the [main.sh](shell/main.sh) script. Prior to commencing the training, ensure that you have downloaded the pretrained model from [torchvision](https://download.pytorch.org/models/resnet50-11ad3fa6.pth).


### *Results for Datasets with Diverse Object Sizes*  

<p align="left"> <img src=readme/segmentation_all_sizes.jpg align="center" width="1080">

<!-- <p align="center"> <img src=readme/trends.jpg align="center" width="640"> -->

### *Results for the Dataset for Only Ultra-small Objects*  

<p align="left"> <img src=readme/segmentation_ultra_small.jpg align="center" width="540">

<!-- ### *Negative Case Analysis*

<p align="left"> <img src=readme/vis.jpg align="center" width="1080">

<p align="left"> <img src=readme/vis_sup1.jpg align="center" width="1080">

<p align="left"> <img src=readme/vis_sup2.jpg align="center" width="1080"> -->


### Ablation studies

For detailed settings of the ablation study and additional experiments, refer to refer to the scripts [ablation.sh](shell/ablation.sh) and [ablation_extra.sh](shell/ablation_extra.sh).

## Inference

To perform inference evaluation on various datasets, it is important to organize the data according to the guidelines provided in [DATA.md](readme/DATA.md). After completing the training process, refrain from making any modifications to the output folder. Instead, utilize [tinyObjectsValidation.py](tools/tinyObjectsValidation.py) for evaluation purposes. Please remember to update the "RootPath" variable in the script to match the location of your output folder.

## Citation

If you use this implementation in your research, please consider citing our paper as follows:

    @article{dai2026exploiting,
      title={Exploiting scale-variant attention for segmenting small medical objects},
      author={Dai, Wei and Liu, Rui and Wu, Zixuan and Wu, Tianyi and Wang, Min and Zhou, Junxian and Yuan, Yixuan and Liu, Jun},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      year={2026},
      publisher={IEEE}
    }
