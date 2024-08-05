# Exploiting Scale-Variant Attention for Segmenting Small Medical Objects
Welcome to the official implementation of ``Exploiting Scale-Variant Attention for Segmenting Small Medical Objects''. This repository offers a robust toolkit designed for advanced tasks in deep learning and computer vision, specifically tailored for semantic segmentation. It supports features such as training progress visualization, logging, and calculation of standard metrics.

[**Exploiting Scale-Variant Attention for Segmenting Small Medical Objects**](https://arxiv.org/pdf/2407.07720v1)  
Wei Dai, Rui Liu, Zixuan Wu, Tianyi Wu, Min Wang, Junxian Zhou, Yixuan Yuan, Jun Liu        
Under review by a peer-reviewed journal 2024. [[arXiv](https://arxiv.org/pdf/2407.07720v1)] <!-- [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024002740?dgcid=rss_sd_all)] -->


![](./readme/architecture_animation.gif)


## Installation

To install the SvANet implementation, please follow the detailed instructions in [INSTALL.md](readme/INSTALL.md).

## Benchmark and Evaluation

Please refer to [DATA.md](readme/DATA.md) for guidelines on preparing the datasets for benchmarking and training.

Execute the training and evaluation processes using the configuration settings in [main.sh](shell/main.sh) script. Before training, please download the pretrained model in [torchvision](https://download.pytorch.org/models/resnet50-11ad3fa6.pth).

<!-- 
### *Results for Datasets with Diverse Object Sizes*  

<p align="left"> <img src=readme/segmentation_all_sizes.jpg align="center" width="1080">

<p align="center"> <img src=readme/trends.jpg align="center" width="640">

### *Results for the Dataset for Only Ultra-small Objects*  

<p align="left"> <img src=readme/segmentation_ultra_small.jpg align="center" width="540">

### *Negative Case Analysis*

<p align="left"> <img src=readme/vis.jpg align="center" width="1080">

<p align="left"> <img src=readme/vis_sup1.jpg align="center" width="1080">

<p align="left"> <img src=readme/vis_sup2.jpg align="center" width="1080"> -->

### Ablation studies

For detailed settings of the ablation study and additional experiments, refer to refer to the scripts [ablation.sh](shell/ablation.sh) and [ablation_extra.sh](shell/ablation_extra.sh).

## Inference


## Citation

If you use this implementation in your research, please consider citing our paper as follows:

    @misc{dai2024svanet,
      title={Exploiting Scale-Variant Attention for Segmenting Small Medical Objects},
      author={Dai, Wei and Liu, Rui and Wu, Zixuan and Wu, Tianyi and Wang, Min and Zhou, Junxian and Yuan, Yixuan and Liu, Jun},
      year={2024},
      eprint={2407.07720},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.07720}, 
    }