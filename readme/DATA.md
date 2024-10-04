# Dataset preparation

## FIVES Dataset
The FIVES dataset comprises 800 high-resolution, multi-disease color fundus photographs with pixel-wise manual annotations. These annotations were standardized through a crowdsourcing effort among medical experts. Each image was evaluated for quality based on factors like illumination, color distortion, blur, and contrast.

- **Access the dataset**: Download from [FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1).

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- FIVES
    `-- |--- train
            |--- 1_A_train.jpg
            |--- 2_A_train.jpg
            |--- ...
            |--- 216_D_train.jpg
            |--- ...
        |--- test
            |--- 1_A_test.png
            |--- 2_A_train.png
            |--- ...
            |--- 166_N_test.png
            |--- ...
        |--- mask
            |--- 1_A_test.png
            |--- 1_A_train.png
            |--- ...
            |--- 216_D_train.png
            |--- ...
~~~


## ISIC 2018 Dataset
This dataset includes skin lesion images with corresponding binary masks.

- **Access the dataset**: Download from [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/#2018).

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- ISIC2018
    `-- |--- train
            |--- ISIC_0000000.jpg
            |--- ISIC_0000001.jpg
            |--- ...
            |--- ISIC_0013319.jpg
            |--- ...
        |--- val
            |--- ISIC_0012255.jpg
            |--- ISIC_0012346.jpg
            |--- ...
            |--- ISIC_0036291.jpg
            |--- ...
        |--- mask
            |--- ISIC_0000000.png
            |--- ISIC_0000001.png
            |--- ...
            |--- ISIC_0012346.png
            |--- ...
~~~


## PolypGen Dataset
PolypGen includes 8,037 frames from various hospitals, featuring both single and sequence frames with 3,762 positive and 4,275 negative samples.

- **Access the dataset**: Download from [PolypGen dataset](https://www.synapse.org/Synapse:syn26376615/wiki/613312).

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- PolypGen
    `-- |--- train
            |--- 4_endocv2021_positive_34.jpg
            |--- 4_endocv2021_positive_954.jpg
            |--- ...
            |--- EndoCV2021_001114.jpg
            |--- ...
        |--- val
            |--- EndoCV2021_C6_0100018.jpg
            |--- EndoCV2021_C6_0100013.jpg
            |--- ...
            |--- C3_EndoCV2021_00162.jpg
            |--- ...
        |--- mask
            |--- 4_endocv2021_positive_34.png
            |--- 4_endocv2021_positive_954.png
            |--- ...
            |--- EndoCV2021_001114.png
            |--- ...
~~~


## ATLAS Dataset
This dataset includes 90 T1 CE-MRI scans of the liver, segmented into liver and tumor masks.

- **Access the dataset**: Download from [A Tumor and Liver Automatic Segmentation](https://atlas-challenge.u-bourgogne.fr/).

- **Slice into 2D sequences**: Use [niftiFileExtractor.py](tools/niftiFileExtractor.py) to slice MRI scans into 2D sequences.

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- ATLAS
    `-- |--- train
            |--- im1.nii_0.jpg
            |--- im1.nii_12.jpg
            |--- ...
            |--- im38.nii_12.jpg
            |--- ...
        |--- val
            |--- im0.nii_0.jpg
            |--- im0.nii_4.jpg
            |--- ...
            |--- im59.nii_28.jpg
            |--- ...
        |--- mask
            |--- im0.nii_0.png
            |--- im0.nii_4.png
            |--- ...
            |--- im38.nii_12.png
            |--- ...
~~~


## KiTS23 Dataset
The KiTS23 challenge dataset for kidney and tumor segmentation.

- **Access the dataset**: Download from [KiTS23 Dataset](https://kits-challenge.org/kits23/).

- **Slice into 2D sequences**: Use [niftiFileExtractor.py](tools/niftiFileExtractor.py) to slice CT scans into 2D sequences.

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- KiTS23
    `-- |--- train
            |--- case_00000_132.jpg
            |--- case_00000_203.jpg
            |--- ...
            |--- case_00272_247.jpg
            |--- ...
        |--- val
            |--- case_00009_28.jpg
            |--- case_00009_37.jpg
            |--- ...
            |--- case_00514_441.jpg
            |--- ...
        |--- mask
            |--- case_00000_132.png
            |--- case_00000_203.png
            |--- ...
            |--- case_00514_441.png
            |--- ...
~~~

## TissueNet Dataset
The data of the TissueNet dataset is tissue excision data with corresponding binary masks. 

- **Access the dataset**: Download from [DeepCell Datasets](https://datasets.deepcell.org/data).

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SvANet_ROOT}
|-- dataset
`-- |-- TissueNet
    `-- |--- train
            |--- 20191121_MIBI_whole_cell_breast_train_346_345.jpg
            |--- 20191121_MIBI_whole_cell_breast_train_347_346.jpg
            |--- ...
            |--- 20200919_CODEX_CRC_train_947_946.jpg
            |--- ...
        |--- test
            |--- 20191121_MIBI_whole_cell_breast_test_184_183.jpg
            |--- 20191121_MIBI_whole_cell_breast_test_185_184.jpg
            |--- ...
            |--- 20200924_CyCIF_Lung_LN_test_1253_1252.jpg
            |--- ...
        |--- mask
            |--- 20191121_MIBI_whole_cell_breast_test_184_183.png
            |--- 20191121_MIBI_whole_cell_breast_test_185_184.png
            |--- ...
            |--- 20200924_CyCIF_Lung_LN_test_1253_1252.png
            |--- ...
~~~


## References

If you use the datasets and our data pre-processing codes, we kindly request that you consider citing our paper as follows:

~~~
@article{jin2022fives,
  title={{FIVES}: A fundus image dataset for artificial Intelligence based vessel segmentation},
  author={Jin, Kai and Huang, Xingru and Zhou, Jingxing and Li, Yunxiang and Yan, Yan and Sun, Yibao and Zhang, Qianni and Wang, Yaqi and Ye, Juan},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={475},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{codella2019skin,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration ({ISIC})},
  author={Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and Celebi, M Emre and Dusza, Stephen and Gutman, David and Helba, Brian and Kalloo, Aadi and Liopyris, Konstantinos and Marchetti, Michael and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}

@article{ali2023multi,
  title={A multi-centre polyp detection and segmentation dataset for generalisability assessment},
  author={Ali, Sharib and Jha, Debesh and Ghatwary, Noha and Realdon, Stefano and Cannizzaro, Renato and Salem, Osama E and Lamarque, Dominique and Daul, Christian and Riegler, Michael A and Anonsen, Kim V and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={75},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@article{quinton2023tumour,
  title={A Tumour and Liver Automatic Segmentation ({ATLAS}) Dataset on Contrast-Enhanced Magnetic Resonance Imaging for Hepatocellular Carcinoma},
  author={Quinton, F{\'e}lix and Popoff, Romain and Presles, Beno{\^\i}t and Leclerc, Sarah and Meriaudeau, Fabrice and Nodari, Guillaume and Lopez, Olivier and Pellegrinelli, Julie and Chevallier, Olivier and Ginhac, Dominique and others},
  journal={Data},
  volume={8},
  number={5},
  pages={79},
  year={2023},
  publisher={MDPI}
}

@misc{heller2023kits21,
      title={The {KiTS21} challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase {CT}}, 
      author={Nicholas Heller and Fabian Isensee and Dasha Trofimova and others},
      year={2023},
      eprint={2307.01984},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{greenwald2022whole,
  title={Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning},
  author={Greenwald, Noah F and Miller, Geneva and Moen, Erick and Kong, Alex and Kagel, Adam and Dougherty, Thomas and Fullaway, Christine Camacho and McIntosh, Brianna J and Leow, Ke Xuan and Schwartz, Morgan Sarah and others},
  journal={Nature Biotechnology},
  volume={40},
  number={4},
  pages={555--565},
  year={2022},
  publisher={Nature Publishing Group US New York}
}

@misc{dai2024svanet,
  title={Exploiting Scale-Variant Attention for Segmenting Small Medical Objects},
  author={Dai, Wei and Liu, Rui and Wu, Zixuan and Wu, Tianyi and Wang, Min and Zhou, Junxian and Yuan, Yixuan and Liu, Jun},
  year={2024},
  eprint={2407.07720},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2407.07720}, 
}
~~~
