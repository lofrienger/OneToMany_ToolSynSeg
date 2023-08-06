# OneToMany_ToolSynSeg
Codes for the paper 

**Generalizing Surgical Instruments Segmentation to Unseen Domains with One-to-Many Synthesis**.

An Wang*, Mobarakol Islam*, Mengya Xu, and Hongliang Ren**

*: First author; **: Corresponding author

Accepted by [**IROS 2023**](https://ieee-iros.org/).

> This work proposes a highly efficient data-centric framework for surgical instrument synthesis and segmentation. Specifically, for data synthesizing, we only leverage one background tissue image and a few foreground instrument images to construct the source image pools through broad augmentations. One tissue image and two or three instrument images are randomly sampled from the background and foreground image pools. Then we compose them with three blending techniques to form the final synthetic surgical scene image. With the generated synthetic datasets, we incorporate the hybrid usage of advanced training-time augmentations when training the segmentation model. The generalization ability across multiple real test datasets gets steadily improved. Moreover, when adding only 10\% real data, we can achieve much better average segmentation results, indicating the practical significance of our framework in synthetic-real joint training.

![overall_framework](overall_framework.jpeg)

**Fig. 1 Overview of the proposed surgical instruments synthesis and segmentation framework.** The framework consists of three consecutive steps: Source Image Pools Construction, Blending-based Image Composition, and Hybrid Training-time Augmentation. 
We first extensively augment the foreground and background seed images and construct the image pools. Then Alpha, Gaussian, or Laplacian Blending is adopted to compose the randomly sampled images from the image pools. Finally, the composited images are further transformed with hybrid in-training augmentation, including Element-wise Patch-based Mixing, Coarsely Dropout, and Chained Augmentation Mixing.

  - [Environments](#environments)
  - [Dataset Generation](#dataset-generation)
    - [Augmentation of background tissue image](#augmentation-of-background-tissue-image)
    - [Augmentation of foreground instruments images](#augmentation-of-foreground-instruments-images)
    - [Blending of composited training samples](#blending-of-composited-training-samples)
  - [Instruments Segmentation](#instruments-segmentation)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

***News*** - One colab [demo](demo/colab/demo.ipynb) was added!

## Environments

* Python=3.8
* Pytorch=1.10
* torchvision=0.11.2
* cuda=11.3
* imgaug=0.4.0
* albumentations=1.1.0
* comet_ml=3.2.0 (used for experiments logging, remove where necessary if you don't need)
* Other commonly seen dependencies can be installed via pip/conda.

## Dataset Generation

### Augmentation of background tissue image
In ./data_gen/srcdata/bg/, the source background tissue image is provided. Adapt `aug_bg.ipynb` to generate augmented background images.

### Augmentation of foreground instruments images
In ./data_gen/srcdata/fg/, different types of instruments with of image-mask pairs are provided (Note: there are 2 versions of Bipolar Forceps in Endovis-2018 dataset.). Adapt `aug_fg.ipynb` to generate augmented foreground images.

### Blending of composited training samples
In ./data_gen/syndata/, adapt `blend.ipynb` to generate the blended images used for training.

## Instruments Segmentation

To evaluate the quality of the generated synthetic dataset, binary instrument segmentation is adopted. 

Example training commands:

1. train on Endo18 dataset
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Endo18_train --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup
```

2. train on Synthetic dataset
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Syn-S3-F1F2 --blend_mode alpha --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup
```

3. joint train: 90% Syn-S3-F1F2 + 10% Endo18
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Joint_Syn-S3-F1F2 --blend_mode alpha --real_ratio 0.1 --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup
```

4. joint train: Syn-S3-F1F2 + 10% Endo18
```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --train_dataset Syn-S3-F1F2_inc_Real --blend_mode alpha --inc_ratio 0.1 --augmix I --augmix_level 2 --coarsedropout hole14_w13_h13_p5 --cutmix_collate FastCollateMixup
```


## Acknowledgements

Part of the codes are adapted from [robot-surgery-segmentation](https://github.com/ternaus/robot-surgery-segmentation) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

## Citation
```
@inproceedings{wang2023generalizing,
    title={Generalizing Surgical Instruments Segmentation to Unseen Domains with One-to-Many Synthesis},
    author={An Wang and Mobarakol Islam and Mengya Xu and Hongliang Ren},
    booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2023},
    organization={IEEE}
}
```
The code structure mainly from our previous code [Single_SurgicalScene_For_Segmentation](https://github.com/lofrienger/Single_SurgicalScene_For_Segmentation) for MICCAI-2022 paper [**Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need**](https://arxiv.org/abs/2206.11804).
```
@inproceedings{wang2022rethinking,
  title={Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need},
  author={Wang, An and Islam, Mobarakol and Xu, Mengya and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={355--364},
  year={2022},
  organization={Springer}
}
```
