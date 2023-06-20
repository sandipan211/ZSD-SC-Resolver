[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resolving-semantic-confusions-for-improved-1/generalized-zero-shot-object-detection-on-ms)](https://paperswithcode.com/sota/generalized-zero-shot-object-detection-on-ms?p=resolving-semantic-confusions-for-improved-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resolving-semantic-confusions-for-improved-1/zero-shot-object-detection-on-ms-coco)](https://paperswithcode.com/sota/zero-shot-object-detection-on-ms-coco?p=resolving-semantic-confusions-for-improved-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resolving-semantic-confusions-for-improved-1/zero-shot-object-detection-on-pascal-voc-07)](https://paperswithcode.com/sota/zero-shot-object-detection-on-pascal-voc-07?p=resolving-semantic-confusions-for-improved-1)

# Resolving Semantic Confusions for Improved Zero-Shot Detection (BMVC Oral Presentation, 2022)

##  :eyeglasses: At a glance
This repository contains the official PyTorch implementation of our [BMVC 2022](www.bmvc2022.org) paper : [Resolving Semantic Confusions for Improved Zero-Shot Detection](https://bmvc2022.mpi-inf.mpg.de/0347.pdf), a work done by Sandipan Sarma, Sushil Kumar and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). 


- Supervised deep learning-based object detection models like Faster-RCNN and YOLO have seen tremendous success in the last decade or so, but are limited by the availability of large-scale annotated datasets, failure to recognize the changing object appearances over time, and ability to detect unseen objects.

- **Zero-shot detection (ZSD)** is a challenging task where we aim to recognize and localize objects simultaneously, **even when our model has not been trained with visual samples of a few target (“unseen”) classes**. This is achieved via knowledge transfer from the seen to unseen classes using semantics (attributes) of the object classes as a bridge.

- **Semantic confusion**: Knowledge transfer in existing ZSD models is not discriminative enough to differentiate between objects with similar semantics, e.g. *car* and *train*. 

- We propose a **generative approach and introduced triplet loss** during feature generation to account for inter-class dissimilarity.

- Moreover, we show that **maintaining cyclic consistency** between the generated visual features and their class semantics is helpful for improving the quality of the generated features.

- **Addressed problems** such as high false positive rate and misclassification of localized objects by resolving semantic confusion, and **comprehensively beat the state-of-the-art methods**.


<p align="center">
  <img alt="method" src="https://user-images.githubusercontent.com/30040548/202925339-e408b9bf-7cac-4c39-af27-b4a636aeb14d.jpg" width="700">
  <br>
    <em>The primary novelty of our model lies in the incorporation of triplet loss based on visual features, assisted by a cyclic-consistency loss</em>
</p>

## :newspaper: News
Uploaded [instructions](https://github.com/sandipan211/ZSD-SC-Resolver/issues/13#issuecomment-1598510805) for applying our method on custom datasets.


# :video_camera: Video
This paper was [presented](https://youtu.be/9mIEhM8ksK0) at the **BMVC Orals, 2022**.


# :bullettrain_side: Training the model

## 1. :office: Creating the work environment
Our code is based on PyTorch and has been implemented using an NVIDIA V100 32 GB DGX Station, with [mmdetection](https://github.com/open-mmlab/mmdetection) as the base framework for object detection, which contains a Faster-RCNN implementation. Install Anaconda/Miniconda on your system and create a conda environment using the following command:

```bash
conda env create -f zsd_environment.yml
```

Once set up, activate the environment and do the following:
```bash
cd ./mmdetection/

# install mmdetection and bind it to your project
python setup.py develop
```

Following commands are being shown for MSCOCO dataset. For PASCAL-VOC dataset, make the appropriate changes to the command line arguments and run the appropriate scripts.

## 2. :hourglass_flowing_sand: Train Faster-RCNN detector on *seen* data
All the configurations regarding training and testing pipelines are stored in a configuration file. To access it and make changes in it, find the file using:

```bash
cd ./mmdetection/configs/faster_rcnn_r101_fpn_1x.py
```

In zero-shot detection, the object categories in a dataset are split into two sets - *seen* and *unseen*. Such sets are defined in previous works for both MSCOCO [[1]](#1) and PASCAL-VOC [[2]](#2) datasets. The splits can be found in ```splits.py```.

To train the Faster-RCNN on seen data, run:
```bash
cd ./mmdetection
./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 1 --validate 
```
For reproducibility, it is recommended to use the pre-trained model given below in this repository. It is important to create a directory named ```work_dirs``` inside ```mmdetection``` folder, where there should be separate directories for MSCOCO and PASCAL-VOC, inside which the weights of the trained Faster-RCNN should be stored. For our pre-trained models, we name them as ```epoch_12.pth``` and ```epoch_4.pth``` after training Faster-RCNN on seen data of MSCOCO and PASCAL-VOC datasets respectively.

The pre-trained weights of Faster-RCNN are stored with the ResNet-101 (backbone CNN) being pre-trained only after removing the overlapping classes from ImageNet [[3]](#3). This pre-trained ResNet is given [here](https://drive.google.com/file/d/1wAgWbceKwS6c_zjZ3KzkDm7SNoKQOynJ/view?usp=share_link), and weights of Faster-RCNN are uploaded both for [PASCAL-VOC](https://mega.nz/file/rIc2Qawa#kjZjqHYtIX6RTpIy9vmxIPZzeU-6v5Rw7Ea6NRylTnU) and [MSCOCO](https://mega.nz/file/TZ81kSpA#LX97kMk0SlOQE2FwBymr1kk1pRPZxzxVwrxDzLAXaDA).

## 3. :outbox_tray: Extract object features 
Inside the ```data``` folder, MSCOCO and PASCAL-VOC image datasets should be stored in appropriate formats, before running the following:

```bash
cd ./mmdetection
python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from ./work_dirs/coco2014/epoch_12.pth --save_dir ./data --data_split train
```

## 4. :left_right_arrow: Training a visual-semantic mapper
Train a visual-semantic mapper using the *seen* data to learn a function mapping visual-space to semantic space. This trained mapper would be used in the next step while computing cyclic-consistency loss, improving feature-synthesis quality of GAN. Run:
```
python train_regressor.py 
```
Weights will be saved in the appropriate paths. For VOC, run ```train_regressor_voc.py```


## 5. :factory: Train the generative model using extracted features
Extracted seen-class object features constitute the *real data distribution*, using which a **Conditional Wasserstein GAN** is trained, with class-semantics of seen/unseen classes acting as the *conditional variables*. During GAN training, triplet loss is computed based on the synthesized object features, enforcing inter-class dissimilarity learning. Moreover, a cyclic-consistency between the synthesized features and their class semantics is computed, encourgaing the GAN to generate visual features that correspond well to their own semantics. For training the GAN, run the script:
```bash
./script/train_coco_generator_65_15.sh
```

## 6. :mag: Evaluation
```bash
cd mmdetection

#evaluation on zsd
./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/classifier_best_latest.pth
```

**NOTE:** Change ```--zsd``` flag to ```---gzsd``` for evaluation in the **generalized ZSD setting**. Change directory names accordingly. The classifier weights required in the evaluation step are given for [VOC](https://github.com/sandipan211/ZSD-SC-Resolver/blob/main/VOC/classifier_best_latest.pth) and [MSCOCO](https://github.com/sandipan211/ZSD-SC-Resolver/blob/main/MSCOCO/classifier_best_latest.pth).

## 7. :trophy: Results

- mAP for ZSD on MS-COCO

    |  **Method** | **ZSD (mAP in %)** |
    |:-----------:|:------------------:|
    |      PL     |        12.40       |
    |     BLC     |        14.70       |
    |   ACS-ZSD   |        15.34       |
    |    SUZOD    |        17.30       |
    |    ZSDTR    |        13.20       |
    | ContrastZSD |        18.60       |
    |   **Ours**  |      **20.10**     |
    
- Recall@100 for ZSD on MS-COCO

    |  **Method** | **ZSD (Recall@100 in %)** |
    |:-----------:|:-------------------------:|
    |      PL     |           37.72           |
    |     BLC     |           54.68           |
    |   ACS-ZSD   |           47.83           |
    |    SUZOD    |           61.40           |
    |    ZSDTR    |           60.30           |
    | ContrastZSD |           59.50           |
    |   **Ours**  |         **65.10**         |
    
    
- mAP for GZSD on MS-COCO

    |  **Method** | **Seen (mAP in %)** | **Unseen (mAP in %)** | **Harmonic Mean (mAP in %)** |
    |:-----------:|:-------------------:|:---------------------:|:----------------------------:|
    |      PL     |        34.07        |         12.40         |             18.18            |
    |     BLC     |        36.00        |         13.10         |             19.20            |
    |   ACS-ZSD   |          -          |           -           |               -              |
    |    SUZOD    |        37.40        |         17.30         |             23.65            |
    |    ZSDTR    |        40.55        |         13.22         |             20.16            |
    | ContrastZSD |        40.20        |         16.50         |             23.40            |
    |   **Ours**  |        37.40        |       **20.10**       |           **26.15**          |
    
    
- Recall@100 for GZSD on MS-COCO

    |  **Method** | **Seen (Recall@100 in %)** | **Unseen (Recall@100 in %)** | **Harmonic Mean (Recall@100 in %)** |
    |:-----------:|:--------------------------:|:----------------------------:|:-----------------------------------:|
    |      PL     |            36.38           |             37.16            |                36.76                |
    |     BLC     |            56.39           |             51.65            |                53.92                |
    |   ACS-ZSD   |              -             |               -              |                  -                  |
    |    SUZOD    |            58.60           |             60.80            |                59.67                |
    |    ZSDTR    |            69.12           |             59.45            |                61.12                |
    | ContrastZSD |            62.90           |             58.60            |                60.70                |
    |   **Ours**  |            58.60           |           **64.00**          |              **61.18**              |
    
- Results for PASCAL-VOC

    <p align="center">
      <img width="883" alt="pascal results" src="https://user-images.githubusercontent.com/30040548/234193175-72b233e7-ff0e-48b0-bb12-3661452089bb.png">
      <br>
    </p>

Log files are also uploaded for [ZSD](https://github.com/sandipan211/ZSD-SC-Resolver/blob/main/results/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try3_zsd_result_3dig.log) and [GZSD](https://github.com/sandipan211/ZSD-SC-Resolver/blob/main/results/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try3_gzsd_result.log).


# :gift: Citation
If you use our work for your research, kindly star :star: our repository and consider citing our work using the following BibTex:
```
@inproceedings{Sarma_2022_BMVC,
author    = {Sandipan Sarma and SUSHIL KUMAR and Arijit Sur},
title     = {Resolving Semantic Confusions for Improved Zero-Shot Detection},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0347.pdf}
}
```


# :scroll: References
<a id="1">[1]</a> 
Shafin Rahman, Salman Khan, and Nick Barnes. Polarity loss for zero-shot object
detection. arXiv preprint arXiv:1811.08982, 2018.

<a id="2">[2]</a> 
Berkan Demirel, Ramazan Gokberk Cinbis, and Nazli Ikizler-Cinbis. Zero-shot object
detection by hybrid region embedding. In BMVC, 2018.

<a id="3">[3]</a> 
Yongqin Xian, Christoph H Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. IEEE transactions on pattern analysis and machine intelligence, 41(9):2251–2265, 2018.
