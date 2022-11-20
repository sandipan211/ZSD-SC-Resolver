# :computer: BMVC 2022 : Resolving Semantic Confusions for Improved Zero-Shot Detection

##  :eyeglasses: At a glance
This repository contains the official PyTorch implementation of our [BMVC 2022](www.bmvc2022.org) paper : Resolving Semantic Confusions for Improved Zero-Shot Detection, a work done by Sandipan Sarma, Sushil Kumar and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). 


- Supervised deep learning-based object detection models like Faster-RCNN and YOLO have seen tremendous success in the last decade or so, but are limited by the availability of large-scale annotated datasets, failure to recognize the changing object appearances over time, and ability to detect unseen objects.

- **Zero-shot detection (ZSD)** is a challenging task where we aim to recognize and localize objects simultaneously, **even when our model has not been trained with visual samples of a few target (“unseen”) classes**. This is achieved via knowledge transfer from the seen to unseen classes using semantics (attributes) of the object classes as a bridge.

- **Semantic confusion**: Knowledge transfer in existing ZSD models is not discriminative enough to differentiate between objects with similar semantics, e.g. *car* and *train*. 

- We propose a **generative approach and introduced triplet loss** during feature generation to account for inter-class dissimilarity.

- Moreover, we show that **maintaining cyclic consistency** between the generated visual features and their class semantics is helpful for improving the quality of the generated features.

- **Addressed problems** such as high false positive rate and misclassification of localized objects by resolving semantic confusion, and **comprehensively beat the state-of-the-art methods**.


<p align="center">
  <img alt="method" src="https://user-images.githubusercontent.com/30040548/202925042-8dc9f465-df7d-4303-95a9-24661915f69b.png" width="700">
  <br>
    <em>The primary novelty of our model lies in the incorporation of triplet loss based on visual features, assisted by a cyclic-consistency loss</em>
</p>

# :bullettrain_side: Training the model

## 1. :office: Creating the work environment
Our code is based on PyTorch and has been implemented using an NVIDIA DGX Station, with [mmdetection](https://github.com/open-mmlab/mmdetection) as the base framework for object detection, which contains a Faster-RCNN implementation. Install Anaconda/Miniconda on your system and create a conda environment using the following command:

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

The pre-trained weights of Faster-RCNN are stored with the ResNet-101 (backbone CNN) being pre-trained only after removing the overlapping classes from ImageNet [[3]](#3). This pre-trained ResNet is given [here](https://drive.google.com/file/d/1wAgWbceKwS6c_zjZ3KzkDm7SNoKQOynJ/view?usp=share_link). 

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


## 4. :factory: Train the generative model using extracted features
Extracted seen-class object features constitute the *real data distribution*, using which a **Conditional Wasserstein GAN** is trained, with class-semantics of seen/unseen classes acting as the *conditional variables*. During GAN training, triplet loss is computed based on the synthesized object features, enforcing inter-class dissimilarity learning. Moreover, a cyclic-consistency between the synthesized features and their class semantics is computed, encourgaing the GAN to generate visual features that correspond well to their own semantics. For training the GAN, run the script:
```bash
./script/train_coco_generator_65_15.sh
```

## 5. :mag: Evaluation
```bash
cd mmdetection

#evaluation on zsd
./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/classifier_best_latest.pth
```

**NOTE:** Change ```--zsd``` flag to ```---gzsd``` for evaluation in the **generalized ZSD setting**. Change directory names accordingly.




# :scroll: References
<a id="1">[1]</a> 
Shafin Rahman, Salman Khan, and Nick Barnes. Polarity loss for zero-shot object
detection. arXiv preprint arXiv:1811.08982, 2018.

<a id="2">[2]</a> 
Berkan Demirel, Ramazan Gokberk Cinbis, and Nazli Ikizler-Cinbis. Zero-shot object
detection by hybrid region embedding. In BMVC, 2018.

<a id="3">[3]</a> 
Yongqin Xian, Christoph H Lampert, Bernt Schiele, and Zeynep Akata. Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. IEEE transactions on pattern analysis and machine intelligence, 41(9):2251–2265, 2018.