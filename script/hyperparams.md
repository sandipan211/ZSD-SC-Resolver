# :bangbang: Hyperparameters to set for implementing the [paper](https://bmvc2022.mpi-inf.mpg.de/0347.pdf)

This file explains the hyperparameters to set in the scripts (definitions [here](https://github.com/sandipan211/ZSD-SC-Resolver/blob/d20cf2a3cb1cb654e0a6cb1aaced468cf072e06c/arguments.py)) 

- manualSeed : random seed for code reproducibility
- cls_weight : to decide how much weight to give to the classification loss, i.e., $\alpha_2$ in Eq. 6 of the paper  (NOTE: cls_weight_unseen is not used in our work)
- nclass_all : total number of classes (including background)
- syn_num : number of features to synthesize per object class. An ablation study of its effect on model performance can be seen in Figure 2(b) of the paper
- val_every : validation rule
- cuda : setting the GPU flag

  ## Regarding the WGAN network
- netG_name : generator network
- netD_name : discriminator network
- nepoch : total epochs to train
- ngh : linear layer neurons for generator network (ablations can be done to check for possible effects of network strength by varying the number of neurons)
- ndh : linear layer neurons for discriminator network (ablations can be done to check for possible effects of network strength by varying the number of neurons)
- lambda1 : for the gradient penalty trick of Wasserstein GANs (default value taken from the original [NIPS 2017 paper](https://dl.acm.org/doi/pdf/10.5555/3295222.3295327)
- critic_iter : number of critic (discriminator) iterations per generator iteration for  Wasserstein GANs (default value taken from the original [NIPS 2017 paper](https://dl.acm.org/doi/pdf/10.5555/3295222.3295327)
- nz : dimension of the random noise vector
- gan_epoch_budget : random pick subset of features to train GAN
- lr : learning rate for training GAN
- lr_step : to decay the learning rate every lr_step epochs
- lz_ratio : mode seeking loss weight, i.e., $\alpha_3$ in our paper
- regressor_lamda : $\alpha_4$ in our paper
- triplet_lamda : $\alpha_5$ in our paper
- pretrain_regressor : The MLP for maintaining cyclic consistency

## Others
- dataset : name of dataset
- batch_size : batch size for training
- attSize : dimension of semantic vector
- resSize : dimension of visual embedding
- lr_cls : learning rate for training classifier
- pretrain_classifier : output of step 2
- class_embedding : class-attribute matrix
- dataroot : root directory for data
- testsplit/trainsplit : features extracted for test/train sets
- classes_split : seen/unseen split of classes
- outname : output results here
