from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att,load_seen_att
from mmdetection.splits import get_unseen_class_labels
from cls_models import ClsModel, ClsUnseen,Regressor
opt = parse_args()


try:
    os.makedirs(opt.outname)
except OSError:
    pass

try:
    os.makedirs(f'{opt.outname}/syn_feat')
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed) #added
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.backends.cudnn.deterministic=True #added
    torch.backends.cudnn.benchmark=False     #added

#cudnn.benchmark = True  #commented

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# unseen_attributes, unseen_att_labels = load_unseen_att(opt)
# attributes, _ = load_all_att(opt)
#loading unseen semantic embedings and labels for unseen class.
unseen_attributes, unseen_att_labels = load_unseen_att(opt)
#loading all the attributes
attributes, _ = load_all_att(opt)
#print(f"seen_attr {attributes.size() }")
#print(attributes)
#loading semantic embeddings for seen classes (++)
seen_attributes, seen_attr_labels = load_seen_att(opt)

# init classifier
trainCls = TrainCls(opt)

print('initializing GAN Trainer')


start_epoch = 0



seenDataset = FeaturesGAN(opt)
#trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')

trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels,seen_attributes,seen_attr_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')
if opt.netD and opt.netG:
    start_epoch = trainFGGAN.load_checkpoint()
    
for epoch in range(start_epoch, opt.nepoch):
    # features, labels = seenDataset.epochData(include_bg=False)
    features, labels = seenDataset.epochData(include_bg=True)
    # train GAN
    trainFGGAN(epoch, features, labels)
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)
    num_of_bg = opt.syn_num*2

    ####for storing syn features for each epoch #####
    np.save(f'{opt.outname}/syn_feat/syn_feats_ep_{epoch}.npy', syn_feature)
    np.save(f'{opt.outname}/syn_feat/syn_label_ep_{epoch}.npy', syn_label)
    ####################

    real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)

    # concatenate synthesized + real bg features
    syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
    syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))
    
    trainCls(syn_feature, syn_label, gan_epoch=epoch)

    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    classes = np.concatenate((['background'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        # print('*'*20)
        # print(f'best clasifier is at: {epoch}')
        # print('*'*20)
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')
