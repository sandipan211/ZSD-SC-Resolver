# from sklearn import datasets
from tabnanny import verbose
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import randint
import torch 
from util import *
from generate import load_unseen_att, load_all_att,load_seen_att
from mmdetection.splits import get_unseen_class_labels , get_class_labels

colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45']
colors2 = ['#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3']

dataset = 'voc'

# labels_to_plot = np.arange(21)
labels_to_plot = np.array([17,  18,  19,  20])
# labels_to_plot = np.array([1,  2,  3,  4,  6,  8,  9, 10, 11, 12])

#CLASSES = np.concatenate((['background'], get_class_labels(dataset)))
CLASSES =  get_class_labels(dataset)
id_to_class = {idx: class_label for idx, class_label in enumerate(CLASSES)}
print(id_to_class)
def tsne_plot_feats(f_feat, f_labels, path_save , p):
    # import pdb; pdb.set_trace()
    tsne = TSNE(n_components=2, random_state=806, verbose=True,n_iter=1500 , perplexity=p )
    syn_feature = np.load(f_feat)
    syn_label = np.load(f_labels)
    #print(syn_label[:50])
    idx = np.where(np.isin(syn_label, labels_to_plot))[0]
    idx = np.random.permutation(idx)[0:2000]
    X_sub = syn_feature[idx]
    y_sub = syn_label[idx]
    #print(y_sub)
    # targets = np.unique(y)

    # colors = []
    for i in range(len(labels_to_plot)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    print(X_sub.shape, y_sub.shape, labels_to_plot.shape)
    
    X_2d = tsne.fit_transform(X_sub)
    print(X_2d)
    
    fig = plt.figure(figsize=(4,3))
    for i, c in zip(labels_to_plot, colors):
        plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c,s=8, label=id_to_class[i-1][:4])
    plt.legend()
    # plt.show()
    #fig.savefig(path_save , format="png")
    fig.savefig(path_save, format='pdf', dpi=600)
    #print(f"saved {path_save}")
    #return X_sub, y_sub

def plot_unseen(epochs=2):
    root='/workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6/syn_feat'
    #root='/workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_e3/syn_feat'
    # feat=np.load(f'{root}/syn_feats.npy')
    # label=np.load(f'{root}/syn_label.npy')
    pl=[20,30,40,45]
    for p in pl:
        for i in range(epochs):
            tsne_plot_feats(f'{root}/syn_feats_ep_{i}.npy', f'{root}/syn_label_ep_{i}.npy', f'{root}/tsne/voc_unseen_real_tsne_ep_{i}_p_{p}.pdf',p)
        
    #     print(f"{epoch:02}/{epochs} ")

    #     plt.close('all')
    #     # import pdb; pdb.set_trace()


plot_unseen(65)
# tsne_plot_feats('../data/voc/all_seen_feats.npy', '../data/voc/all_seen_labels.npy', 'plots/seen_real_tsne.png')
# def plot_seen():
# root = '/home/gdata/sushil/zsd/checkpoints/VOC_wgan_triplet_varMargin_seed806_cycCommented_try2/syn_feat'
# features = np.load(f"{root}/all_seen_feats.npy")
# labels = np.load(f"{root}/all_seen_labels.npy")

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots
# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1_merged/ /home/nasir/Downloads/plots_0.1_merged

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots/
