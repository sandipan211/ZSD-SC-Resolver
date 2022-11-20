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
import seaborn as sns
colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45', '#f2ccff','#330066','#ff531a', '#cccc00', ' #3385ff']
#colors2 = ['#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3', '']

dataset = 'coco'

# labels_to_plot = np.arange(21)
labels_to_plot = np.array([ 5,  7, 13, 16, 22, 29, 30, 32, 43, 49, 53, 62, 65, 71, 79])
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
    
    fig = plt.figure(figsize=(8,7))
    for i, c in zip(labels_to_plot, colors):
        plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c,s=8, label=id_to_class[i-1][:4])
    plt.legend()
    # plt.show()
    fig.savefig(path_save , format="png")
    #print(f"saved {path_save}")
    #return X_sub, y_sub

def plot_unseen(epochs):
    root='/workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try3/syn_feat'
    #root='/workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_e3/syn_feat'
    # feat=np.load(f'{root}/syn_feats.npy')
    # label=np.load(f'{root}/syn_label.npy')
    pl=[20,30,40]
    for p in pl:
        for i in range(epochs):
            tsne_plot_feats(f'{root}/syn_feats_ep_{i}.npy', f'{root}/syn_label_ep_{i}.npy', f'{root}/tsne/coco_unseen_real_tsne_ep_{i}_p_{p}.png',p)
    
    # tsne = TSNE(n_components = 2, random_state = 0,verbose=True)
    # # tsne_data = tsne.fit_transform(feat)
    # # tsne_data = np.vstack((tsne_data.T, label)).T
    # # tsne_df = pd.DataFrame(data = tsne_data,
    # # columns =("Dim_1", "Dim_2", "label"))
    #  # Ploting the result of tsne
    # X_2d = tsne.fit_transform(feat)
    # fig = plt.figure(figsize=(6, 5))
    # for i, c1, c2 in zip(labels_to_plot, colors, colors2):
    #     indx = np.where(label == i)[0]
    #     plt.scatter(X_2d[indx[indx<2000],   0], X_2d[indx[indx<2000], 1], c=c1, label=f"s_{id_to_class[i][:5]}")
    #     #plt.scatter(X_2d[indx[indx>=2000], 0], X_2d[indx[indx>=2000], 1], c=c2, label=f"r_{id_to_class[i][:5]}")
    # plt.legend()
    # fig.savefig(f'{root}/tsne_unseen.png')
        
 
    # real_f, real_l = tsne_plot_feats(f'{root}/syn_feats.npy', f'{root}/syn_label.npy', f'{root}/unseen_real_tsne.png')
    #real_f, real_l = tsne_plot_feats('../data/voc/all_val_plot_feats.npy', '../data/voc/all_val_plot_labels.npy', 'plots_0.1/unseen_real_tsne_bg_0.1.png')
    # print(f"len of real feats: {len(real_f)}")
    # for epoch in range(0, epochs):
    #     f_feat = f'{root}/syn_feats.npy'
    #     f_labels = f'{root}/syn_label.npy'
    #     path_save = f'{root}/{epoch}_unseen.png'
    #     syn_f, syn_l = tsne_plot_feats(f_feat, f_labels, path_save)
    #     print(f"len of syn feats: {len(syn_f)}")
        
    #     # # merge and plot
    #     # feats_all = np.concatenate((syn_f, real_f))
    #     # label_all = np.concatenate((syn_l, real_l))
    #     tsne = TSNE(n_components=2, random_state=0, verbose=True)
    #     print(f"len of all feats: {len(feats_all)}")

    #     # X_2d = tsne.fit_transform(feats_all)
    #     X_2d = tsne.fit_transform(syn_f)

    #     fig = plt.figure(figsize=(6, 5))
    #     # for i, c1, c2 in zip(labels_to_plot, colors, colors2): indx = np.where(label_all == i)[0]; plt.scatter(X_2d[indx[indx<5000], 0], X_2d[indx[indx<5000], 1], c=c1, label=f"s_{id_to_class[i][:3]}");plt.scatter(X_2d[indx[indx>=5000], 0], X_2d[indx[indx>=5000], 1], c=c2, label=f"r_{id_to_class[i][:3]}")
    #     # for i, c1, c2 in zip(labels_to_plot, colors, colors2):
    #     #     indx = np.where(label_all == i)[0]
    #     #     plt.scatter(X_2d[indx[indx<2000],   0], X_2d[indx[indx<2000], 1], c=c1, label=f"s_{id_to_class[i][:5]}")
    #     #     plt.scatter(X_2d[indx[indx>=2000], 0], X_2d[indx[indx>=2000], 1], c=c2, label=f"r_{id_to_class[i][:5]}")
    #     # plt.legend()

    #     for i, c1, c2 in zip(labels_to_plot, colors, colors2):
    #         indx = np.where(syn_l == i)[0]
    #         plt.scatter(X_2d[indx[indx<2000],   0], X_2d[indx[indx<2000], 1], c=c1, label=f"s_{id_to_class[i][:5]}")
    #         plt.scatter(X_2d[indx[indx>=2000], 0], X_2d[indx[indx>=2000], 1], c=c2, label=f"r_{id_to_class[i][:5]}")
    #     plt.legend()
    #     fig.savefig(f'{root}/{epoch}_unseen.png')
        
    #     print(f"{epoch:02}/{epochs} ")

    #     plt.close('all')
    #     # import pdb; pdb.set_trace()



plot_unseen(55)
# tsne_plot_feats('../data/voc/all_seen_feats.npy', '../data/voc/all_seen_labels.npy', 'plots/seen_real_tsne.png')
# def plot_seen():
# root = '/home/gdata/sushil/zsd/checkpoints/VOC_wgan_triplet_varMargin_seed806_cycCommented_try2/syn_feat'
# features = np.load(f"{root}/all_seen_feats.npy")
# labels = np.load(f"{root}/all_seen_labels.npy")

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots
# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1_merged/ /home/nasir/Downloads/plots_0.1_merged

# rsync -avzzp jr1@172.31.20.60:/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/plots_0.1* /home/nasir/Downloads/dgx_plots/
