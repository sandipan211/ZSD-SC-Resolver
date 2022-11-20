from turtle import width
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import torch
from collections import OrderedDict
import pandas as pd 
import numpy as np 
from splits import get_unseen_class_ids, get_seen_class_ids
import os
from mmdet.datasets import build_dataset
from mmdet.apis.runner import copy_synthesised_weights 
from mmcv import Config

config_file = 'configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712_single_img.py'
syn_weights = './work_dirs/pascal_voc_t2/GAN_outputs/classifier_best.pth'
checkpoint_file = './work_dirs/pascal_voc_workdirs/epoch_4_old.pth'
score_thr = 0.4
try:
    os.makedirs(f'./work_dirs/pascal_voc_workdirs/det_results_single_img')
except OSError:
    pass

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.test)
model.CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person', 
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
]
# seen_bg_weight, seen_bg_bias = copy_synthesised_weights(model, syn_weights, 'voc', split='16_4')

root = '/workspace/arijit_pg/aman/data/custom'

start = 0
img_infos = dataset.img_infos
# [start:start+200]
# [:1000]
model.to('cuda:0')
# results = mmcv.load('./work_dirs/pascal_voc_workdirs/results_test_without_rpn_conf_cls.pkl')
results = mmcv.load('./work_dirs/pascal_voc_workdirs/single_img.pkl')


# for idx, info in enumerate(img_infos):
#     img = f"{root}/{info['filename']}"

#     # data_split = 'train' if 'train' in img_id else 'val'

#     # filename = f'Data/DET/{data_split}/{img_id}.JPEG'
#     # if info['filename'] in zsd:
#     # result = inference_detector(model, img)
#     result = results[idx]#inference_detector(model, img)
#     # import pdb; pdb.set_trace()
    # out_file = f"./work_dirs/pascal_voc_workdirs/det_results_single_img/{img.split('/')[-1]}"
#     show_result(f"{img}", result, model.CLASSES, out_file=out_file,show=False, score_thr=score_thr, dataset='voc')
#     print(f"[{idx:03}/{len(img_infos)}]")
    

for idx, info in enumerate(img_infos):
    # if info['filename'] != 'JPEGImages/000074.jpg':
        # continue
    img = f"{root}/{info['filename']}"
    print(info['filename'])
    # if info['filename'] in zsd:
    #result1 = inference_detector(model, img)
    result=results[idx]
    # print(result)
    fresult=[]
    # rth = 0
    # sa = []
    # for r in result:
    #     for r1 in r:
    #         sa.append(r1[-1])
    
    # sa.sort()
    # rth=sa[-4]
    # print(rth)

    for r in result:
    #     q=[]
    #     for r1 in r:
    #         if r1[-1] > rth:
    #             q.append(r1)
    #     print(r)
    #     print(q)
    #     print("############################")
        if len(r):
            r=r[0:1]
    #         q=np.array(q)
    #     else:
    #         q=r[0:0]
        fresult.append(r)
            # break
    print(fresult)
    # result = det_results[start+idx]#inference_detector(model, img)
    # import pdb; pdb.set_trace()
    out_file = f"./work_dirs/pascal_voc_workdirs/det_results_single_img/{img.split('/')[-1]}"
    show_result(f"{img}", fresult, model.CLASSES, out_file=out_file,show=False, score_thr=score_thr, dataset='voc')
   # print(f"[{idx:03}/{len(img_infos)}]")
    
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# copy_syn_weights(syn_weights, model)
# copy_synthesised_weights(model, syn_weights)

# root = '/raid/mun/codes/data/pascalv_voc/VOCdevkit/'
# df = pd.read_csv('../VOC/testval_voc07_unseen.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name[14:]}" for file_name in file_names]
# files_path = np.array(files_path)
# for idx, img in enumerate(files_path[:100]):

#     result = inference_detector(model, img)
#     out_file = f"det_results/voc/{img.split('/')[-1]}"
#     show_result(img, result, model.CLASSES, out_file=out_file,show=False, score_thr=0.3)
#     print(f"[{idx:03}/{len(files_path)}]")

# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --syn_weights /raid/mun/codes/zero_shot_det
# ection/cvpr18xian_pascal_voc/checkpoints/VOC/classifier_best.pth --out voc_detections.p
