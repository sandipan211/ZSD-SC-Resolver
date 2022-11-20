#added to use mmdet.apis present in zsd/mmdetection
# import sys
# sys.path.insert(1, '/home/gdata/sushil/zsd/mmdetection/')
# ##########
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import torch
from collections import OrderedDict
import pandas as pd 
import numpy as np 
#from constants import get_unseen_class_ids, get_seen_class_ids
import os
from mmdet.datasets import build_dataset
from mmdet.apis.runner import copy_synthesised_weights 
from mmcv import Config

config_file = 'configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py'
dir_name='VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6'
path='/workspace/arijit_ug/sushil/zsd'
syn_weights = '/workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6'
checkpoint_file = './work_dirs/voc/epoch_4.pth'
split='65_15'
#test_setting='zsd'
score_thr = 0.6
try:
    #os.makedirs('det_results/coco')
    os.makedirs(f'/workspace/arijit_ug/sushil/zsd/checkpoints/voc/{dir_name}/det_results/voc_zsd_{score_thr}')
    #os.makedirs(f'det_results/voc_{score_thr}')
except OSError:
    pass

# import pdb; pdb.set_trace()

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.test, {'test_mode': True})
model.CLASSES = dataset.CLASSES
# copy_syn_weights(syn_weights, model)
#copy_synthesised_weights(model, syn_weights, 'voc', split='16_4')
root = '/workspace/arijit_ug/sushil/zsd/data/pascal'

# df = pd.read_csv('../MSCOCO/validation_coco_unseen_all.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name}" for file_name in file_names]
# files_path = np.array(files_path)
# img_infos
# for idx, img in enumerate(files_path[:1000]):
# import pdb; pdb.set_trace()
import random
# color = "%06x" % random.randint(0, 0xFFFFFF)
# from splits import COCO_ALL_CLASSES
# color_map = {label: (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)) for label in COCO_ALL_CLASSES}
# det_results = mmcv.load('gen_coco_results.pkl')
# det_results = mmcv.load('coco_results.pkl')
det_results=mmcv.load(f'/workspace/arijit_ug/sushil/zsd/checkpoints/voc/{dir_name}/'+(dir_name)+'_zsd_result.pkl')
#VOC_wgan_seen_cyclicSeenUnseen_triplet_varMar_try1_zsd_result
#print(det_results[1])
# #print(det_results[-1])
# quit()
# gen_filenames = [
# 'COCO_val2014_000000008676.jpg',
# 'COCO_val2014_000000012827.jpg',  'COCO_val2014_000000056430.jpg',  'COCO_val2014_000000403817.jpg',
# 'COCO_val2014_000000483108.jpg',
# 'COCO_val2014_000000012085.jpg',  'COCO_val2014_000000027371.jpg',  'COCO_val2014_000000069411.jpg',
# 'COCO_val2014_000000428454.jpg',
# 'COCO_val2014_000000553721.jpg']

# zsd = [
#     'COCO_val2014_000000052066.jpg',  'COCO_val2014_000000058225.jpg' ,
#     'COCO_val2014_000000128644.jpg' , 'COCO_val2014_000000350073.jpg' ,'COCO_val2014_000000519299.jpg',
#     'COCO_val2014_000000054277.jpg',  'COCO_val2014_000000101088.jpg' , 'COCO_val2014_000000171058.jpg',  
#     'COCO_val2014_000000512455.jpg'  ,'COCO_val2014_000000572517.jpg',
# ]
start = 0
img_infos = dataset.img_infos
# [start:start+200]
# [:1000]
#print(img_infos)
#print(img_infos)

for idx, info in enumerate(img_infos):
    print(info['filename'])
    img = f"{root}/VOC2007/{info['filename']}"
    # if info['filename'] in zsd:
    #result = inference_detector(model, img)
    # result = det_results[start+idx]#inference_detector(model, img)
    # import pdb; pdb.set_trace()
    result=det_results[idx]
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
    out_file = f"{path}/checkpoints/voc/{dir_name}/det_results/voc_zsd_{score_thr}/{img.split('/')[-1]}"
    show_result(f"{img}", result, model.CLASSES, out_file=out_file,show=False, score_thr=score_thr, dataset='voc')
    #print(f"[{idx:03}/{len(img_infos)}]")
    
        
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