
import argparse
import os
import os.path as osp
import shutil
import tempfile
#from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.apis import get_root_logger, init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis.runner import copy_synthesised_weights 
from mmdet.core import eval_map, average_precision, get_cls_results
import numpy as np
from confusion_matrix_util import *


import matplotlib.pyplot as plt

#from util import *
import seaborn as sns
from splits import *

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


# def multi_gpu_test(model, data_loader, tmpdir=None):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)
#         results.append(result)

#         if rank == 0:
#             batch_size = data['img'][0].size(0)
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()

#     # collect results from all ranks
#     results = collect_results(results, len(dataset), tmpdir)

#     return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--syn_weights', help='the dir to save logs and models')

    parser.add_argument('--out', help='output result file')
    
    parser.add_argument('--zsd', action='store_true', help='test only for unseen classes')
    parser.add_argument('--gzsd', action='store_true', help='test for seen classes with unseen classes')

    parser.add_argument('--dataset', default='coco', help='coco, voc, imagenet')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)

    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def plot_confusion_matrix(c_mat, xtick_marks, ytick_marks, opt, dataset_name='coco'):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(c_mat, cmap=plt.cm.Blues)

    plt.xticks(np.arange(xtick_marks.shape[0]), xtick_marks, rotation=75)
    plt.yticks(np.arange(ytick_marks.shape[0]), ytick_marks)
    # import pdb; pdb.set_trace()

    for (i, j), z in np.ndenumerate(c_mat):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    fig = plt.gcf()
    fig.savefig(f'{dataset_name}_confusion_matrix.pdf', format='pdf', dpi=600)
    plt.close()

def mat_eval(result_file,opt, dataset, iou_thr=0.5, dataset_name='voc', split='16_4'):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    # print(len(gt_labels))
    # #print(len(gt_bboxes))
    # print(gt_labels)
    #print(gt_bboxes.dtype())
    # print(len(det_results))
    # print(det_results[0])
    # print(len(det_results[0]))
    # print(det_results[0][4])
    # print(det_results[0][4][0][4])
    ##### required gr truth format ####
    gr_th=[]
    for i in range(len(gt_bboxes)):
        cr_bbox=gt_bboxes[i]
        #print(cr_bbox)
        cr_label=gt_labels[i]
        #print(cr_label)
        if len(cr_bbox)>0:
            
            for j,k in enumerate(cr_bbox):
                temp_d=[]
                temp_d=np.append(temp_d,cr_label[j])
                temp_d=np.append(temp_d,k)
                temp_d_np=np.array(temp_d)
                gr_th.append(temp_d_np)
            

    #print(gr_th)
    gr_th=np.array(gr_th)
    ###############################
    #converting detection result to req format
    new_det_format=[]
    for i in range(len(det_results)): ## will go through each image
        curr_img=det_results[i]  
        #print(len(curr_img))
        for j in range(len(curr_img)):  ## for each image will go through all classes as index representing classes having all the bboxes
            temp_det=[]
            lab=j
            curr_det=curr_img[j]
            
            if(len(curr_det)):
                print(curr_det)
                print(lab)
                for k in curr_det:
                    temp_temp_det=k
                    temp_temp_det=np.append(temp_temp_det,lab)
                    
                    new_det_format.append(temp_temp_det)


                    #print(temp_temp_det)
                    # for l in range(len(temp_temp_det)):
                    #     #print(temp_temp_det[l])
                    #     temp_det.append(temp_temp_det[l])
                    #     #print(temp_det)

                    # temp_det.append(lab)
            
            # if len(temp_det):
            #     new_det_format.append(temp_det)
            #     temp_det.clear()

    #print(new_det_format)
    new_det_format=np.array(new_det_format)
    # #####################################
    conf_matrix=ConfusionMatrix(num_classes=21,CONF_THRESHOLD=0.3,IOU_THRESHOLD=iou_thr)
    conf_matrix.process_batch(new_det_format,gr_th)
    confusion_mat=conf_matrix.return_matrix()
    confusion_mat=np.array(confusion_mat)
    print(confusion_mat.shape)
    conf_matrix.print_matrix()
    print(confusion_mat.dtype)
    classes = np.concatenate((['background'], get_unseen_class_labels(dataset_name, split=split)))
    plot_confusion_matrix(confusion_mat, classes, classes, opt, dataset_name=dataset_name)





    # mean_ap, eval_results = eval_map(
    #     det_results,
    #     gt_bboxes,
    #     gt_labels,
    #     gt_ignore=gt_ignore,
    #     scale_ranges=None,
    #     iou_thr=iou_thr,
    #     dataset=dataset.CLASSES,
    #     dataset_name=dataset_name,
    #     print_summary=True,
    #     split=split)
   
    # return mean_ap, eval_results
    


def main():
    args = parse_args()
    logger = get_root_logger('INFO')

    for arg in vars(args): logger.info(f"######################  {arg}: {getattr(args, arg)}")
    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # import pdb; pdb.set_trace()

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.test_cfg.rcnn.zsd = args.zsd
    cfg.test_cfg.rcnn.gzsd = args.gzsd
    cfg.test_cfg.rcnn.score_thr = 0.05
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    if cfg.test_cfg.rcnn.gzsd and hasattr(dataset,'cat_ids'):
        dataset.cat_to_load = dataset.cat_ids
    
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    model.CLASSES = dataset.CLASSES
    logger.info(cfg.data.test.split)
    dataset_name = args.dataset
    if args.syn_weights:
        seen_bg_weight, seen_bg_bias = copy_synthesised_weights(model, args.syn_weights, dataset_name, split=cfg.data.test.split)
        model.bbox_head.seen_bg_weight = torch.from_numpy(seen_bg_weight).cuda()
        model.bbox_head.seen_bg_bias = torch.from_numpy(seen_bg_bias).cuda()

    

    print("for IOU=0.5 ::")
    mat_eval(args.out,args, dataset,iou_thr=0.5, dataset_name=dataset_name, split=cfg.data.test.split)
    #print(f'iou 0.5 mAP is :{mean_ap1}')

   

if __name__ == '__main__':
    main()

