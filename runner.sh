cd /workspace/arijit_ug/sushil/zsd/
#ls
pwd
#cd mmdetection
#source ~/opt/conda/etc/profile.d/conda.sh
#conda init powershell
#conda activate zsd
#$conda list


#./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dir/coco2014/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth
 nvidia-smi
 nvidia-smi -L #prints summary of all gpus

 #cd mmdetection
  # ./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 1 --validate 

  #train faster-rcnn
 # python ./tools/train.py  configs/faster_rcnn_r101_fpn_1x.py 

#extract features (-step 2)
#cd mmdetection
#python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data --data_split train

#python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data --data_split test

#training regressor
#python train_regressor.py 

#training generator (step 3)
#./script/train_coco_generator_65_15.sh


#step -4 (evaluation step)
 #cd mmdetection

#evaluation on zsd
  #./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out coco_65_15_all_incl_cyc_loss_gzsd_result.pkl --gzsd --syn_weights ../checkpoints/coco_65_15_all_incl_cyc_loss/classifier_best_latest.pth


### coco data #######

#extract features (-step 2)
# cd mmdetection
# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data/coco2014 --data_split train

# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data/coco2014 --data_split test

# #training generator (step 3)
# cd ..
#./script/train_coco_generator_65_15.sh


# #step -4 (evaluation step)
# cd mmdetection

# # #evaluation on zsd
#  ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1/classifier_best_latest.pth

# #evaluation on gzsd
# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1_gzsd_result.pkl --gzsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_triplet_varMargin_try1/classifier_best_latest.pth
# nvidia-smi
#  cd ..
  #./script/train_coco_generator_65_15_1.sh
# # #step -4 (evaluation step)

 cd mmdetection

# # # # #evaluation on gzsd
   ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try6/classifier_best_latest.pth

   ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try7/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try7_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/ab_st_final/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try7/classifier_best_latest.pth

# # #evaluation on gzsd
#  ./tools/dist_test_1.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_original_nasir_try_4/coco_65_15_original_nasir_try_4_gzsd_result.pkl --gzsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_original_nasir_try_4/classifier_best_latest.pth

# # #show bounding boxes result
# # ###for coco dataset
# # ##showing outputcoco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try21
#  ./tools/dist_test_show_coco.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_original_nasir_try_4/classifier_best_latest.pth --out coco_detections.p

