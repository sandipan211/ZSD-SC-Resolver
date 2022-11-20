cd /workspace/arijit_ug/sushil/zsd/
#ls
pwd
#cd mmdetection
#source ~/opt/conda/etc/profile.d/conda.sh
#conda init powershell
#conda activate zsd
#$conda list

#pip list
#conda list
#whoami
#vim /opt/conda/lib/python3.7/site-packages/torch/serialization.py
#python setup.py develop
#./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dir/coco2014/epoch_12.pth 8 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best_137.pth
 nvidia-smi
 nvidia-smi -L #prints summary of all gpus

#  cd mmdetection
#   # ./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 1 --validate 

#   #train faster-rcnn
# python ./tools/train.py  configs/faster_rcnn_r101_fpn_1x.py 

#extract features (-step 2)
#cd mmdetection
#python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data --data_split train

#python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/coco2014/epoch_12.pth --save_dir /workspace/arijit_ug/sushil/zsd/data --data_split test

#training regressor
#python train_regressor.py

#training generator (step 3)
 #./script/train_coco_generator_65_15_1.sh

#training unseen_classifier
#    python train_unseen_classifier.py


 
#./script/train_coco_generator_65_15_1.sh
#step -4 (evaluation step)
# cd mmdetection

# # #evaluation on zsd
#  ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11_zsd_result.pkl --zsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11/classifier_best_latest.pth

# #evaluation on gzsd
# ./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py ./work_dirs/coco2014/epoch_12.pth 1 --dataset coco --out /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11_gzsd_result.pkl --gzsd --syn_weights /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_triplet_varMargin_try11/classifier_best_latest.pth

#############pascal dataset #################3

# #extract features (-step 2)
# cd mmdetection
# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes seen --load_from ./work_dirs/voc/epoch_4.pth --save_dir ../data/pascal/feat/ --data_split train
# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes unseen --load_from ./work_dirs/voc/epoch_4.pth --save_dir ../data/pascal/feat/ --data_split test


##cyclic regressor training
#python train_regressor_voc.py
# #train gan
# # cd ..

#  ./script/train_voc_generator.sh


# # #python train_regressor_voc.py
# # #step -4 (evaluation step)
  cd mmdetection


# # #evaluation on zsd
#  ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/voc/epoch_4.pth 1 --dataset voc --out ../checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry_zsd_result.pkl --zsd --syn_weights  /workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry/classifier_best_latest.pth

# # #evaluation on gzsd
# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/voc/epoch_4.pth 1 --dataset voc --out ../checkpoints/voc/VOC_wgan_seen_cycSeenUnseen_triplet_varMar_try5/VOC_wgan_seen_cycSeenUnseen_triplet_varMar_try5_gzsd_2.pkl --gzsd --syn_weights  /workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cycSeenUnseen_triplet_varMar_try5/classifier_best_latest.pth
# # #evaluation on zsd
  ./tools/dist_test_2.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/voc/epoch_4.pth 1 --dataset voc --out ../checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry_zsd_result.pkl --zsd --syn_weights  /workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_retry/classifier_best_latest.pth
