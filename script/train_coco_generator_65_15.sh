
python trainer.py --manualSeed 806 \
--cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 55 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 128 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 50000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier mmdetection/work_dirs/coco2014/epoch_12.pth \
--class_embedding MSCOCO/fasttext.npy \
--dataroot /workspace/arijit_ug/sushil/zsd/data/coco \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 65_15 \
--lz_ratio 0.01 \
--outname /workspace/arijit_ug/sushil/zsd/checkpoints/coco_65_15_wgan_modeSeek_seen_cycSeenUnseen_tripletSeenUnseen_varMargin_try16 \
--regressor_lamda 0.01 \
--triplet_lamda 0.1 \
--pretrain_regressor MSCOCO/regressor_1.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
 
