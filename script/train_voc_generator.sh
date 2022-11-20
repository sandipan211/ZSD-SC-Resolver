python trainer.py --manualSeed 806 \
--cls_weight 0.0001 --cls_weight_unseen 0 --nclass_all 21 --syn_num 500 --val_every 1 \
--cuda --netG_name MLP_G \
--batch_size 64 \
--netD_name MLP_D --nepoch 65 --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 \
--critic_iter 5 \
--dataset voc  --nz 300 --attSize 300 --resSize 1024 \
--lr 0.00001 \
--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
--dataroot /workspace/arijit_ug/sushil/zsd/data/pascal/feat \
--pretrain_classifier /workspace/arijit_ug/sushil/zsd/mmdetection/work_dirs/voc/epoch_4.pth \
--class_embedding /workspace/arijit_ug/sushil/zsd/VOC/fasttext_synonym.npy \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--lz_ratio 0.0001 \
--outname /workspace/arijit_ug/sushil/zsd/checkpoints/voc/VOC_wgan_seen_cyclicSeenUnseen_tripletSeenUnseen_varMar_try6_marginMat \
--regressor_lamda 0.001 \
--triplet_lamda 0.01 \
--pretrain_regressor /workspace/arijit_ug/sushil/zsd/VOC/voc_regressor_s100.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \