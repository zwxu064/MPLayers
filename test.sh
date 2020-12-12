#!/bin/bash -l

# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
# module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
# module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
# module load xfce/4.12 opencv/3.4.3

python main.py \
--backbone "resnet101" \
--gpu-ids "0" \
--workers 4 \
--val-batch-size 12 \
--checkname "deeplab-resnet" \
--eval-interval 1 \
--dataset_root="datasets/pascal_scribble" \
--dataset "pascal" \
--save-interval 1 \
--rloss-scale 0.5 \
--output_directory "experiments/run/TRWP/seg" \
--checkpoint_dir "experiments/run/TRWP" \
--mpnet_mrf_mode "TRWP" \
--enable_mplayer_epoch 0 \
--crop-size 512 \
--base-size 512 \
--optimizer "SGD" \
--ft \
--resume "pretrained/TRWP/model.pth.tar" \
--resume_unary "pretrained/vanilla/model.pth.tar" \
--edge_mode "canny" \
--mpnet_scale_list 0.5 \
--enable_test \
