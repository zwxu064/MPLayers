#!/bin/bash -l

mode="TRWP"
enable_run_script=true
enable_test=true
dataset_root="datasets/pascal_scribble"

if ${enable_test}; then
  bash_name="experiments/scripts/${mode}_test.sh"
  log_name="experiments/logs/${mode}_test.txt"
else
  bash_name="experiments/scripts/${mode}_train.sh"
  log_name="experiments/logs/${mode}_train.txt"
fi

echo -e "#!/bin/bash -l

#SBATCH --job-name=${mode}
#SBATCH --time=2:00:00
#SBATCH --mem=17G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=${log_name}

module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
module load xfce/4.12 opencv/3.4.3

python main.py \\
--backbone \"resnet101\" \\
--gpu-ids \"0\" \\
--workers 4 \\
--val-batch-size 12 \\
--checkname \"deeplab-resnet\" \\
--eval-interval 1 \\
--dataset_root=\"${dataset_root}\" \\
--dataset \"pascal\" \\
--save-interval 1 \\
--rloss-scale 0.5 \\
--output_directory \"experiments/run/${mode}/seg\" \\
--checkpoint_dir \"experiments/run/${mode}\" \\
--mpnet_mrf_mode \"${mode}\" \\
--enable_mplayer_epoch 0 \\
--crop-size 512 \\
--base-size 512 \\
--optimizer \"SGD\" \\
--ft \\
--resume \"pretrained/${mode}/model.pth.tar\" \\
--resume_unary \"pretrained/vanilla/model.pth.tar\" \\
--edge_mode \"canny\" \\
--mpnet_scale_list 0.5 \\" > ${bash_name}

if ${enable_test}; then
  echo -e "--enable_test \\" >> ${bash_name}
fi

eval "chmod 755 ${bash_name}"

# Run
if ${enable_run_script}; then
    eval "sbatch ${bash_name}"
fi
