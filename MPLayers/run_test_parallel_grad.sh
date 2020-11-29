#!/bin/bash
#SBATCH --job-name=psm
#SBATCH --time=2:00:00
#SBATCH --mem=15GB
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL

# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
# module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
# module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
# module load xfce/4.12 opencv/3.4.3

python test_parallel_grad.py
