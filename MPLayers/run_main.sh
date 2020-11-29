# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
# module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
# module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
# module load xfce/4.12 opencv/3.4.3

python main.py --server="data61" \
               --img_name="tsukuba" \
               --context="TL" \
               --mode="TRWP" \
               --n_dir=4 \
               --n_iter=50 \
               --p_weight=20 \
               --n_disp=16 \
               --truncated=2 \
               --enable_saving_label
