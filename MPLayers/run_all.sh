#!/bin/bash

# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
# module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
# module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
# module load xfce/4.12 opencv/3.4.3

img_names=("tsukuba" "teddy" "venus" "cones" "map" "000002_11" "000041_10" "000119_10" "delivery_area_1l" "facade_1s")
p_funcs=("TL" "TL" "TQ" "TL" "TL" "TL" "TL" "TL" "TL" "TL")
n_disps=(16 60 20 55 29 96 96 96 32 32)
truncs=(2 1 7 8 6 95 95 95 31 31)
p_weights=(20 10 50 10 4 10 10 10 10 10)
modes=("SGM" "ISGMR" "TRWP")
n_dirs=(4 8 16)
enable_min_a_dirs=(0 1)

for idx in "${!img_names[@]}"; do
  img_name=${img_names[idx]}
  p_func=${p_funcs[idx]}
  n_disp=${n_disps[idx]}
  trunc=${truncs[idx]}
  p_weight=${p_weights[idx]}
  for mode in "${modes[@]}"; do
    if [ ${mode} == "SGM" ]; then
      n_iter=2
    else
      n_iter=50
    fi

    for n_dir in ${n_dirs[@]}; do
      for enable_min_a_dir in ${enable_min_a_dirs[@]}; do
        echo ${img_name} ${p_func} ${p_weight} ${n_disp} ${trunc} ${mode} ${n_dir} ${n_iter} ${enable_min_a_dir}
        python main.py --server="data61" \
                       --img_name=${img_name} \
                       --context=${p_func} \
                       --mode=${mode} \
                       --n_dir=${n_dir} \
                       --n_iter=${n_iter} \
                       --enable_min_a_dir=${enable_min_a_dir} \
                       --p_weight=${p_weight} \
                       --n_disp=${n_disp} \
                       --truncated=${trunc}
      done
    done
  done
done
