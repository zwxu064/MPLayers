#!/bin/bash

img_names=("tsukuba" "teddy" "venus" "cones" "map" "000002_11" "000041_10" "000119_10" "delivery_area_1l" "facade_1s" "penguin" "house")
p_funcs=("TL" "TL" "TQ" "TL" "TL" "TL" "TL" "TL" "TL" "TL" "TQ" "TQ")
n_disps=(16 60 20 55 29 96 96 96 32 32 256 256)
truncs=(2 1 7 8 6 95 95 95 31 31 200 -1)
p_weights=(20 10 50 10 4 10 10 10 10 10 25 5)
modes=("SGM" "ISGMR" "TRWP" "MeanField")  # SGM needs longer time since it is CPU
n_dirs=(4 8 16)

#img_names=("teddy" "penguin")
#p_funcs=("TL" "TQ")
#n_disps=(60 256)
#truncs=(1 200)
#p_weights=(10 25)
#modes=("ISGMR" "TRWP")
#n_dirs=(4 16)

enable_min_a_dirs=(false)
enable_run_script=true

for idx in "${!img_names[@]}"; do
  img_name=${img_names[idx]}
  p_func=${p_funcs[idx]}
  n_disp=${n_disps[idx]}
  trunc=${truncs[idx]}
  p_weight=${p_weights[idx]}
  for mode in "${modes[@]}"; do
    n_iter=50

    for n_dir in ${n_dirs[@]}; do
      for enable_min_a_dir in ${enable_min_a_dirs[@]}; do
        bash_name="../experiments/scripts/${img_name}_${p_func}_${p_weight}_${n_disp}_${trunc}_${mode}_${n_dir}_${n_iter}.sh"
        log_name="../experiments/logs/${img_name}_${p_func}_${p_weight}_${n_disp}_${trunc}_${mode}_${n_dir}_${n_iter}.txt"
        echo -e "${bash_name}"

        echo -e "#!/bin/bash -l" > ${bash_name}

        echo -e "
#SBATCH --job-name="${img_name}_${p_func}_${p_weight}_${n_disp}_${trunc}_${mode}_${n_dir}_${n_iter}"
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=${log_name}" >> ${bash_name}

        echo -e "
module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
module load xfce/4.12 opencv/3.4.3" >> ${bash_name}

        echo -e "
python main.py \\
--img_name=\"${img_name}\" \\
--context=\"${p_func}\" \\
--mode=\"${mode}\" \\
--n_dir=${n_dir} \\
--n_iter=${n_iter} \\
--p_weight=${p_weight} \\
--n_disp=${n_disp} \\
--truncated=${trunc} \\
--enable_saving_label \\" >> ${bash_name}

        if ${enable_min_a_dir}; then
          echo -e "--enable_min_a_dir \\" >> ${bash_name}
        fi

        eval "chmod 755 ${bash_name}"

        if ${enable_run_script}; then
            eval "sbatch ${bash_name}"
        fi
      done
    done
  done
done
