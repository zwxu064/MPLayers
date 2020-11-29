#!/bin/bash -l

# Config
model_name="resnet101"
mode="ISGMR"  # vanilla(unary)/vanilla_ft/TRWP/ISGMR/MeanField/SGM
task_mode="fully"  # fully/weakly
cnn_mode="test"  # train/test
edge_mode="canny"  # canny/sobel/CRISP/BASS
epochs=40
term_weight=5
max_iter=5
num_dirs=8
scale_list=0.5
lr=1e-6
enable_run_script=true
enable_debug=true
lr_scheduler="poly"
sp=""
sp_threshold=0
warmup_epochs=0
enable_ft_single_lr=false
enable_adjust_val=false
enable_score_scale=false
disable_aspp=false

if ${disable_aspp}; then
    aspp_mode="noaspp"
else
    aspp_mode="aspp"
fi

if [ ${mode} == "TRWP" ]; then
    num_dirs=4
elif [[ ${mode} =~ ^(ISGMR|SGM) ]]; then
    num_dirs=8
fi

if [ ${mode} == "MeanField" ]; then
    batch_size=10
else
    batch_size=12
fi

if [ ${mode} == "SGM" ]; then
    max_iter=1
fi

if [ ${mode} == "vanilla" ]; then
    resume="/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/results/pretrained/unary/${model_name}_${aspp_mode}"
elif [ ${edge_mode} == "canny" ]; then
    resume="/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/results/master/fully/"
    resume+="resnet101_train_fully_${mode}_edgecanny/okay/"
    resume+="tw${term_weight}_iter${max_iter}_dir${num_dirs}_lr${lr}_poly_epc${epochs}_warm0_scale${scale_list}_lr2"
    resume+="/pascal/deeplab-resnet/train_batch12_withoutcrf_${mode}"
elif [ ${edge_mode} == "gt_edge" ]; then
    resume="/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/results/master/${task_mode}"
    resume+="/${mode}_gt_edge/tw${term_weight}_lr${lr}_poly_epc_warm0_scale${scale_list}_lr2/pascal/deeplab-resnet"
    resume+="/train_batch12_withoutcrf_${mode}"
fi

resume+="/model_best.pth.tar"
#resume+="/ckpt_60.pth.tar"

if [ ${mode} == "vanilla" ]; then
    term_weight=0
    max_iter=0
    warmup_epochs=0
    enable_ft=false
    enable_ft_single_lr=false
    edge_mode=""
    bash_name="${cnn_mode}_unary.sh"
    lr=0.007
    epochs=60
    scale_list=0
    lr_scheduler="poly"
    num_dirs=0
elif [[ ${mode} =~ ^(vanilla_ft|TRWP|ISGMR|MeanField|SGM) ]]; then
    enable_ft=true

    if [ ${mode} == "vanilla_ft" ]; then
        term_weight=0
        max_iter=0
        bash_name="${cnn_mode}_vanilla_ft.sh"
        edge_mode=""
    else
        bash_name="${cnn_mode}_${mode}_${edge_mode}.sh"

        # Edge
        if [[ ${edge_mode} =~ ^(canny|sobel) ]]; then
            scale_list=0.5  # ensure not too small
        elif [ ${edge_mode} == "BASS" ]; then
            sp="BASS"
            sp_threshold=800
        elif [ ${edge_mode} == "CRISP" ]; then
            sp="CRISP"
            sp_threshold=0.05
        fi
    fi
else
    echo "Error! Invalid mode ${mode}"
    exit
fi

if [ ${cnn_mode} == "train" ]; then
    if ${enable_debug}; then
        cluster_time=1
    else
        cluster_time=24
    fi

    enable_test=false
    enable_adjust_val=false
    resume=""
    val_bz=12
else
    cluster_time=1
    enable_test=true
    val_bz=1
fi

run_mode="${model_name}_${cnn_mode}_${task_mode}_${mode}_edge${edge_mode}"
run_type="tw${term_weight}_iter${max_iter}_dir${num_dirs}"
run_type+="_lr${lr}_${lr_scheduler}_epc${epochs}_warm${warmup_epochs}_scale${scale_list}"

if ${enable_score_scale}; then
    run_type+="_sscale"
fi

if ${enable_ft_single_lr}; then
    run_type+="_lr1"
else
    run_type+="_lr2"
fi

if ${enable_adjust_val}; then
    run_type+="_adjval"
fi

if ${disable_aspp}; then
    run_type+="_noaspp"
fi

#run_type+="_layernorm"
#run_type+="MFEdgeW0.01"

bash_name="scripts/${bash_name}"
log_name="${run_mode}_${run_type}"
checkpoint_dir="/home/xu064/WorkSpace/git-lab/pytorch-projects/rloss/results/master/${task_mode}"

if ${enable_debug}; then
    log_name="debug_${log_name}"
    checkpoint_dir+="/debug/${run_mode}/${run_type}"
    user_mail=""
else
    checkpoint_dir+="/${run_mode}/${run_type}"
    user_mail=""
fi

echo ${log_name}

# Write batch head
echo -e "#!/bin/bash -l" \
> ${bash_name}

echo -e "
#SBATCH --job-name=${log_name}
#SBATCH --time=${cluster_time}:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${user_mail}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/${log_name}.txt" \
>> ${bash_name}

# Write modules
echo -e "
# !!! Option 1: PyTorch 1.1.0
# # module load python/2.7.13 cuda/9.0.176 pytorch/0.4.1-py27-cuda90  # This is original config, too old
# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36 python/3.6.1 cuda/9.0.176 pytorch/1.1.0-py36-cuda90\n
# !!! Option 2: PyTorch 0.4.1, MPLayers on Data61 can only be supported by PyTorch 0.4.1 by far
module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
module load pytorch/0.4.1-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1
module load xfce/4.12 opencv/3.4.3" \
>> ${bash_name}

# Write main command
echo -e "
python3 train_withdensecrfloss.py \\
--server=\"data61\" \\
--backbone=\"${model_name}\" \\
--lr=${lr} \\
--gpu-ids=\"0\" \\
--workers=4 \\
--epochs=${epochs} \\
--batch-size=${batch_size} \\
--val-batch-size=${val_bz} \\
--checkname=\"deeplab-resnet\" \\
--rloss-scale=0.5 \\
--eval-interval=1 \\
--dataset=\"pascal\" \\
--save-interval=1 \\
--densecrfloss=0 \\
--mpnet_mrf_mode=\"${mode}\" \\
--lr-scheduler=\"${lr_scheduler}\" \\
--warmup_epochs=${warmup_epochs} \\
--base-size=512 \\
--crop-size=512 \\
--mode=\"${task_mode}\" \\
--mpnet_term_weight=${term_weight} \\
--mpnet_smoothness_mode=\"Potts\" \\
--mpnet_scale_list=${scale_list} \\
--mpnet_n_dirs=${num_dirs} \\
--mpnet_max_iter=${max_iter} \\
--edge_mode=\"${edge_mode}\" \\
--sigmoid_scale=1 \\
--optimizer=\"SGD\" \\
--superpixel=\"${sp}\" \\
--superpixel_threshold=${sp_threshold} \\
--checkpoint_dir=\"${checkpoint_dir}\" \\
--output_directory=\"${checkpoint_dir}/seg\" \\" \
>> ${bash_name}

# Finetune
if ${enable_ft}; then
    echo -e "--ft \\" >> ${bash_name}
fi

if [ ${cnn_mode} == "train" ]; then
    if ${enable_ft}; then
        resume_unary="/home/xu064/Results/Weakly_Seg/pretrained/unary/${model_name}_${aspp_mode}/model_best.pth.tar"
        echo -e "--resume_unary=\"${resume_unary}\" \\" >> ${bash_name}
    fi
else
    echo -e "--resume=\"${resume}\" \\" >> ${bash_name}
fi

# Finetune with single LR
if ${enable_ft_single_lr}; then
    echo -e "--enable_ft_single_lr \\" >> ${bash_name}
fi

if ${enable_test}; then
    echo -e "--enable_test \\" >> ${bash_name}
    if ${enable_adjust_val}; then
        echo -e "--enable_adjust_val \\" >> ${bash_name}
    fi
fi

if ${enable_score_scale}; then
    echo -e "--enable_score_scale \\" >> ${bash_name}
fi

if ${disable_aspp}; then
    echo -e "--disable_aspp \\" >> ${bash_name}
fi

eval "chmod 755 ${bash_name}"

# Run
if ${enable_run_script}; then
    eval "sbatch ${bash_name}"
fi
