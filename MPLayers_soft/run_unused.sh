CUDA_VISIBLE_DEVICES="2" \
python main.py \
--left_img_path="/mnt/data/users/u5710355/WorkSpace/pytorch-projects/MPLayers/datasets/stereo/tsukuba/imL.ppm" \
--right_img_path="/mnt/data/users/u5710355/WorkSpace/pytorch-projects/MPLayers/datasets/stereo/tsukuba/imR.ppm" \
--save_dir="/mnt/data/users/u5710355/WorkSpace/pytorch-projects/MPLayers/experiments" \
--img_name="tsukuba" \
--mode="TRWP" \
--n_dir=4 \
--n_iter=50 \
--context="TL" \
--truncated=2 \
--n_disp=16 \
--p_weight=10 \
--enable_cuda \
--enable_display \
--enable_saving_label