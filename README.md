# Related Publication

  This work is accepted as an oral paper by ACCV 2020. If you find our paper or code useful, please cite our work as follows.

  [**"Fast and Differentiable Message Passing on Pairwise Markov Random Fields"**](https://arxiv.org/abs/1910.10892) (Oral)\
  Zhiwei Xu, Thalaiyasingam Ajanthan, Richard Hartley\
  Asian Conference on Computer Vision (ACCV), November 2020, Japan

# Requirements
  To compile MP Layers, please do as follows
  
  - Install opencv for C++, opencv3.4.3 in our case, then set "[path]" for MP layers (ours is in "./MPLayers/compile.sh") in **~/.bashrc** by
        ```
        export PYTHONPATH=[path]/MPLayers:[path]/MPLayers/Stereo:[path]/MPLayers/Segmentation:$PYTHONPATH$
        ```

  - Edit files as follows

    In **"./MPLayers/compile.sh"** and **"./MPLayers_soft/cuda/compile.sh"**, set the "[path]" in Step 1 by
    ```
    python setup.py develop --install-dir=[path]/MPLayers
    ```

    In **"./MPLayers/setup.py"** and **"./MPLayers_soft/cuda/setup.py"**,
    ```
    REPLACE
        include_dir = ['aux', '../tools/cpp',
                       '/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/include',
                       '/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/include/opencv',
                       '/apps/opencv/3.4.3/include',
                       '/apps/opencv/3.4.3/include/opencv']
        library_dir = ['/mnt/scratch/zhiwei/Installations/anaconda3/envs/train-cuda/lib',
                       '/apps/opencv/3.4.3/lib64']
    BY
        include_dir = ['aux', '../tools/cpp',
                       '[opencv path]/include',
                       '[opencv path]/include/opencv']
        library_dir = ['[opencv path]/lib',
                       '[opencv path]/lib64']
    ```
  - Start compiling for MPLayer libraries (will be stored in "lib_stere_slim" and "lib_seg_slim") by running
    
    For ISGMR,
    ```
    cd MPLayers;
    set --mode="stereo" in compile.sh;
    ./compile.sh;
    
    When it is finished
    set --mode="segmentation" in compile.sh;
    ./compile.sh
    ```

    For TRWP,
    ```
    cd MPLayers_soft/cuda;
    set --mode="stereo" in compile.sh;
    ./compile.sh;
    
    When it is finished
    set --mode="segmentation" in compile.sh;
    ./compile.sh
    ```
To run deep semantic segmentation, please install
  
  ```
  python/3.6.1
  tensorboardx/1.2.0-py36-cuda90
  torchvision/0.2.1-py36
  pytorch/0.4.1-py36-cuda90 (we also tested on pytorch/1.1.0-py36-cuda10, pytorch/0.4.0-py36-cuda90)
  cuda/9.2.88
  gcc/6.4.0
  eigen/3.2.9
  ```

# Difference between MPLayers and MPLayers_soft
  
  MPLayers contains MAP TRWP (TRWP) and MAP ISGMR (ISGMR), but TRWP will not be used.
  
  MPLayers_soft contains MAP and marginal TRWP (TRWP_hard_soft) but soft TRWP was not used in our paper; ISGMR is not compiled in MPLayers_soft.


# How to Use

**Energy Minimization**
  
- Dataset
    
  - Stereo: All "Middlebury", "ETH3D", and "KITTI2015" images used in our paper are in "./datasets"
      
  - Denoise: Download ["Denoise"](https://1drv.ms/u/s!AngC1-tRlyPMgS7cZ4MNqS1VD4Nf?e=lb94Rg) files (containing unary terms of "penguin" and "house") and put them in "./datasets/Denoise"
    
- Running
  ```
  cd MPLayers;
  ./run_all.sh;
  ```
  
- Max Labels

  Since denoise needs 256 labels, we set MAX_DISPARITY=int(256) in "MPLayers/setup.py" and "MPLayers/cuda/setup.py".
  One could reset it on demand, such as MAX_DISPARITY=192 or 96, to largely use the max number of blocks and threads in CUDA.
  
**Deep Semantic Segmentation**

  - Dataset: Download Berkeley benchmark and PASCAL VOC 2012 using the scripts from ["./data/"](https://github.com/meng-tang/rloss.git), put the merged datasets as "./datasets/pascal_scribble" (although ours is for fully-supervised learning).
    It should contains folders such as "ImageSets", "JPEGImages", "SegmentationClassAug", etc.
  
  - Ensure folders of "datasets", "experiments", and "pretrained" exist.
    Then, download [our models](https://1drv.ms/u/s!AngC1-tRlyPMgRx6ahmhqxqJDf65?e=UQRUBN) and stored in each subfolder, e.g., "pretrained/TRWP/model.pth.tar".
      
  - To test using "run.sh"
    ```
    add "--enable_test" in the command
    set "--mpnet_mrf_mode" as "TRWP", "ISGMR", "SGM", "MeanField", or "vanilla"
    ./run.sh
    ```
      
  - To train using "run.sh"
    ```
    remove "--enable_test" and "--resume" from the command
    set --resume_unary="pretrained/vanilla/model.pth.tar" in the command for "--mpnet_mrf_mode" in ["TRWP", "ISGMR", "SGM", "MeanField"]
    remove "--resume_unary" from the command for --mpnet_mrf_mode="vanilla"
    ./run.sh
    ```
  
# Note
  We will keep updating this repository.
