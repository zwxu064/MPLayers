# Related Publication

  This work is accepted as an oral paper by ACCV 2020.

  [**"Fast and Differentiable Message Passing on Pairwise Markov Random Fields"**](https://arxiv.org/abs/1910.10892) (Oral)\
  Zhiwei Xu, Thalaiyasingam Ajanthan, Richard Hartley\
  Asian Conference on Computer Vision (ACCV), November 2020, Japan
  
  If you find our paper or code useful, please cite our work as follows.
  ```
  @article{xu2020mplayers,
  title={Fast and Differentiable Message Passing on Pairwise Markov Random Fields},
  author={Zhiwei Xu, Thalaiyasingam Ajanthan, and Richard Hartley},
  journal={Asian Conference on Computer Vision},
  year={2020}
  }
  ```

# Requirements
  To compile MP Layers, please do as follows
  
  - Install OpenCV for C++, OpenCV3.4.3 in our case, then set "[path]" for MP layers (ours is in "./MPLayers/compile.sh") in **~/.bashrc** by
    ```
    export PYTHONPATH=[path]/MPLayers:[path]/MPLayers/Stereo:[path]/MPLayers/Segmentation:$PYTHONPATH$;
    source ~/.bashrc;
    ```

  - Edit files as follows

    In **"./MPLayers/compile.sh"**, set the "[path]" by
    ```
    python setup.py --mode="stereo" develop --install-dir=[path]/MPLayers  # while mode: "stereo"|"segmentation"
    ```

    In **"./MPLayers/setup.py"**,
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
                       '[OpenCV path]/include',
                       '[OpenCV path]/include/opencv']
        library_dir = ['[OpenCV path]/lib',
                       '[OpenCV path]/lib64']
    ```
  - Start compiling for MPLayer libraries (will be stored in "lib_stere_slim" and "lib_seg_slim") by running
    ```
    cd MPLayers;
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

# Library
  
  In either "./MPLayers/lib_seg" or "./MPLayers/lib_stereo", it contains libraries of "TRWP", "ISGMR", "TRWP_hard_soft", and "compute_terms".
  
  ```
  TRWP: MAP TRWP used in our paper.
  ISGMR: MAP ISGMR used in our paper.
  TRWP_hard_soft: MAP and marginal TRWP although marginal TRWP was not used in our paper.
  compute_terms: only used for stereo unary terms via OpenCV.
  ```

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
    It should contain folders such as "ImageSets", "JPEGImages", "SegmentationClassAug", etc.
  
  - Ensure folders of "datasets", "experiments", and "pretrained" exist.
    Then, download [our models](https://1drv.ms/u/s!AngC1-tRlyPMgRx6ahmhqxqJDf65?e=UQRUBN) and stored in each subfolder, e.g., "pretrained/TRWP/model.pth.tar".
      
  - To test using "test.sh"
    ```
    set "--mpnet_mrf_mode" as "TRWP", "ISGMR", "SGM", "MeanField", or "vanilla"
    ./test.sh
    ```
      
  - To train using "train.sh"
    ```
    set --resume_unary="pretrained/vanilla/model.pth.tar" if "--mpnet_mrf_mode" in ["TRWP", "ISGMR", "SGM", "MeanField"]
    remove "--resume_unary" if --mpnet_mrf_mode="vanilla"
    ./train.sh
    ```
  
# Note
  We will keep updating this repository.
