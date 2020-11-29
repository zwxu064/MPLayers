# Related Publication
- This work is accepted as an oral paper by ACCV 2020. If you find our paper or code useful, please cite our work as follows.

    [**"Fast and Differentiable Message Passing on Pairwise Markov Random Fields"**](https://arxiv.org/abs/1910.10892) (Oral)\
    Zhiwei Xu, Thalaiyasingam Ajanthan, Richard Hartley\
    Asian Conference on Computer Vision (ACCV), November 2020, Japan

# Requirements
  Our code supports PyTorch with versions > 1.0 and <= 1.0. For other dependencies, please see "test.sh".

# How to Use
  - Step 1: Replace the soft links of "datasets", "experiments", "pretrained" folders by yours. Then, download [our models](https://1drv.ms/u/s!AngC1-tRlyPMgRx6ahmhqxqJDf65?e=UQRUBN) from OneDrive.
  
  - Step 2: [To test] set "enable_test=true" in "test.sh" and change mode to "TRWP", "ISGMR", "SGM", "MeanField", or "vanilla".
  
  - Step 3: [To train] set "enable_test=false" and change mode to one mentioned in Step 2.

# Note
  We will keep updating this repository.
