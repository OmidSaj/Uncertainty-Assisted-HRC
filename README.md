# Uncertainty-Assisted Image-Processing for Human-Robot Close Collaboration
This is the official repository for the code and models in [this paper](https://ieeexplore.ieee.org/document/9712200). At this time, a demo of the proposed uncertainty-assisted framework on the [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) test set is presented. You might use the compressed test images and masks in .npz format from the release section [here](https://github.com/OmidSaj/Uncertainty-Assisted-HRC/releases/tag/data_v1). 

**Update 10/1/2022
The [HRC dataset](https://github.com/OmidSaj/Uncertainty-Assisted-HRC/releases/tag/data_v2) and [code](https://github.com/OmidSaj/Uncertainty-Assisted-HRC/tree/main/UB_dataset_experiments) are now added to this repo. To see how the dataset is loaded and fed to the ML models, feel free to review the jupyter notebooks (like [this](https://github.com/OmidSaj/Uncertainty-Assisted-HRC/blob/main/UB_dataset_experiments/R2-ZL-tf26-agent/001_DenseNet-MFW-train_sp_1.ipynb))for each stage of the process and update the necessary directries. 

![Surrogate models](https://github.com/OmidSaj/Uncertainty-Assisted-HRC/blob/main/assets/SRG.JPG)

# Deep learning models 
Models are developed using [TensorFlow](https://www.tensorflow.org/) 2.6 in Python. 
