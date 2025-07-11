# Organ and Tumor Segmentation on CT

In this project, we deal with the segmentation of organs, in particular kidneys, and tumors. We train two different neural networks on 3D CT scan images and compare their performance using the DICE score.

For the dataset, we use the images from the KiTS19 challenge which contain images from 210 patients (further denoted as samples) from three different planes (coronal, sagittal and axial plane).

|<img src="https://public.grand-challenge-user-content.org/logos/challenge/360/Screenshot_from_2019-01-02_17-23-36.x20.jpeg" alt="KiTS19 3D Scan" width="300">|<img src="https://github.com/user-attachments/assets/65944cee-b6f6-4ca9-9f90-ee2db74e31d5" alt="Coronal, Sagittal and Axial Plane" width="300"> |
|-|-|

## Installation

### Python Packages

To get started, install either requirements_full.txt or requirements_minimal.txt in a fresh Conda Project Environment.
We used the Miniconda3 version 23.5.2-0.

#### Create environment

```
conda create --prefix ~/project_env python=3.9
conda activate ~/project_env
pip install -r requirements_minimal.txt
```

### KiTS19 CT Scan images

In order to run the training, we first need the dataset of images.
This can be downloaded via the following script of https://github.com/neheller/kits19

```
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```

For seamless download, activate the dedicated conda project environment and run `python3 -m starter_code.get_imaging` inside the environment.

Note: The package `nibabel` may serve problems from version 5.0.0, the preferred version `4.0.2` is part of the requirement files (see above).


Once the dataset has been downloaded, find recursively in this directory for `kits19/data` (via VSCode Search) and replace the full path with the path from your locally downloaded dataset that contains `.../kits19/data` (if parent directory was not amended) or `../data` (if parent directory was amended).

## Overview

- 2D_data_stats.ipynb: Exploratory analysis between background, kidney and tumor
- dataset_statistics.ipynb: Exploratory analysis of the dataset, such as slices per case, density comparison via HU, proportions of kidney vs. tumor, ...
- eval_unet.ipynb: Post evaluation of the 3D residual U-Net on the validation set. Shows ground truth and prediction in a visualized 3D space
- eval_unetr.py: Post evaluation of the 2d U-Netr on the validation set. Shows ground truth and prediction in a visualized 3D space and randomly selected slices for comparison
- prepare_2D_dataset.py: Prepare 2D training by saving each image and label from a slice per case in a dedicated folder
- requirement_full.txt: Full requirement files, extracted via AI monitoring https://wandb.ai/
- train_3d_unet.ipynb: Notebook script to train the 3D residual U-Net.
- train_3d_unet.py: Equivalent of train_3d_unet.ipynb but as Python script
- train_unetr.py: Script to train the 2D U-Netr
- unetr_hyperparameter_search.py: Exploratory analysis of hyperparameter search for Unetr
- utils/data_load.py: Loads the image data into the DataLoader classes. Contains script for loading the data for evaluation and pre-processing scripts of the data.
- utils/inference.py: Inference/Prediction methods for the evaluation
- utils/train_utils.py: Contains the training and post-training validation script, both for 2D U-Netr and 3D residual U-Net
- utils/transforms.py: Contains the transformation script of the data via MonAI pipeline
- utils/viz.py: Contains the scripts for the visualization part of the evaluation

## Note on hyperparameter search

While the exploratory hyperparameter search for Unetr was conducted via unetr_hyperparameter_search.py, the hyperparameter search for 3D residual U-Net was conducted via different iterations of the script using different branches as part of this GitHub repository.
