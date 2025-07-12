# 3D Kidney and Tumor Segmentation using Deep Learning

This project implements and compares two deep learning architectures for automatic segmentation of kidneys and kidney tumors from 3D CT scans using the KiTS19 dataset. We evaluate a 3D residual U-Net against a 2D U-NETR (Vision Transformer) to assess their performance on this challenging medical image segmentation task.

## Project Overview

This project trains and compares two neural network architectures:

- **3D Residual U-Net**: 3D Residual U-Net approach
- **2D U-NETR**: Vision Transformer-based approach

Both models are implemented using the MONAI library (v1.5) and evaluated using DICE score metrics and visualizations in the evaluation.

## Dataset

We use the [KiTS19 Challenge dataset](https://kits19.grand-challenge.org) which contains:
- **210 training cases** with CT scans and expert annotations
- **90 test cases** for evaluation
- High-resolution 3D CT images (coronal, sagittal, axial) with kidney and tumor segmentations

|<img src="https://public.grand-challenge-user-content.org/logos/challenge/360/Screenshot_from_2019-01-02_17-23-36.x20.jpeg" alt="KiTS19 3D Scan" width="300">|<img src="https://github.com/user-attachments/assets/65944cee-b6f6-4ca9-9f90-ee2db74e31d5" alt="Coronal, Sagittal and Axial Plane" width="300"> |
|-|-|

### Data Structure
```
data/
├── case_00000/
│   ├── imaging.nii.gz
│   └── segmentation.nii.gz
├── case_00001/
│   ├── imaging.nii.gz
│   └── segmentation.nii.gz
...
└── kits.json
```

Segmentation labels: 0 = background, 1 = kidney, 2 = tumor

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- Miniconda3 version 23.5.2-0

### Environment Setup

1. **Create a new conda environment:**
```bash
conda create --prefix ~/project_env python=3.9
conda activate ~/project_env
```

2. **Install dependencies:**
```bash
# For minimal installation
pip install -r requirements_minimal.txt

# For full installation with all development tools
pip install -r requirements_full.txt
```

Note: Installing requirements other than this leads to problem fetching the dataset! Sometime, Python package may still be missing due to MONAI, therefore use `requirements_full.txt` or visit https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies

3. **Register kernel:**

Finally, register the kernel to Jupyter via

```bash
python -m ipykernel install --user --name project_env --display-name "Python (project_env)"
```

This is necessary when selecting the kernel via Jupyter browser.

### Dataset Download

1. **Clone the KiTS19 repository:**
```bash
git clone https://github.com/neheller/kits19
cd kits19
```

2. **Download the imaging data:**
```bash
python -m starter_code.get_imaging
```

Note: Run this script in the conda environment `project_env`!

3. **Update data paths:**
   - Search for `kits19/data` references in the project files
   - Replace with your local dataset path (e.g., `path/to/your/kits19/data`)

## Quick Start

### Training the 3D Residual U-Net
```bash
# Notebook version (recommended for development), otherwise run in Jupyter browser
jupyter notebook train_3d_unet.ipynb

# Script version
python train_3d_unet.py
```

### Training the 2D U-NETR

Note: Please run `prepare_2D_dataset.py` before!

```bash
python train_unetr.py
```

### Evaluation
```bash
# Evaluate 3D U-Net, otherwise run in Jupyter browser
jupyter notebook eval_unet.ipynb

# Evaluate 2D U-NETR
python eval_unetr.py
```

## Project Structure

### Core Files
- `train_3d_unet.ipynb` / `train_3d_unet.py`: 3D Residual U-Net training implementation
- `train_unetr.py`: 2D U-NETR training script
- `eval_unet.ipynb`: 3D U-Net evaluation with 3D visualizations
- `eval_unetr.py`: 2D U-NETR evaluation with 3D visualizations and slice-by-slice comparisons

### Data Analysis
- `dataset_statistics.ipynb`: Comprehensive dataset analysis (slice counts, HU distributions, class proportions)
- `2D_data_stats.ipynb`: 2D slice-level analysis of background, kidney, and tumor regions
- `prepare_2D_dataset.py`: Converts 3D volumes to 2D slices for U-NETR training

### Utilities
- `utils/data_load.py`: DataLoader implementations and preprocessing pipelines
- `utils/transforms.py`: MONAI-based data augmentation and preprocessing transforms
- `utils/train_utils.py`: Training loops and validation routines for both architectures
- `utils/inference.py`: Model inference and prediction utilities
- `utils/viz.py`: 3D and 2D visualization tools for qualitative evaluation

### Hyperparameter Optimization
- `unetr_hyperparameter_search.py`: Automated hyperparameter tuning for U-NETR

## Note on Hyperparameter search

While the exploratory hyperparameter search for Unetr was conducted via unetr_hyperparameter_search.py, the hyperparameter search for 3D residual U-Net was conducted via different iterations of the script using different branches as part of this GitHub repository.

## Monitoring

For 3D Residual U-Net, the model was monitored via Weights and Biases (https://wandb.ai/).
This is optional but nice-to-have.

To enable monitoring, install

```bash
pip install wandb
```

Note: `wandb` is already part of requirements_full.txt

And login to wandb. The instruction to log in is detailed in Weights & Biases dashboard.

And set the flag `use_wandb` to `True` in `train_kits19_model` methods to activate it. You may amend the `wandb_project` and `wandb_notes` if you want.

<img width="1794" height="632" alt="image" src="https://github.com/user-attachments/assets/83e37667-9f36-4580-bc05-7790efa1417a" />

Otherwise, the training is monitored and outputted to the file `training_progress.json`.

## Contributing

This project was developed as part of a Computer Vision course project at the University of Cologne.

## References

For the training, we use the models (for both 2D and 3D models) from MONAI v1.5.0 (https://github.com/Project-MONAI/MONAI) and the dataset from https://github.com/neheller/kits19 (see guide above).
