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

> [!NOTE]
> Installing requirements other than this leads to problem fetching the dataset! Sometime, Python package may still be missing due to MONAI, therefore use `requirements_full.txt` or visit https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies

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

> [!NOTE]
> Run this script in the conda environment `project_env`!

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

> [!NOTE]
> Please run `prepare_2D_dataset.py` before!

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

> [!NOTE]
> We run the script(s) on RAMSES. Tested in Jupyter Browser, and ran for long-process (training) as a daemon process using tmux.

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

### Checkpoints
- `checkpoints/3d-unet`: Contains the model checkpoint file (`kits19-model-3d-unet.pth`, used in eval_unet.ipynb), the training progress in JSON throughout all epochs, the model checkpoint from accelerator library and the training log from wandb in `wandb/output.log.txt`

## Note on Hyperparameter search

While the exploratory hyperparameter search for Unetr was conducted via unetr_hyperparameter_search.py, the hyperparameter search for 3D residual U-Net was conducted via different iterations of the script using different branches as part of this GitHub repository.

## Monitoring

For 3D Residual U-Net, the model was monitored via Weights and Biases (https://wandb.ai/).
This is optional but nice-to-have.

To enable monitoring, install

```bash
pip install wandb
```

> [!NOTE]
> `wandb` is already part of requirements_full.txt and requirements_minimal.txt

And login to wandb. The instruction to log in is detailed in Weights & Biases dashboard.

And set the flag `use_wandb` to `True` in `train_kits19_model` methods to activate it. You may amend the `wandb_project` and `wandb_notes` if you want.

<img width="1794" height="632" alt="image" src="https://github.com/user-attachments/assets/83e37667-9f36-4580-bc05-7790efa1417a" />

Otherwise, the training is monitored and outputted to the file `training_progress.json`.

## Interactive Result

This is a segmentation of the kidney and tumor from the sagittal plane, predicted using 3D U-Net via sliding window inference.

![case_00199_sagittal_slices (2)](https://github.com/user-attachments/assets/26e7f114-9dc4-447a-8c4e-a75527752702)


## Contributing

This project was developed as part of a Computer Vision course project at the University of Cologne.

## Testing

The script(s) has been successfully tested to run in the Conda environment with the requirements (`requirements_full.txt`).

## References

For the training, we use the models (for both 2D and 3D models) from MONAI v1.5.0 (https://github.com/Project-MONAI/MONAI) and the dataset from https://github.com/neheller/kits19 (see guide above).

### Academic References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015, 234-241.
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. International conference on medical image computing and computer-assisted intervention, 424-432.
- Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-net: Fully convolutional neural networks for volumetric medical image segmentation. 2016 fourth international conference on 3D vision (3DV), 565-571.
- Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A. L., & Zhou, Y. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv preprint arXiv:2102.04306.
- Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H. R., & Xu, D. (2022). Unetr: Transformers for 3d medical image segmentation. Proceedings of the IEEE/CVF winter conference on applications of computer vision, 574-584.

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.

- Heller, N., Sathianathen, N., Kalapara, A., Walczak, E., Moore, K., Kaluzniak, H., Rosenberg, J., Blake, P., Rengel, Z., Oestreich, M., Dean, J., Tradewell, M., Shah, A., Tejpaul, R., Edgerton, Z., Peterson, M., Raza, S., Regmi, S., Papanikolopoulos, N., & Weight, C. (2020). The KiTS19 Challenge Data: 300 Kidney Tumor Cases with Clinical Context, CT Semantic Segmentations, and Surgical Outcomes. arXiv preprint arXiv:1904.00445.
- Heller, N., Isensee, F., Maier-Hein, K. H., Hou, X., Xie, C., Li, F., Nan, Y., Mu, G., Lin, Z., Han, M., et al. (2020). The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge. Medical Image Analysis, 101821.

- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv preprint arXiv:1608.03983.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101.
- Nie, Y., Carratù, M., O'Nils, M., Sommella, P., Moise, A. U., & Lundgren, J. (2022). Skin Cancer Classification based on Cosine Cyclical Learning Rate with Deep Learning. 2022 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), 1-6.

- Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Jorge Cardoso, M. (2017). Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. Deep learning in medical image analysis and multimodal learning for clinical decision support, 240-248.
- Yeung, M., Sala, E., Schönlieb, C. B., & Rundo, L. (2022). Unified focal loss: Generalising dice and cross entropy-based losses to handle class imbalanced medical image segmentation. Computerized Medical Imaging and Graphics, 95, 102026.

- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in neural information processing systems, 32.

- Isensee, F., & Maier-Hein, K. H. (2019). An attempt at beating the 3D U-Net. arXiv preprint arXiv:1908.02182.
- He, J., Luo, Z., Lian, S., Su, S., & Li, S. (2024). Towards accurate abdominal tumor segmentation: A 2D model with Position-Aware and Key Slice Feature Sharing. Computers in Biology and Medicine, 179, 108743.
- Drole, L., Poles, I., D'Arnese, E., & Santambrogio, M. D. (2023). Towards a Lightweight 2D U-Net for Accurate Semantic Segmentation of Kidney Tumors in Abdominal CT Images. IEEE EUROCON 2023-20th International Conference on Smart Technologies, 12-17.

- Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024). Segment anything in medical images. Nature Communications, 15(1), 654.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.
