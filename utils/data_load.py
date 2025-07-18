from glob import glob
import os
import numpy as np
import random
from pathlib import Path

from monai.data import Dataset, DataLoader, pad_list_data_collate, CacheDataset
from torch.utils.data import ConcatDataset

import nibabel as nib

def get_data_list(data_root):
    """
    Get list of all KiTS19 cases for training.
    
    Args:
        data_root (str): Root directory containing case folders.
        
    Returns:
        list: List of dictionaries with keys 'image', 'label', 'case_id' for each valid case.
        Expected structure: data_root/case_XXXXX/imaging.nii.gz, segmentation.nii.gz
    """
    # Find all case directories matching pattern "case_*"
    case_folders = glob(os.path.join(data_root, "case_*"))
    case_folders.sort()

    data_list = []
    # Iterate through each case folder
    for case_folder in case_folders:
        # Extract case ID from folder name
        case_id = os.path.basename(case_folder)

        # Build paths to imaging and segmentation files
        imaging_path = os.path.join(case_folder, "imaging.nii.gz")
        segmentation_path = os.path.join(case_folder, "segmentation.nii.gz")

        # Only include cases where both files exist
        if os.path.exists(imaging_path) and os.path.exists(segmentation_path):
            data_list.append({
                "image": imaging_path,
                "label": segmentation_path,
                "case_id": case_id
            })

    return data_list


def filter_background_cases(data_list, fraction_to_keep):
    """
    Filters out a fraction of slices that are background-only

    Args:
        data_list (list): List of dictionaries with "image" and "label" keys.
        fraction_to_keep (float): Fraction of total background-only slices to keep.

    Returns:
        filtered_data (list): list in the same format as data_list, with background-only slices filtered out
    """

    filtered_data = []
    background_samples = []

    for d in data_list:
        label = np.load(d["label"])
        if np.all(label == 0):
            background_samples.append(d)
        else:
            filtered_data.append(d)

    # Randomly keep 10% of background samples
    k = int(len(background_samples) * fraction_to_keep)
    kept_background = random.sample(background_samples, k=k)
    print(f"   Kept {len(kept_background)} background-only slices out of {len(background_samples)}", flush=True)

    filtered_data.extend(kept_background)
    return filtered_data


def upsample_tumor_cases(data_list, n_duplicates):
    """
    Upsample cases with tumor (label 2) by duplicating them n_duplicates times. Attempt to help with class imbalance.
    Args:
        data_list (list): List of dictionaries with "image" and "label" keys.
        n_duplicates (int): Number of times to duplicate cases with tumor.
    Returns:
        list: New list with upsampled cases.
    """
    tumor_data = []
    other_data = []
    for d in data_list:
        label = np.load(d["label"])
        if np.any(label == 2):  # Check if tumor is present
            for _ in range(n_duplicates):
                tumor_data.append(d)
        else:
            other_data.append(d)
    
    print(f"   Upsampled  to {len(tumor_data)} tumor slices, originally {len(tumor_data) // n_duplicates} slices with tumor", flush=True)
    return tumor_data, other_data


def get_2D_data(train_cases, val_cases, data_dir):
    """
    Get 2D data for training and validation from the specified directory.

    Args:
        train_cases (list): List of case IDs for training.
        val_cases (list): List of case IDs for validation.
        data_dir (str): Directory containing the prepared 2D data.

    Returns:
        tuple: Two lists containing training and validation data dictionaries.
        Each dictionary contains "image", "label", and "case_id".
    """

    images = sorted(glob(f"{data_dir}/case_*/images/*.npy"))
    labels = sorted(glob(f"{data_dir}/case_*/labels/*.npy"))

    data_list = [{"image": img, "label": lbl, "case_id": [part for part in Path(img).parts if part.startswith("case_")][0]} for img, lbl in zip(images, labels)]

    train_list = [d for d in data_list if d["case_id"] in train_cases]
    train_list = filter_background_cases(train_list, fraction_to_keep=0.2) # Filter out background-only cases


    val_list = [d for d in data_list if d["case_id"] in val_cases]
    return train_list, val_list


def get_case_dataset(case, data_dir, transforms, num_workers):
    """
    Get dataset and dataloader for a specific case's 2D slices along with original NIfTI files.
    
    Args:
        case (str): Case ID (e.g., "case_00000").
        data_dir (str): Directory containing the prepared 2D data.
        transforms (callable): Transform to apply to the data.
        num_workers (int): Number of workers for DataLoader.
        
    Returns:
        tuple: (val_loader, image, segmentation) where:
            - val_loader: DataLoader for 2D slices
            - image: Original NIfTI image object
            - segmentation: Original NIfTI segmentation object
    """
    # Get all 2D slice images and labels for the specified case
    images = sorted(glob(f"{data_dir}/{case}/images/*.npy"))
    labels = sorted(glob(f"{data_dir}/{case}/labels/*.npy"))

    # Create data list with slice-level information
    data_list = [{"image": img, "label": lbl, "case_id": case, "slice_num": img.split("/")[-1].split(".")[0]} for img, lbl in zip(images, labels)]

    # Create dataset and dataloader for 2D slices
    val_ds = Dataset(data=data_list, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    # Load original 3D NIfTI files for reference
    imaging_path = os.path.join(data_dir, case, "imaging.nii.gz")
    segmentation_path = os.path.join(data_dir, case, "segmentation.nii.gz")

    # Load nifti files for header information and metadata
    image = nib.load(imaging_path)
    segmentation = nib.load(segmentation_path)

    return val_loader, image, segmentation

def get_case_dataset_3d(case, data_dir, transforms, num_workers):
    """
    Get dataloader for 3D case processing for a given case
    
    Args:
        case (str): Case ID (e.g., "case_00000").
        data_dir (str): Directory containing the NIfTI files.
        transforms (callable): Transform to apply to the 3D volume.
        num_workers (int): Number of workers for DataLoader.
        
    Returns:
        tuple: (val_loader, original_image, transformed_label) where:
            - val_loader: DataLoader for 3D volume
            - original_image: Original NIfTI image object with metadata
            - transformed_label: Transformed label tensor for evaluation
    """
    # Build paths to 3D NIfTI files
    imaging_path = os.path.join(data_dir, case, "imaging.nii.gz")
    segmentation_path = os.path.join(data_dir, case, "segmentation.nii.gz")
    
    # Create data list with single entry for the full 3D volume
    data_list = [{
        "image": imaging_path, 
        "label": segmentation_path, 
        "case_id": case
    }]
    
    # Create dataset and dataloader for 3D volume processing
    val_ds = Dataset(data=data_list, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    # Get the transformed data for ground truth comparison
    # Extract the first (and only) item from the dataset
    transformed_data = val_ds[0]
    
    # Extract the transformed label tensor for evaluation
    transformed_label = transformed_data["label"]
    
    # Load original image with metadata for reconstruction
    original_image = nib.load(imaging_path)
    
    return val_loader, original_image, transformed_label


def get_test_case(case, data_dir, transforms, num_workers):
    """
    Get dataset and dataloader for test case inference (no labels available).
    
    Args:
        case (str): Case ID (e.g., "case_00000").
        data_dir (str): Directory containing the NIfTI files.
        transforms (callable): Transform to apply to the data.
        num_workers (int): Number of workers for DataLoader.
        
    Returns:
        tuple: (test_loader, image_nifti) where:
            - test_loader: DataLoader for slice-by-slice inference
            - image_nifti: Original NIfTI image object for reconstruction
    """
    # Load the 3D imaging volume for test case (no labels available)
    imaging_path = os.path.join(data_dir, case, "imaging.nii.gz")
    image_nifti = nib.load(imaging_path)
    image_volume = image_nifti.get_fdata(dtype=np.float32)  # (H, W, D)

    # Create data list with one entry per slice for 2D processing
    data_list = [
        {"image": image_volume[i, :, :], "case_id": case}
        for i in range(image_volume.shape[0])
    ]

    # Create dataset and dataloader for slice-by-slice inference
    test_ds = Dataset(data=data_list, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return test_loader, image_nifti


def get_2D_datasets(train_list, val_list, augment_transforms, no_aug_transforms, batch_size=4, num_workers=4):
    """
    Create training and validation datasets and loaders for 2D data.
    Args:
        train_list (list): List of training data dictionaries.
        val_list (list): List of validation data dictionaries.
        aug_transform (callable): Augmentation transform for training data.
        no_aug_transform (callable): No-augmentation transform for validation data.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        
    Returns:
        tuple: Training and validation DataLoaders, and their corresponding datasets.
    """
    # Split training data into tumor and non-tumor cases for different handling
    tumor_list, other_list = upsample_tumor_cases(train_list, n_duplicates=3)  # Upsample tumor cases

    # Apply augmentation to tumor data to improve learning, no augmentation for other data
    tumor_ds = CacheDataset(data=tumor_list, transform=augment_transforms, cache_rate=1.0)
    other_ds = CacheDataset(data=other_list, transform=no_aug_transforms, cache_rate=1.0)

    # Combine tumor and non-tumor datasets for training
    train_ds = ConcatDataset([tumor_ds, other_ds])
    # Validation dataset without augmentation
    val_ds = CacheDataset(data=val_list, transform=no_aug_transforms, cache_rate=1.0)

    # Create dataloaders with appropriate settings
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds, val_ds


def get_3D_dataset(train_data, val_data, train_transforms, val_transforms, BATCH_SIZE=1, NUM_WORKERS=4):
    """
    Create training and validation datasets and loaders for 3D data.
    Args:
        train_data (list): List of training data dictionaries.
        val_data (list): List of validation data dictionaries
        train_transforms (callable): Transform for training
        val_transforms (callable): Transform for validation
        BATCH_SIZE (int): Batch size for DataLoader.
        NUM_WORKERS (int): Number of workers for DataLoader.
    Returns:
        tuple: Training and validation DataLoaders.
    """

    # Create 3D datasets
    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_transforms)

    # Create DataLoaders with collate function to handle variable sizes

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_list_data_collate,  # handles size mismatches
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        collate_fn=pad_list_data_collate,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    return train_loader, val_loader