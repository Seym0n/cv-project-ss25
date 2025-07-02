import glob
import os
import numpy as np
import random
from pathlib import Path

from monai.data import Dataset, DataLoader, pad_list_data_collate
from torch.utils.data import ConcatDataset




def get_data_list(data_root):
    """
    Get list of all KiTS19 cases for training.
    Expected structure: data_root/case_XXXXX/imaging.nii.gz, segmentation.nii.gz
    """
    case_folders = glob(os.path.join(data_root, "case_*"))
    case_folders.sort()

    data_list = []
    for case_folder in case_folders:
        case_id = os.path.basename(case_folder)

        imaging_path = os.path.join(case_folder, "imaging.nii.gz")
        segmentation_path = os.path.join(case_folder, "segmentation.nii.gz")

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

    val_list = [d for d in data_list if d["case_id"] in val_cases]
    return train_list, val_list


def get_2D_datasets(train_list, val_list, aug_transform, no_aug_transform, batch_size=4, num_workers=4):
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

    train_list = filter_background_cases(train_list) # Filter out background-only cases
    tumor_list, other_list = upsample_tumor_cases(train_list, n_duplicates=10)  # Upsample tumor cases

    # augment tumor data, but not other data
    tumor_ds = Dataset(data=tumor_list, transform=aug_transform)
    other_ds = Dataset(data=other_list, transform=no_aug_transform)
    
    # combine datasets
    train_ds = ConcatDataset([tumor_ds, other_ds])
    val_ds = Dataset(data=val_list, transform=no_aug_transform)

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