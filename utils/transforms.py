from monai.transforms import (
    MapTransform, ResizeD, Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ToTensord, EnsureTyped,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandShiftIntensityd, SpatialPadd, CropForegroundd, RandRotated,
    RandCropByLabelClassesd, Rand3DElasticd
)
import numpy as np
import torch

class LoadNumpy(MapTransform):
    """
    Custom MONAI transform to load numpy arrays from file paths.
    
    Args:
        keys (list): List of keys in the data dictionary to process.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        # Create a copy of the input data dictionary
        d = dict(data)
        for key in self.keys:
            # Load numpy array from file path
            array = np.load(d[key])

            # Ensure channel-first dimension for 2D arrays
            if array.ndim == 2:  # (H, W) → (1, H, W) 
                array = np.expand_dims(array, axis=0)
            d[key] = array
        return d

class PrepareSliceData(MapTransform):
    """
    Custom MONAI transform to prepare slice data for 2D processing.
    Ensures correct data types and channel dimensions for image slices.
    
    Args:
        keys (list): List of keys in the data dictionary to process.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        # Create a copy of the input data dictionary
        d = dict(data)
        for key in self.keys:
            array = d[key]
            
            # Convert to numpy array if not already
            if not isinstance(array, np.ndarray):
                array = np.array(array)
            
            # Ensure channel-first dimension for different input formats
            if array.ndim == 2:  # (H, W) → (1, H, W) 
                array = np.expand_dims(array, axis=0)
            elif array.ndim == 3 and array.shape[0] != 1:  # (H, W, C) → (C, H, W)
                array = np.transpose(array, (2, 0, 1))
            
            # Set appropriate data type - float32 for images, preserve original for labels
            if key == "image":
                array = array.astype(np.float32)
            
            d[key] = array
        return d
    

def get_2D_transforms():
    """
    Returns a tuple of two MONAI transform pipelines:
    1. augment_transforms: A pipeline for data augmentation including loading, resizing, intensity
    scaling, spatial augmentations, intensity augmentations, and conversion to tensor.
    2. no_aug_transforms: A pipeline for data preprocessing without augmentation, including loading,
    resizing, intensity scaling, and conversion to tensor.

    Both pipelines handle 2D images and labels, resizing them to a common size of (512, 512).
    """

    augment_transforms = Compose([
        LoadNumpy(keys=["image", "label"]),
        
        ResizeD(
            keys=["image", "label"],
            spatial_size=(512, 512),
            mode=["bilinear", "nearest"]
        ),
        
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=500,
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),

        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ToTensord(keys=["image", "label"]),
    ])

    no_aug_transforms = Compose([
        LoadNumpy(keys=["image", "label"]),
        # case_00160 has different image size, so we resize all images to a common size
        ResizeD(keys=["image", "label"], spatial_size=(512, 512), mode=["bilinear", "nearest"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=500,  # Clamp CT values
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),

        # Convert to tensor
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ToTensord(keys=["image", "label"]),
    ])

    return augment_transforms, no_aug_transforms

def get_2D_test_transforms():
    """
    Get transform pipeline for 2D test data preprocessing.
    
    Returns:
        Compose: MONAI transform pipeline for test data without augmentation.
                Includes data loading, resizing, intensity scaling, and tensor conversion.
    """

    test_transforms = Compose([
        # Prepare slice data by adding channel dimension and ensuring correct format
        PrepareSliceData(keys=["image"]),  
        # Resize images to standard size for model input
        ResizeD(keys=["image"], spatial_size=(512, 512), mode=["bilinear"]),
        # Scale CT intensity values to normalized range
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=500,  # Clamp CT Hounsfield unit values
            b_min=-1.0, b_max=1.0,   # Normalize to [-1, 1] range
            clip=True,
        ),

        # Ensure correct data type for PyTorch
        EnsureTyped(keys=["image"], dtype=torch.float32),
        # Convert numpy arrays to PyTorch tensors
        ToTensord(keys=["image"])
    ])

    return test_transforms


def get_3D_transforms():
    """
    Get transform pipelines for 3D data preprocessing and augmentation.
    
    Returns:
        tuple: (train_transforms, val_transforms) where:
            - train_transforms: MONAI pipeline with augmentation for training
            - val_transforms: MONAI pipeline without augmentation for validation
            
    Both pipelines include data loading, spacing normalization, orientation standardization,
    intensity scaling, foreground cropping, and tensor conversion optimized for 3D U-Net.
    """

    # Training transforms with enhanced augmentation
    train_transforms = Compose([
        # Load images and ensure channel first
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Resample to target spacing
        Spacingd(
            keys=["image", "label"], 
            pixdim=(3.22, 1.62, 1.62),
            mode=("bilinear", "nearest")
        ),
        
        # Orientation to standard (RAS+)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Intensity preprocessing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-79,
            a_max=304,
            b_min=-2.341,
            b_max=2.640,
            clip=True
        ),
        
        # Crop foreground to focus on kidney/tumor
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10
        ),
        
        # Ensure minimum size after cropping
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=(96, 192, 192),
            mode="constant"
        ),
        
        # Tumor-biased sampling
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(80, 160, 160),  # Common size
            ratios=[0.3, 3, 7],  # Weight background 0.3, tumor 3, kidney 7
            num_classes=3,
            num_samples=1,
            warn=False
        ),
        
        # Data augmentation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        
        RandRotated(
            keys=["image", "label"],
            range_x=15.0, # up to 15 degrees rotation
            range_y=15.0, 
            range_z=15.0,
            prob=0.5,
            mode=("bilinear", "nearest"),
            padding_mode="border",
            keep_size=True
        ),

        # Intensity augmentations
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.005),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        
        # Elastic deformation
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            prob=0.2,
            mode=("bilinear", "nearest")
        ),
        
        # Convert to tensor
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ToTensord(keys=["image", "label"])
    ])
    
    # Validation transforms (no augmentation, sliding window setup)
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        Spacingd(
            keys=["image", "label"],
            pixdim=(3.22, 1.62, 1.62),
            mode=("bilinear", "nearest")
        ),
        
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-79, 
            a_max=304,
            b_min=-2.341,
            b_max=2.640,
            clip=True
        ),
        
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10
        ),
        
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(80, 160, 160),
            mode="constant" 
        ),
        
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ToTensord(keys=["image", "label"])
    ])
    
    return train_transforms, val_transforms