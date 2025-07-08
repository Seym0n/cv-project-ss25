from monai.transforms import (
    MapTransform, ResizeD, Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ToTensord, CenterSpatialCropd, EnsureTyped,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandShiftIntensityd, SpatialPadd, CropForegroundd,
    RandSpatialCropd, Resized, OneOf, RandCropByLabelClassesd, Rand3DElasticd
)
import numpy as np
import torch

class LoadNumpy(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            array = np.load(d[key])

            # ensure channel first dimension
            if array.ndim == 2:  # (H, W) → (1, H, W) 
                array = np.expand_dims(array, axis=0)
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


def get_3D_transforms():
    """
    Preprocessing pipeline for KiTS19 with augmentation and validation
    """
    print("Version 2 of 3D transforms")
    # Define possible patch sizes for multi-scale training
    patch_sizes = [(64, 128, 128), (80, 160, 160), (96, 192, 192)]
    
    # Training transforms with enhanced augmentation
    train_transforms = Compose([
        # 1. Load images and ensure channel first
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 2. Resample to target spacing
        Spacingd(
            keys=["image", "label"], 
            pixdim=(3.22, 1.62, 1.62),
            mode=("bilinear", "nearest")
        ),
        
        # 3. Orientation to standard (RAS+)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # 4. Intensity preprocessing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-79,
            a_max=304,
            b_min=-2.341,
            b_max=2.640,
            clip=True
        ),
        
        # 5. Ensure minimum size for largest patch
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(96, 192, 192),  # Max patch size
            mode="constant"
        ),
        
        # 6. Crop foreground to focus on kidney/tumor
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10
        ),
        
        # 7. Ensure minimum size after cropping
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=(96, 192, 192),
            mode="constant"
        ),
        
        # Direct Tumor-biased sampling
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(80, 160, 160),  # Default size
            ratios=[0.3, 3, 7],  # No background, kidney=30%, tumor=70%
            num_classes=3,
            num_samples=1,
            warn=False
        ),
        
        # 9. Random patch size variation (Step 3)
        OneOf([
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=size,
                random_size=False
            ) for size in patch_sizes
        ], weights=[0.3, 0.4, 0.3]),  # Equal-ish probability for patch sizes
        
        # 10. Enhanced data augmentation (Step 1)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        
        # Intensity augmentations
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),  # Increased range
        RandAdjustContrastd(keys=["image"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.005),  # Reduced noise
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        
        # Elastic deformation
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            prob=0.2,
            mode=("bilinear", "nearest")
        ),
        
        # 11. Convert to tensor
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