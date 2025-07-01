from monai.transforms import (
    MapTransform, ResizeD, Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ToTensord, CenterSpatialCropd, EnsureTyped,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandShiftIntensityd, SpatialPadd, CropForegroundd,
    RandSpatialCropd, Resized
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
        # case_00160 has different image size, so we resize all images to a common size
        ResizeD(keys=["image", "label"], spatial_size=(512, 512), mode=["bilinear", "nearest"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200, a_max=500,  # Clamp CT values
            b_min=-1.0, b_max=1.0,
            clip=True,
        ),

        # Spatial augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # vertical flip
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        # Intensity augmentations (image only)
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        # Convert to tensor
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