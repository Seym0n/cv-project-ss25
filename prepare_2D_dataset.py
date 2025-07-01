import sys
import os
import numpy as np

# used to load the nifti-files
sys.path.append('/scratch/cv-course2025/lschind5/kits19/')
from starter_code.utils import load_case

# set to location of kits19 data
data_dir = "/scratch/cv-course2025/lschind5/kits19/data/"

cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
sorted_cases = sorted(cases, key=lambda x: int(x.split('_')[1]))

slice_counts = {}
for case in sorted_cases:
    try:
        volume, segmentation = load_case(case)
        slice_counts[case] = volume.shape[0]

        vol = volume.get_fdata(dtype=np.float32)
        seg = segmentation.get_fdata(dtype=np.float32)
        # convert to integer type
        seg = seg.astype(np.int8)

        # make sure images and labels match, otherwise unusable
        assert vol.shape == seg.shape, f"Volume and segmentation shapes do not match for case {case}: {vol.shape} vs {seg.shape}"

        if vol.shape[1] != 512 or vol.shape[2] != 512:
            print(f"Case {case} has unexpected dimensions: {vol.shape}.")
        
        path_to_case = os.path.join(data_dir, case)
        
        # make directories for images and labels
        os.makedirs(os.path.join(path_to_case, "images"), exist_ok=True)
        os.makedirs(os.path.join(path_to_case, "labels"), exist_ok=True)

        # save individual slices as numpy files
        for i in range(volume.shape[0]):
            image_slice = vol[i, :, :]
            label_slice = seg[i, :, :]

            image_path = os.path.join(path_to_case, "images", f"slice{i:04d}")
            label_path = os.path.join(path_to_case, "labels", f"slice{i:04d}")

            np.save(image_path, image_slice)
            np.save(label_path, label_slice)

    except Exception as e:
        print(f"Error processing case {case}: {e}")
