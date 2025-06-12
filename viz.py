import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib

def plot_segmentation_3d(segmentation):
    """
    Plots the 3D segmentation

    Parameters:
    segmentation (nifti): The 3D segmentation data in NIfTI format
    """

    segmentation = nib.as_closest_canonical(segmentation)
    spacing = segmentation.header.get_zooms()
    volume = segmentation.get_fdata(dtype=np.float32)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    format = {
        1: ('lightblue', 0.5),  # kidney
        2: ('orange', 1)         # tumor
    }

    for class_id, (color, alpha) in format.items():
        mask = (volume == class_id)
        if np.any(mask):  # only process if the class is present
            verts, faces, _, _ = measure.marching_cubes(mask, level=0, spacing=spacing)

            mesh = Poly3DCollection(verts[faces], alpha=alpha)
            mesh.set_facecolor(color)
            ax.add_collection3d(mesh)
    
    ax.set_xlabel("X (sagittal)")
    ax.set_ylabel("Y (coronal)")
    ax.set_zlabel("Z (axial slices)")
    
    # TODO: find best orientation, save image, formatting options
    ax.view_init(elev=15, azim=120)
    plt.tight_layout()
    plt.show()