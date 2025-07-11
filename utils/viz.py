from unittest import case
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
from skimage import measure
from pathlib import Path

def plot_segmentation_comparison(segmentation, prediction, tumor_dice, kidney_dice, case, figsize=(25, 15), output_path=None, model_type="2d"):
    """
    Create side-by-side comparison of ground truth and prediction
    
    Parameters:
    segmentation: Ground truth segmentation
    prediction: Predicted segmentation
    figsize: Figure size for the comparison plot
    model_type: "2d" for 2D UNETR, "3d" for 3D U-Net (for future orientation fixes)
    """
    
    fig = plt.figure(figsize=figsize)
    
    # Ground truth subplot
    ax1 = fig.add_subplot(121, projection='3d')
    axis_limits = plot_single_segmentation(segmentation, ax1, title="Ground Truth", 
                           format_dict={1: ('lightgray', 0.2), 2: ('orangered', 0.8)},
                           model_type=model_type)

    # Prediction subplot
    ax2 = fig.add_subplot(122, projection='3d')
    plot_single_segmentation(segmentation, ax2, title="Prediction",
                           format_dict={1: ('lightgray', 0.2), 2: ('orangered', 0.8)}, 
                           prediction=prediction, axis_limits=axis_limits,
                           model_type=model_type)

    fig.text(0.5, 0.88, f"Segmentation Comparison {case} | Tumor Dice: {tumor_dice:.2f} | Kidney Dice: {kidney_dice:.2f}", 
             fontsize=35, ha='center', va='top')
    
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"{case}_segmentation_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_single_segmentation(segmentation, ax, title="", format_dict=None, prediction=None, axis_limits=None, model_type="2d"):
    """
    Helper function to plot a single segmentation on a given axis
    
    Args:
        model_type: "2d" for 2D UNETR, "3d" for 3D U-Net (for future orientation fixes)
    """
    if format_dict is None:
        format_dict = {1: ('blue', 0.6), 2: ('red', 0.8)}
    
    # Process data
    if hasattr(segmentation, 'get_fdata'):
        # segmentation = nib.as_closest_canonical(segmentation)
        spacing = segmentation.header.get_zooms()
        volume = segmentation.get_fdata(dtype=np.float32)
        # volume = np.transpose(volume, (2, 0, 1))  # Transpose to match (Z, Y, X) order
    else:
        volume = segmentation
        spacing = (1, 1, 1)

    if prediction is not None:
        volume = prediction

    # Model-specific orientation handling
    if model_type == "3d":
        volume = np.transpose(volume, (2, 0, 1))
        volume = np.flip(volume, axis=0)
        volume = np.flip(volume, axis=1)
        volume = np.flip(volume, axis=2)
        spacing = (spacing[2], spacing[0], spacing[1])
        
    else:  # model_type == "2d"
        print("Applying 2D UNETR orientation...")
        volume = np.transpose(volume, (1, 2, 0))  # (Y, X, Z)
        spacing = (spacing[1], spacing[2], spacing[0])

    # Plot each class
    for class_id, (color, alpha) in format_dict.items():
        mask = (volume == class_id)
        if np.any(mask):
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
                mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.5)
                mesh.set_facecolor(color)
                mesh.set_edgecolor('black')
                ax.add_collection3d(mesh)
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not generate mesh for class {class_id}: {e}")
    
    # Set axis limits if provided
    if axis_limits is not None:
        ax.set_xlim(axis_limits['xlim'])
        ax.set_ylim(axis_limits['ylim'])
        ax.set_zlim(axis_limits['zlim'])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (axial slices)")
    
    ax.text2D(0.5, 0.1, title, transform=ax.transAxes, 
            fontsize=25, ha='center', va='top')

    # ax.set_title(title, fontsize=25)
    # ax.title.set_position([0.5, 0.7])
    ax.view_init(elev=15, azim=150)
    ax.grid(True, alpha=0.3)

    return {
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
        'zlim': ax.get_zlim()
    }


def plot_predictions_3D(random_val_data, output_path=None, model_type="2d"):
    """
    Plot 3D predictions with proper tensor/numpy handling for both 2D UNETR and 3D U-Net
    
    Args:
        random_val_data: Dictionary containing validation data
        output_path: Path to save plots
        model_type: "2d" for 2D UNETR, "3d" for 3D U-Net
    """
    for case_id, case_data in random_val_data.items():
        image = case_data["image"]
        ground_truth = case_data["ground_truth"]
        predictions = case_data["predictions"]
        
        # Handle ground_truth: convert torch tensor to numpy if needed
        if hasattr(ground_truth, 'numpy'):
            # It's a torch tensor
            ground_truth_np = ground_truth.numpy()
        elif hasattr(ground_truth, 'get_fdata'):
            # It's a NIfTI image
            ground_truth_np = ground_truth.get_fdata()
        else:
            # Already numpy
            ground_truth_np = ground_truth
        
        # Handle predictions: convert torch tensor to numpy if needed  
        if hasattr(predictions, 'numpy'):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions
        
        # Remove channel dimension if present (shape might be [1, D, H, W])
        if ground_truth_np.ndim == 4 and ground_truth_np.shape[0] == 1:
            ground_truth_np = ground_truth_np.squeeze(0)  # Remove channel dim
        
        if predictions_np.ndim == 4 and predictions_np.shape[0] == 1:
            predictions_np = predictions_np.squeeze(0)  # Remove channel dim
        
        # Get DICE scores
        tumor_dice = case_data.get("tumor_dice", 0.0)
        kidney_dice = case_data.get("kidney_dice", 0.0)
        
        print(f"Plotting {case_id}: GT shape {ground_truth_np.shape}, Pred shape {predictions_np.shape}")
        
        plot_segmentation_comparison(
            ground_truth_np, predictions_np, tumor_dice, kidney_dice, case_id, 
            output_path=output_path, model_type=model_type
        )


def plot_test_slices(
    test_data,
    output_path = None,
    num_slices = 3,
    figsize = (15, 5),
    slice_selection = 'middle',
    show_contours = True,
    show_overlays = True,
    overlay_alpha = 0.4,
    contour_width = 2.0,
    dpi = 300
) -> None:
    """
    Plot prediction slices for KiTS19 dataset with enhanced visualization options.
    
    Parameters:
    -----------
    test_data : Dict
        Dictionary containing case data with 'image' and 'predictions' keys
    output_path : str or Path, optional
        Path to save the plot. If None, plot is only displayed
    num_slices : int, default=3
        Number of slices to display per case
    figsize : tuple, default=(15, 5)
        Figure size (width, height) in inches
    slice_selection : str, default='middle'
        How to select slices: 'middle', 'evenly_spaced', or 'random'
    show_contours : bool, default=True
        Whether to show contour lines around segmented regions
    show_overlays : bool, default=True
        Whether to show colored overlays for segmented regions
    overlay_alpha : float, default=0.4
        Transparency of overlay masks (0.0 to 1.0)
    contour_width : float, default=2.0
        Width of contour lines
    dpi : int, default=300
        Resolution for saved images
    """
    
    # Color scheme for different classes
    class_colors = {
        1: {'overlay': 'cool', 'contour': 'royalblue', 'label': 'Kidney'},
        2: {'overlay': 'autumn', 'contour': 'orangered', 'label': 'Tumor'}
    }
    
    for case_id, case_data in test_data.items():
        try:
            # Extract data
            image = case_data["image"]
            predictions = case_data["predictions"]
            
            # Convert predictions to numpy if needed
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            
            # Get image data
            if hasattr(image, 'get_fdata'):
                image_data = image.get_fdata()
            else:
                image_data = image
            
            
            # Select slice indices
            total_slices = predictions.shape[0]

            slice_indices = _select_slices(total_slices, num_slices, slice_selection)
            
            # Create figure
            fig, axes = plt.subplots(1, len(slice_indices), figsize=figsize)
            if len(slice_indices) == 1:
                axes = [axes]
            
            fig.suptitle(f'Test: {case_id} - Prediction Visualization', fontsize=16, y=0.98)
            
            # Plot each slice
            for ax, idx in zip(axes, slice_indices):
                img_slice = image_data[idx]
                pred_slice = predictions[idx]
                
                # Normalize image for display
                img_normalized = _normalize_image(img_slice)
                
                # Display base image
                ax.imshow(img_normalized, cmap='gray', aspect='equal')
                
                # Add overlays and contours for each class
                for cls in [1, 2]:
                    if cls not in class_colors:
                        continue
                    
                    mask = (pred_slice == cls)
                    
                    if not np.any(mask):
                        continue
                    
                    # Add colored overlay
                    if show_overlays:
                        overlay_cmap = plt.colormaps.get_cmap(class_colors[cls]['overlay'])
                        ax.imshow(
                            np.ma.masked_where(~mask, mask),
                            cmap=overlay_cmap,
                            alpha=overlay_alpha,
                            aspect='equal'
                        )
                    
                    # Add contours
                    if show_contours:
                        try:
                            contours = measure.find_contours(mask.astype(float), 0.5)
                            for contour in contours:
                                ax.plot(
                                    contour[:, 1], contour[:, 0],
                                    linewidth=contour_width,
                                    color=class_colors[cls]['contour'],
                                    alpha=0.8
                                )
                        except ValueError:
                            # Skip if contour finding fails
                            pass
                
                # Formatting
                ax.set_title(f"Slice {idx}", fontsize=12, pad=5)
                ax.axis('off')
                
                # Add slice statistics
                kidney_pixels = np.sum(pred_slice == 1)
                tumor_pixels = np.sum(pred_slice == 2)
                ax.text(0.02, 0.98, f'K: {kidney_pixels}\nT: {tumor_pixels}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Create legend
            legend_patches = []
            for cls in [1, 2]:
                if cls in class_colors:
                    legend_patches.append(
                        mpatches.Patch(
                            color=class_colors[cls]['contour'],
                            label=class_colors[cls]['label']
                        )
                    )
            
            if legend_patches:
                fig.legend(
                    handles=legend_patches,
                    loc='lower center',
                    ncol=len(legend_patches),
                    bbox_to_anchor=(0.5, -0.05),
                    fontsize=12
                )
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            # Save plot
            if output_path:
                    output_path_dir = Path(output_path)
                    output_path_dir.mkdir(parents=True, exist_ok=True)
                    filename = output_path_dir / f"{case_id}_slice_plots.png"
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight')

            plt.show()
            
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            continue

def _select_slices(total_slices, num_slices, selection_method):
    """Select slice indices based on the specified method."""
    if selection_method == 'middle':
        # Select slices around the middle
        center = total_slices // 2
        if num_slices == 1:
            return [center]
        elif num_slices == 2:
            return [center - 1, center + 1]
        else:
            # For 3 or more slices, distribute around center
            half_range = (num_slices - 1) // 2
            start = max(0, center - half_range)
            end = min(total_slices - 1, center + half_range)
            return list(range(start, end + 1))[:num_slices]
    
    elif selection_method == 'evenly_spaced':
        # Evenly spaced slices across the volume
        if num_slices == 1:
            return [total_slices // 2]
        else:
            indices = np.linspace(0, total_slices - 1, num_slices + 2, dtype=int)
            return indices[1:-1].tolist()  # Exclude first and last
    
    elif selection_method == 'random':
        # Random selection
        np.random.seed(42)  # For reproducibility
        return sorted(np.random.choice(total_slices, num_slices, replace=False))
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

def _normalize_image(image):
    """Normalize image for display."""
    if image.dtype == np.uint8:
        return image
    
    # Clip extreme values (optional)
    p1, p99 = np.percentile(image, (1, 99))
    image_clipped = np.clip(image, p1, p99)
    
    # Normalize to [0, 1]
    img_min, img_max = image_clipped.min(), image_clipped.max()
    if img_max > img_min:
        return (image_clipped - img_min) / (img_max - img_min)
    else:
        return image_clipped