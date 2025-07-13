import torch
import numpy as np
import nibabel as nib
from monai.inferers import sliding_window_inference
import scipy.ndimage as ndi


def get_case_predictions(model, case_dataset, device):
    """
    Get predictions for a single case using the provided model for 2D slice-by-slice inference.
    
    Args:
        model (torch.nn.Module): The trained model to use for predictions.
        case_dataset (DataLoader): The dataset containing the case data with 2D slices.
        device (torch.device): The device to run the model on (CPU or GPU).
    
    Returns:
        torch.Tensor: Post-processed prediction volume of shape (slices, H, W) with class labels.
    """
    # Set model to evaluation mode
    model.eval()
    prediction_volume = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Process each batch (slice) in the dataset
        for batch in case_dataset:
            # Move input data to the specified device
            inputs = batch["image"].to(device)
            # Forward pass through the model
            outputs = model(inputs)

            # Apply softmax to get class probabilities
            outputs_soft = torch.softmax(outputs, dim=1)
            # Get predicted class labels using argmax
            predictions = torch.argmax(outputs_soft, dim=1, keepdim=True)

            # Move predictions back to CPU and store
            prediction_volume.append(predictions.cpu())

    # Concatenate all slice predictions and reorder dimensions
    vol = torch.cat(prediction_volume, dim=0).permute(1, 0, 2, 3)
    # Apply post-processing to remove small artifacts
    post_processed = post_process_prediction(vol.squeeze(0))

    return post_processed

def get_case_predictions_3d(model, case_dataset, device):
    """
    Get predictions for a 3D case using sliding window inference.
    
    Args:
        model (torch.nn.Module): The trained 3D model to use for predictions.
        case_dataset (DataLoader): The dataset containing the 3D volume data.
        device (torch.device): The device to run the model on (CPU or GPU).
    
    Returns:
        torch.Tensor: Post-processed prediction volume of shape (slices, H, W) with class labels.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Process the single 3D volume in the dataset
        for batch_data in case_dataset:
            # Move 3D volume to the specified device
            inputs = batch_data["image"].to(device)
            
            # Use sliding window inference for memory-efficient 3D processing
            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=(80, 160, 160),  # Size of sliding window patches
                sw_batch_size=1,          # Number of patches processed simultaneously
                overlap=0.25,             # Overlap between adjacent patches
                predictor=model           # The 3D model to use for prediction
            )
            
            # Convert logits to class probabilities and then to predicted labels
            outputs_soft = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs_soft, dim=1, keepdim=True)
            
            # Remove batch dimension and move to CPU: [1, 1, D, H, W] -> [1, D, H, W]
            vol = predictions.cpu().squeeze(0)  # Output Shape: [1, D, H, W]
            
            # Apply post-processing to remove small artifacts
            post_processed = post_process_prediction(vol.squeeze(0))
            
            return post_processed

def evaluate_predictions(val_data, exclude_false_positives=False, slice_wise=False, exclude_background_slices=False):
    """
    Evaluate the predictions against the ground truth for validation data.
    
    Args:
        val_data: A dictionary containing validation data with case IDs, images, ground truths, and predictions.
        exclude_false_positives: If True, exclude false positive predictions from dice score calculation.
                               This means only calculating dice on regions where ground truth has foreground pixels.
        slice_wise: If True, calculate DICE scores per slice (2D) and accumulate across all slices.
                   If False, calculate DICE scores per case (3D volume).
        exclude_background_slices: If True, exclude slices that contain only background (no kidney or tumor)
                                 from dice score calculation. This removes slices 'above' and 'below'
                                 the kidney and tumor in CT scans. Most effective with slice_wise=True.
    
    Returns:
        Average dice score for kidney, tumor, and background classes.
    """
    metrics = {}
    # Example metric calculation (DICE score)
    kidney_dice_scores = []
    tumor_dice_scores = []
    bg_dice_scores = []

    for case_id, case_data in val_data.items():
        if isinstance(case_data["ground_truth"], nib.Nifti1Image):
            ground_truth = torch.from_numpy(case_data["ground_truth"].get_fdata())
        else:
            ground_truth = case_data["ground_truth"]

        predictions = case_data["predictions"]

        # Remove channel dimension from BOTH ground truth and predictions
        if predictions.ndim == 4:
            predictions = predictions.squeeze(0)
        
        # Remove channel dimension from ground truth if present
        if ground_truth.ndim == 4:
            ground_truth = ground_truth.squeeze(0)
        

        def dice_score(pred, gt, exclude_fp=False):
            """Calculate DICE score with optional false positive exclusion."""
            if exclude_fp:
                # Only consider pixels where ground truth has the target class
                # This effectively removes false positives from the calculation
                mask = gt > 0  # Only calculate where ground truth is positive
                if mask.sum() == 0:
                    # If no positive ground truth pixels, return 1.0 if no predictions, 0.0 if predictions exist
                    return 1.0 if pred.sum() == 0 else 0.0
                
                # Apply mask to both prediction and ground truth
                pred_masked = pred * mask
                gt_masked = gt * mask
                
                intersection = (pred_masked * gt_masked).sum()
                # For denominator, only count true positives and false negatives (exclude false positives)
                denominator = gt_masked.sum() + pred_masked.sum()
            else:
                # Standard DICE calculation
                if gt.sum() == 0 and pred.sum() == 0:
                    return 1.0  # Perfect prediction of absence
                elif gt.sum() == 0:
                    return 0.0  # False positive
                else:
                    intersection = (pred * gt).sum()
                    denominator = pred.sum() + gt.sum()
            
            return (2.0 * intersection) / denominator if denominator > 0 else 0.0

        if slice_wise:
            # Calculate DICE scores per slice (2D) and accumulate
            num_slices = ground_truth.shape[0]
            
            kidney_dice_case = []
            tumor_dice_case = []
            bg_dice_case = []
            for slice_idx in range(num_slices):
                gt_slice = ground_truth[slice_idx]
                pred_slice = predictions[slice_idx]
                
                # Check if this slice should be excluded (background-only)
                if exclude_background_slices:
                    # Check if ground truth slice contains any foreground pixels (kidney=1 or tumor=2)
                    has_foreground = ((gt_slice == 1) | (gt_slice == 2)).any()
                    if not has_foreground:
                        continue  # Skip this slice
                
                # KIDNEY DICE: Include both kidney (1) and tumor (2) as foreground
                kidney_pred_slice = ((pred_slice == 1) | (pred_slice == 2)).float()
                kidney_label_slice = ((gt_slice == 1) | (gt_slice == 2)).float()
                
                kidney_dice = dice_score(kidney_pred_slice, kidney_label_slice, exclude_false_positives)
                kidney_dice_scores.append(kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice)
                kidney_dice_case.append(kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice)

                # TUMOR DICE: calculated following KiTS19 paper
                tumor_pred_slice = (pred_slice == 2).float()
                tumor_label_slice = (gt_slice == 2).float()
                
                tumor_dice = dice_score(tumor_pred_slice, tumor_label_slice, exclude_false_positives)
                tumor_dice_scores.append(tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice)
                tumor_dice_case.append(tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice)

                # Background DICE
                bg_pred_slice = (pred_slice == 0).float()
                bg_label_slice = (gt_slice == 0).float()
                
                bg_dice = dice_score(bg_pred_slice, bg_label_slice, exclude_false_positives)
                bg_dice_scores.append(bg_dice.item() if torch.is_tensor(bg_dice) else bg_dice)
                bg_dice_case.append(bg_dice.item() if torch.is_tensor(bg_dice) else bg_dice)

            case_data["kidney_dice"] = np.mean(kidney_dice_case) if kidney_dice_case else 0.0
            case_data["tumor_dice"] = np.mean(tumor_dice_case) if tumor_dice_case else 0.0
            case_data["bg_dice"] = np.mean(bg_dice_case) if bg_dice_case else 0.0

        else:
            # Calculate DICE scores per case (3D volume)
            # Apply background slice exclusion for case-wise evaluation if requested
            if exclude_background_slices:
                # Find slices that contain kidney (1) or tumor (2) in ground truth
                foreground_mask = (ground_truth == 1) | (ground_truth == 2)  # Shape: (slices, H, W)
                
                # Check which slices have foreground pixels (sum over H, W dimensions)
                slice_has_foreground = foreground_mask.sum(dim=(1, 2)) > 0  # Shape: (slices,)
                valid_slice_indices = torch.where(slice_has_foreground)[0]
                
                # Only evaluate slices that contain foreground
                if len(valid_slice_indices) > 0:
                    ground_truth = ground_truth[valid_slice_indices]  # Select valid slices
                    predictions = predictions[valid_slice_indices]
                else:
                    # Skip this case if no foreground slices found
                    print(f"Warning: No foreground slices found in case {case_id}")
                    continue
            
            # KIDNEY DICE: Include both kidney (1) and tumor (2) as foreground
            kidney_pred = ((predictions == 1) | (predictions == 2)).float()
            kidney_label = ((ground_truth == 1) | (ground_truth == 2)).float()

            kidney_dice = dice_score(kidney_pred, kidney_label, exclude_false_positives)
            kidney_dice_scores.append(kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice)

            # TUMOR DICE: Always calculated, following KiTS19 protocol
            tumor_pred = (predictions == 2).float()
            tumor_label = (ground_truth == 2).float()

            tumor_dice = dice_score(tumor_pred, tumor_label, exclude_false_positives)
            tumor_dice_scores.append(tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice)

            # Background DICE
            bg_pred = (predictions == 0).float()
            bg_label = (ground_truth == 0).float()

            bg_dice = dice_score(bg_pred, bg_label, exclude_false_positives)
            bg_dice_scores.append(bg_dice.item() if torch.is_tensor(bg_dice) else bg_dice)

            case_data["kidney_dice"] = kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice
            case_data["tumor_dice"] = tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice
            case_data["bg_dice"] = bg_dice.item() if torch.is_tensor(bg_dice) else bg_dice

    metrics["kidney_dice"] = np.mean(kidney_dice_scores)
    metrics["tumor_dice"] = np.mean(tumor_dice_scores)
    metrics["bg_dice"] = np.mean(bg_dice_scores)
    
    if exclude_false_positives:
        metrics["note"] = "False positives excluded from DICE calculation"
    
    if exclude_background_slices:
        if "note" in metrics:
            metrics["note"] += "; Background-only slices excluded from calculation"
        else:
            metrics["note"] = "Background-only slices excluded from calculation"
    
    if slice_wise:
        metrics["evaluation_mode"] = "slice-wise (2D)"
    else:
        metrics["evaluation_mode"] = "case-wise (3D)"

    return metrics, val_data



def post_process_prediction(prediction, min_kidney_size=5000):
    """
    Post-process the prediction mask by removing small connected components
    for the kidney class (class 1) that are too small to be (part of) kidneys.

    Args:
        prediction (torch.Tensor or np.ndarray): Prediction mask of shape (slices, H, W) with class labels.
        min_kidney_size (int): Minimum number of voxels for a kidney component to be kept.

    Returns:
        torch.Tensor or np.ndarray: Post-processed prediction mask with same shape and type as input.
    """
    # Convert tensor to numpy if needed for processing
    if torch.is_tensor(prediction):
        pred_np = prediction.cpu().numpy()
    else:
        pred_np = prediction

    # Only process kidney class (class 1) since tumors are less predictable in size
    kidney_mask = (pred_np == 1)

    # Find connected components in the kidney mask using 3D connectivity
    labeled, num = ndi.label(kidney_mask)
    
    # Create binary mask for components that should be kept
    keep_mask = np.zeros_like(kidney_mask, dtype=bool)
    
    if num > 0:
        # Calculate the size (number of voxels) of each connected component
        sizes = ndi.sum(kidney_mask, labeled, range(1, num + 1))
        
        # Keep only components that meet the minimum size threshold
        for i in range(num):
            if sizes[i] >= min_kidney_size:
                keep_mask |= (labeled == (i + 1))

    # Create processed prediction by removing small kidney components
    processed = pred_np.copy()
    processed[(pred_np == 1)] = 0  # Remove all original kidney labels
    processed[keep_mask] = 1       # Restore only the large kidney components

    # Convert back to original tensor type and device if input was a tensor
    if torch.is_tensor(prediction):
        return torch.from_numpy(processed).to(prediction.device)
    return processed