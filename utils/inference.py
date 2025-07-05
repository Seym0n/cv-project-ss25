import torch
import numpy as np


def get_case_predictions(model, case_dataset, device):
    """
    Get predictions for a single case using the provided model.
    
    Args:
        model: The trained model to use for predictions.
        case_dataset: The dataset containing the case data.
        device: The device to run the model on (CPU or GPU).
    
    Returns:
        A dictionary with the case ID and the predicted segmentation.
    """
    model.eval()
    prediction_volume = []
    with torch.no_grad():
        for batch in case_dataset:

            inputs = batch["image"].to(device)
            outputs = model(inputs)

            outputs_soft = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs_soft, dim=1, keepdim=True)

            prediction_volume.append(predictions.cpu())
    
    return np.concatenate(prediction_volume, axis=0)


def evaluate_predictions(val_data):
    """
    Evaluate the predictions against the ground truth for validation data.
    
    Args:
        val_data: A dictionary containing validation data with case IDs, images, ground truths, and predictions.
    
    Returns:
        Average dice score for kidney, tumor, and background classes.
    """
    metrics = {}
    # Example metric calculation (DICE score)
    kidney_dice_scores = []
    tumor_dice_scores = []
    bg_dice_scores = []

    for case_id, case_data in val_data.items():
        ground_truth = case_data["ground_truth"]
        predictions = case_data["predictions"]
        
        # KIDNEY DICE: Include both kidney (1) and tumor (2) as foreground
        kidney_pred = ((predictions == 1) | (predictions == 2)).float()
        kidney_label = ((ground_truth == 1) | (ground_truth == 2)).float()

        def dice_score(pred, gt):
            """Calculate DICE score with proper edge case handling."""

            if gt.sum() == 0 and pred.sum() == 0:
                return 1.0  # Perfect prediction of absence
            elif gt.sum() == 0:
                return 0.0  # False positive
            else:
                intersection = (pred * gt).sum()
                return (2.0 * intersection) / (pred.sum() + gt.sum())

        

        kidney_dice = dice_score(kidney_pred, kidney_label)
        kidney_dice_scores.append(kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice)

        # TUMOR DICE: Always calculated, following KiTS19 protocol
        tumor_pred = (predictions == 2).float()
        tumor_label = (ground_truth == 2).float()

        tumor_dice = dice_score(tumor_pred, tumor_label)
        tumor_dice_scores.append(tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice)

        # Background DICE
        bg_pred = (predictions == 0).float()
        bg_label = (ground_truth == 0).float()

        bg_dice = dice_score(bg_pred, bg_label)
        bg_dice_scores.append(bg_dice.item() if torch.is_tensor(bg_dice) else bg_dice)


    metrics["kidney_dice"] = np.mean(kidney_dice_scores)
    metrics["tumor_dice"] = np.mean(tumor_dice_scores)
    metrics["bg_dice"] = np.mean(bg_dice_scores)

    return metrics