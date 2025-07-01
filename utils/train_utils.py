import torch
import numpy as np
from accelerate import Accelerator
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule


def validate(model, val_loader, loss_function, dice_metric, device):
    """
    Fixed validation function with correct DICE calculation for KiTS19.
    Key fixes:
    1. Kidney DICE includes both kidney (1) and tumor (2) labels as foreground
    2. Uses argmax instead of thresholding softmax outputs
    3. Proper debugging output
    """
    model.eval()
    val_loss = 0
    val_batches = 0

    # Separate metrics for each class
    kidney_dice_scores = []
    tumor_dice_scores = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            val_batches += 1

            # Convert outputs to predictions using argmax
            outputs_soft = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs_soft, dim=1, keepdim=True)

            # Process each sample in batch
            for batch_sample in range(predictions.shape[0]):
                pred_sample = predictions[batch_sample]  # Shape: [1, H, W]
                label_sample = labels[batch_sample]      # Shape: [1, H, W]

                # Remove channel dimension for processing
                pred_sample = pred_sample.squeeze(0)  # Shape: [H, W]
                label_sample = label_sample.squeeze(0)  # Shape: [H, W]

                # KIDNEY DICE: Include both kidney (1) and tumor (2) as foreground
                kidney_pred = ((pred_sample == 1) | (pred_sample == 2)).float()
                kidney_label = ((label_sample == 1) | (label_sample == 2)).float()

                if kidney_label.sum() > 0:  # Only calculate if kidney present
                    # Calculate DICE manually
                    intersection = (kidney_pred * kidney_label).sum()
                    dice_score = (2.0 * intersection) / (kidney_pred.sum() + kidney_label.sum())
                    kidney_dice_scores.append(dice_score.item())

                # TUMOR DICE: Only tumor (class 2)
                tumor_pred = (pred_sample == 2).float()
                tumor_label = (label_sample == 2).float()

                if tumor_label.sum() > 0:  # Only calculate if tumor present
                    # Calculate DICE manually
                    intersection = (tumor_pred * tumor_label).sum()
                    dice_score = (2.0 * intersection) / (tumor_pred.sum() + tumor_label.sum())
                    tumor_dice_scores.append(dice_score.item())

    # Calculate mean DICE scores
    kidney_dice = np.mean(kidney_dice_scores) if kidney_dice_scores else 0.0
    tumor_dice = np.mean(tumor_dice_scores) if tumor_dice_scores else 0.0

    print(f"   Validation samples with kidney: {len(kidney_dice_scores)}", flush=True)
    print(f"   Validation samples with tumor: {len(tumor_dice_scores)}", flush=True)

    return val_loss / val_batches, kidney_dice, tumor_dice

def train_epoch(model, train_loader, optimizer, loss_fn, device, accelerator):
    """ Train the model for one epoch. """

    model.train()
    epoch_loss = 0
    batch_count = 0

    for batch_data in train_loader:

        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].long().to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        batch_count += 1

        if batch_count % 1000 == 0:
            print(f"Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)

    return epoch_loss / batch_count


def train_kits19_model(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs=100, save_path="best_kits19_model.pth"):
    """
    Complete training loop following KiTS19 winning methodology.
    Fixed version with proper variable names.
    """

    accelerator = Accelerator(mixed_precision="fp16") # Use mixed precision for faster training

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Prepare everything with accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps= max(3, int(0.1 * num_epochs)),
        t_total=num_epochs
    )

    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())

    print(f"   Model: {model_name}", flush=True)
    print(f"   Parameters: {param_count:,}", flush=True)
    print(f"   Epochs: {num_epochs}", flush=True)

    best_tumor_dice = 0
    train_losses, val_losses, kidney_dices, tumor_dices = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        print("-" * 50, flush=True)

        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, accelerator
        )
        train_losses.append(train_loss)

        # Validation
        val_loss, kidney_dice, tumor_dice = validate(
            model, val_loader, loss_fn, dice_metric, device
        )
        val_losses.append(val_loss)
        kidney_dices.append(kidney_dice)
        tumor_dices.append(tumor_dice)

        print(f"📊 Results:", flush=True)
        print(f"   Train Loss: {train_loss:.4f}", flush=True)
        print(f"   Val Loss: {val_loss:.4f}", flush=True)
        print(f"   Kidney DICE: {kidney_dice:.4f}", flush=True)
        print(f"   Tumor DICE: {tumor_dice:.4f}", flush=True)

        # Save best model based on tumor dice (as per challenge rules)
        if tumor_dice > best_tumor_dice:
            best_tumor_dice = tumor_dice
            torch.save(model.state_dict(), save_path)
            print(f"🎯 New best model saved! Tumor DICE: {best_tumor_dice:.4f}", flush=True)
        
        # Update scheduler
        scheduler.step()

    print(f"\n🏆 Training Completed!", flush=True)
    print(f"   Best Tumor DICE: {best_tumor_dice:.4f}", flush=True)
    print(f"   Model saved to: {save_path}", flush=True)

    return train_losses, val_losses, kidney_dices, tumor_dices