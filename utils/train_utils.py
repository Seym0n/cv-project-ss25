import torch
import numpy as np
import json
import os
from accelerate import Accelerator
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule
from monai.inferers import sliding_window_inference
try:
    import wandb
except ImportError:
    wandb = None


def validate(model, val_loader, loss_function, device, type="2d-vit"):
    """
    Validation function with DICE calculation for KiTS19 challenge.
    
    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        loss_function (callable): Loss function to compute validation loss.
        device (torch.device): Device to run validation on (CPU or GPU).
        type (str): Model type, either "2d-vit" or "3d-unet".
    
    Returns:
        tuple: (val_loss, kidney_dice_avg, tumor_dice_avg) where:
            - val_loss: Average validation loss across all batches
            - kidney_dice_avg: Average kidney DICE score (includes kidney+tumor)
            - tumor_dice_avg: Average tumor DICE score
    """
    # Set model to evaluation mode
    model.eval()
    val_loss = 0
    val_batches = 0

    # Initialize lists to store DICE scores for each class
    kidney_dice_scores = []
    tumor_dice_scores = []

    # Statistics tracking
    cases_with_tumors = 0
    cases_with_false_positive_tumors = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # Move data to device
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            # Forward pass based on model type
            if type == "2d-vit":
                # Direct forward pass for 2D Vision Transformer models
                outputs = model(inputs)

            if type == "3d-unet":
                # Use sliding window inference for 3D U-Net models
                roi_size = (80, 160, 160)
                sw_batch_size = 1

                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.25
                )

            # Compute validation loss
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            val_batches += 1

            # Convert model outputs to class predictions using argmax
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

            # Process each sample in the batch for DICE calculation
            for batch_sample in range(predictions.shape[0]):
                pred_sample = predictions[batch_sample].squeeze(0)  # [H, W, D]
                label_sample = labels[batch_sample].squeeze(0)      # [H, W, D]

                # KIDNEY DICE: Include both kidney (1) and tumor (2) as foreground (as in KiTS19 challenge)
                kidney_pred = ((pred_sample == 1) | (pred_sample == 2)).float()
                kidney_label = ((label_sample == 1) | (label_sample == 2)).float()

                def dice_score(pred, gt):
                    """Calculate DICE score with proper edge case handling."""
                    if gt.sum() == 0 and pred.sum() == 0:
                        return 1.0  # Perfect prediction when both are empty
                    elif gt.sum() == 0:
                        return 0.0  # False positive case
                    else:
                        intersection = (pred * gt).sum()
                        return (2.0 * intersection) / (pred.sum() + gt.sum())

                # Calculate kidney DICE score
                kidney_dice = dice_score(kidney_pred, kidney_label)
                kidney_dice_scores.append(kidney_dice.item() if torch.is_tensor(kidney_dice) else kidney_dice)

                # TUMOR DICE: Only tumor class (2) as foreground
                tumor_pred = (pred_sample == 2).float()
                tumor_label = (label_sample == 2).float()
                
                # Calculate tumor DICE score
                tumor_dice = dice_score(tumor_pred, tumor_label)
                tumor_dice_scores.append(tumor_dice.item() if torch.is_tensor(tumor_dice) else tumor_dice)
                
                # Track tumor statistics for monitoring
                if tumor_label.sum() > 0:
                    cases_with_tumors += 1
                if tumor_label.sum() == 0 and tumor_pred.sum() > 0:
                    cases_with_false_positive_tumors += 1
                
                # Debug output for first few samples
                if batch_idx < 2:
                    print(f"   Sample {batch_idx}: Kidney DICE={kidney_dice:.3f}, Tumor DICE={tumor_dice:.3f}")
                    print(f"     Tumor present: {tumor_label.sum() > 0}, Tumor predicted: {tumor_pred.sum() > 0}")
    

    # Calculate mean DICE scores
    kidney_dice_avg = np.mean(kidney_dice_scores)
    tumor_dice_avg = np.mean(tumor_dice_scores)

    print(f"   Validation samples with kidney: {len(kidney_dice_scores)}", flush=True)
    print(f"   Validation samples with tumor: {len(tumor_dice_scores)}", flush=True)

    return val_loss / val_batches, kidney_dice_avg, tumor_dice_avg

def train_epoch(model, train_loader, optimizer, loss_fn, device, accelerator):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function to compute training loss.
        device (torch.device): Device to run training on (CPU or GPU).
        accelerator (Accelerator): Hugging Face Accelerator for mixed precision training.
    
    Returns:
        float: Average training loss for the epoch.
    """

    # Set model to training mode
    model.train()
    epoch_loss = 0
    batch_count = 0

    # Iterate through training batches
    for batch_data in train_loader:
        # Move data to device and ensure correct data types
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].long().to(device)

        # Forward pass through the model
        outputs = model(inputs)
        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Backward pass using accelerator for mixed precision
        accelerator.backward(loss)
        # Update model parameters
        optimizer.step()
        # Clear gradients for next iteration
        optimizer.zero_grad()

        # Track loss statistics
        epoch_loss += loss.item()
        batch_count += 1

        # Print progress every 1000 batches
        if batch_count % 1000 == 0:
            print(f"Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)

    # Return average loss for the epoch
    return epoch_loss / batch_count


def save_progress_to_json(progress_file, epoch, train_loss, val_loss, kidney_dice, tumor_dice, learning_rate):
    """
    Save training progress to JSON file for tracking and visualization.
    
    Args:
        progress_file (str): Path to the JSON file to save progress.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        kidney_dice (float): Kidney DICE score for the current epoch.
        tumor_dice (float): Tumor DICE score for the current epoch.
        learning_rate (float): Current learning rate.
    
    Returns:
        None
    """
    
    # Load existing progress data if file already exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        # Initialize new progress tracking structure
        progress = {
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
            "kidney_dice_scores": [],
            "tumor_dice_scores": [],
            "learning_rates": []
        }
    
    # Append current epoch's metrics to the tracking lists
    progress["epochs"].append(epoch)
    progress["train_losses"].append(train_loss)
    progress["val_losses"].append(val_loss)
    progress["kidney_dice_scores"].append(kidney_dice)
    progress["tumor_dice_scores"].append(tumor_dice)
    progress["learning_rates"].append(learning_rate)
    
    # Save the updated progress data back to the JSON file
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def train_kits19_model(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs=100, save_path="best_kits19_model.pth", type="2d-vit", progress_file="training_progress.json", use_wandb=False, wandb_project="kits19-segmentation", wandb_config=None, wandb_notes=None, scheduler_warmup_steps=25, scheduler_cycles=3.5):
    """
    Training loop following KiTS19 winning methodology.
    Args:
        model: The model to train.
        loss_fn: Loss function to use.
        optimizer: Optimizer for training.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the training on (CPU or GPU).
        num_epochs: Number of epochs to train.
        save_path: Path to save the best model.
        type: Type of model (e.g., "2d-vit", "3d-unet").
        progress_file: Path to save training progress JSON file.
        use_wandb: Whether to use wandb for logging (default: False).
        wandb_project: Name of the wandb project (default: "kits19-segmentation").
        wandb_config: Configuration dictionary for wandb (default: None).
        wandb_notes: Notes to add to wandb run (default: None).
        scheduler_warmup_steps: Number of warmup steps for the scheduler (default: 25).
        scheduler_cycles: Number of cycles for the cosine scheduler (default: 3.5).
    """

    accelerator = Accelerator(mixed_precision="fp16") # Use mixed precision for faster training
    
    # Initialize wandb if requested
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it with: pip install wandb")
        
        # Set up wandb config
        config = {
            "epochs": num_epochs,
            "model_type": type,
            "model_name": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "mixed_precision": "fp16",
            "warmup_steps": scheduler_warmup_steps,
            "scheduler_cycles": scheduler_cycles
        }
        
        # Update config with user-provided config
        if wandb_config:
            config.update(wandb_config)
        
        wandb.init(project=wandb_project, config=config, notes=wandb_notes)
        wandb.watch(model)

    # Prepare everything with accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=scheduler_warmup_steps,
        t_total=num_epochs,
        cycles=scheduler_cycles
    )

    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())

    print(f"   Model: {model_name}", flush=True)
    print(f"   Parameters: {param_count:,}", flush=True)
    print(f"   Epochs: {num_epochs}", flush=True)
    print(f"   Progress will be saved to: {progress_file}", flush=True)

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
            model, val_loader, loss_fn, device, type=type
        )
        val_losses.append(val_loss)
        kidney_dices.append(kidney_dice)
        tumor_dices.append(tumor_dice)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"📊 Results:", flush=True)
        print(f"   Train Loss: {train_loss:.4f}", flush=True)
        print(f"   Val Loss: {val_loss:.4f}", flush=True)
        print(f"   Kidney DICE: {kidney_dice:.4f}", flush=True)
        print(f"   Tumor DICE: {tumor_dice:.4f}", flush=True)
        print(f"   Learning Rate: {current_lr:.6f}", flush=True)
        
        # Log to wandb if enabled
        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "kidney_dice": kidney_dice,
                "tumor_dice": tumor_dice,
                "learning_rate": current_lr,
                "best_tumor_dice": best_tumor_dice
            })

        # Save progress to JSON file
        save_progress_to_json(
            progress_file=progress_file,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            kidney_dice=kidney_dice,
            tumor_dice=tumor_dice,
            learning_rate=current_lr
        )

        # Save best model based on tumor dice (as per challenge rules)
        if tumor_dice > best_tumor_dice:
            best_tumor_dice = tumor_dice
            torch.save(model.state_dict(), save_path)
            accelerator.save_model(model, save_path + '-accelerator.pth')
            print(f"🎯 New best model saved! Tumor DICE: {best_tumor_dice:.4f}", flush=True)
            
            # Log models to wandb if enabled
            if use_wandb and wandb is not None:
                wandb.log_model(path=save_path, name=f"best_model_epoch_{epoch+1}")
                wandb.log_model(path=save_path + '-accelerator.pth', name=f"best_model_accelerator_epoch_{epoch+1}")
        
        # Update scheduler
        scheduler.step()

    print(f"\n🏆 Training Completed!", flush=True)
    print(f"   Best Tumor DICE: {best_tumor_dice:.4f}", flush=True)
    print(f"   Model saved to: {save_path}", flush=True)
    print(f"   Training progress saved to: {progress_file}", flush=True)
    
    # Finish wandb run if enabled
    if use_wandb and wandb is not None:
        wandb.finish()

    return train_losses, val_losses, kidney_dices, tumor_dices