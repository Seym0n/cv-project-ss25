import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNETR
from monai.losses import DiceFocalLoss, DiceCELoss

from sklearn.model_selection import train_test_split

from utils.data_load import get_data_list, get_2D_data, get_2D_datasets
from utils.transforms import get_2D_transforms
from utils.train_utils import train_kits19_model


if __name__ == "__main__":
    print(f"🏆 Complete UNETR 2D Training with MONAI", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"MONAI: {monai.__version__}", flush=True)

    # Set reproducibility
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    DATA_ROOT = "/scratch/cv-course2025/lschind5/kits19/data"
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    NUM_WORKERS = 4

    # load data
    all_cases = get_data_list(DATA_ROOT)
    print(f"   Found {len(all_cases)} cases", flush=True)

    # Split data
    train_data, val_data = train_test_split(
        all_cases, test_size=0.2, random_state=42
    )

    train_cases = [case["case_id"] for case in train_data]
    val_cases = [case["case_id"] for case in val_data]

    train_list, val_list = get_2D_data(train_cases, val_cases, DATA_ROOT)
    print(f"   Training on {len(train_list)} cases, validating on {len(val_list)} cases", flush=True)

    # Get transforms
    train_transforms, val_transforms = get_2D_transforms()

    # Get datasets and loaders
    train_loader, val_loader, train_dataset, _ = get_2D_datasets(train_list, val_list, train_transforms, val_transforms, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # find class proportions
    background_pixel_counts = 0
    tumor_pixel_counts = 0
    kidney_pixel_counts = 0

    for data in train_dataset:
        image = data["image"]
        label = data["label"]
        
        background_pixel_counts += (label == 0).sum().item()
        tumor_pixel_counts += (label == 1).sum().item()
        kidney_pixel_counts += (label == 2).sum().item()

    total_pixels = background_pixel_counts + tumor_pixel_counts + kidney_pixel_counts
    background_proportion = background_pixel_counts / total_pixels
    tumor_proportion = tumor_pixel_counts / total_pixels
    kidney_proportion = kidney_pixel_counts / total_pixels
    print(f"   Background proportion: {background_proportion:.4f}", flush=True)
    print(f"   Tumor proportion: {tumor_proportion:.4f}", flush=True)
    print(f"   Kidney proportion: {kidney_proportion:.4f}", flush=True)
    

    loss_functions = [DiceFocalLoss, DiceCELoss]
    weights = [[0.3, 1, 3], [background_proportion**-1, kidney_proportion**-1, tumor_proportion**-1]]
    learning_rates= [1e-4, 3e-4, 1e-3]

    results_file = "unetr_hyperparameter_search_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Hyperparameter Search Results for UNETR 2D Training\n")
        f.write(f"Data Root: {DATA_ROOT}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Number of Workers: {NUM_WORKERS}\n")
        f.write(f"Device: {device}\n\n\n")
        f.write("Loss Function, Weights, Learning Rate, Train Loss, Val Loss, Kidney Dice, Tumor Dice\n")

    for loss_fn, weight, LR in zip(loss_functions, weights, learning_rates):
        weight_str = "_".join([f"{w:.2f}" for w in weight])
        print(f"Training with {loss_fn.__name__} and LR={LR} and weights={weight_str}", flush=True)

        # model setup
        model = UNETR(  # patch size fixed to 16x16 for 2D
            in_channels=1,
            out_channels=3, # background, kidney, tumor
            img_size=512,
            feature_size=32,
            norm_name='batch',
            spatial_dims=2).to(device)

        loss_fn_instance = loss_fn(
                to_onehot_y=True,  # convert target to one-hot format
                softmax=True,       # apply softmax to model outputs
                weight=torch.tensor(weight)  # Adjust weights for background, kidney, tumor
            ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        weight_str = "_".join([f"{w:.2f}" for w in weight])
        save_path = f"best_{loss_fn.__name__}_{weight_str}_{LR}.pth"
        train_losses, val_losses, kidney_dices, tumor_dices = train_kits19_model(
            model, loss_fn_instance, optimizer, train_loader, val_loader, device, NUM_EPOCHS, save_path=save_path
        )
        print(f"   Finished training with {loss_fn.__name__} and LR={LR} and weights={weight_str}", flush=True)

        # Save results
        best_tumor_index = tumor_dices.index(max(tumor_dices))
        best_tumor_dice = tumor_dices[best_tumor_index]
        best_val_loss = val_losses[best_tumor_index]
        best_kidney_dice = kidney_dices[best_tumor_index]
        best_train_loss = train_losses[best_tumor_index]

        with open(results_file, "a") as f:
            f.write(f"{loss_fn.__name__}, {weight_str}, {LR}, {best_train_loss:.4f}, {best_val_loss:.4f}, {best_kidney_dice:.4f}, {best_tumor_dice:.4f}\n")

        # clear memory
        del model, loss_fn_instance, optimizer
        torch.cuda.empty_cache()