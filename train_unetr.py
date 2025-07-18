import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss

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
    NUM_EPOCHS = 25
    NUM_WORKERS = 4
    LR = 3e-5

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
    train_loader, val_loader, _, _ = get_2D_datasets(train_list, val_list, train_transforms, val_transforms, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # model setup
    model = UNETR(  # patch size fixed to 16x16 for 2D
        in_channels=1,
        out_channels=3, # background, kidney, tumor
        img_size=512,
        feature_size=16,
        norm_name='batch',
        spatial_dims=2,
        dropout_rate=0.1, # help with overfitting
        ).to(device)

    loss_fn_instance = DiceCELoss(
            to_onehot_y=True,  # convert target to one-hot format
            softmax=True,       # apply softmax to model outputs
            weight=torch.tensor([0.3, 1, 3])  # Adjust weights for background, kidney, tumor
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses, val_losses, kidney_dices, tumor_dices = train_kits19_model(
        model, loss_fn_instance, optimizer, train_loader, val_loader, device, NUM_EPOCHS, save_path="best0307.pth",
        type="2d-vit", use_wandb=False, scheduler_warmup_steps= max(3, int(0.1 * NUM_EPOCHS)), scheduler_cycles=0.5
        )