import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNETR
from monai.losses import DiceFocalLoss
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from sklearn.model_selection import train_test_split

from utils.data_load import get_3D_dataset, get_data_list, get_2D_data, get_2D_datasets
from utils.transforms import get_2D_transforms, get_3D_transforms
from utils.train_utils import train_kits19_model


if __name__ == "__main__":
    print(f"🏆 Complete Residual 3D-Unet Training with MONAI", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"MONAI: {monai.__version__}", flush=True)

    # Set reproducibility
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    DATA_ROOT = "/scratch/cv-course2025/lschind5/kits19/data"  # Update this path
    BATCH_SIZE = 1  
    NUM_EPOCHS = 200
    NUM_WORKERS = 4

    # load data
    all_cases = get_data_list(DATA_ROOT)
    print(f"   Found {len(all_cases)} cases", flush=True)

    # Split data
    train_data, val_data = train_test_split(
        all_cases, test_size=0.2, random_state=42
    )

    # Transforms

    train_transforms, val_transforms = get_3D_transforms()

    train_loader, val_loader = get_3D_dataset(train_data, val_data, train_transforms, val_transforms)

    # Model

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,  # Background, kidney, tumor
        channels=(24, 48, 96, 192, 320),  # Feature progression from paper
        strides=(2, 2, 2, 2),  # Downsampling strides
        num_res_units=2,  # Residual blocks per level
        act="LEAKYRELU",  # LeakyReLU activation 
        norm=Norm.INSTANCE,  # Instance normalization
        dropout=0.0,
        kernel_size=3,
        up_kernel_size=3  # For transposed convolutions
    )

    loss_fn = DiceFocalLoss(
            to_onehot_y=True,  # convert target to one-hot format
            softmax=True,       # apply softmax to model outputs
            weight=[0.3, 1, 3]  # Adjust weights for background, kidney, tumor
        ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Train the model
    train_kits19_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_path="kits19-model-3d-unet.pth",
        type="3d-unet",
    )