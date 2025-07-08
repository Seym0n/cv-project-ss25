from random import random
import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNETR

from sklearn.model_selection import train_test_split

from utils.data_load import get_data_list, get_case_dataset, get_test_case
from utils.inference import get_case_predictions, evaluate_predictions

from utils.viz import plot_predictions_3D, plot_test_slices
import numpy as np
import random


if __name__ == "__main__":
    print(f"🏆 Evaluate 2D UNETR with MONAI", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"MONAI: {monai.__version__}", flush=True)

    # Set reproducibility
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    DATA_ROOT = "/scratch/cv-course2025/lschind5/kits19/data"
    NUM_WORKERS = 4
    MODEL_PATH = "/scratch/cv-course2025/lschind5/"

    # load data
    all_cases = get_data_list(DATA_ROOT)
    print(f"   Found {len(all_cases)} cases", flush=True)

    # Split data
    train_data, val_data = train_test_split(
        all_cases, test_size=0.2, random_state=42
    )

    val_cases = [case["case_id"] for case in val_data]
    test_cases = [f"case_{i:05d}" for i in range(211, 301)]

    # Load the model
    model = UNETR()

    val_data = {}
    for case in val_cases:
        
        case_dataset, case_image, case_label = get_case_dataset(case, DATA_ROOT)

        case_predicitions = get_case_predictions(model, case_dataset, device)

        val_data[case] = {
            "case_id": case,
            "image": case_image,
            "ground_truth": case_label,
            "predictions": case_predicitions
        }    
    
    # Save predictions
    np.savez(f"{MODEL_PATH}/unetr_2d_val_predictions.npz", **val_data)

    # Get evaluation metrics
    metrics = evaluate_predictions(val_data)

    # plot val predictions
    random_val_cases = random.sample(val_cases, 2)
    random_val_data = {case: val_data[case] for case in random_val_cases}
    plot_predictions_3D(random_val_data, output_path=f"{MODEL_PATH}/unetr_2d_val_predictions.png")

    # plot random test predictions
    random_test_cases = random.sample(test_cases, 2)

    test_data = {}
    for case in random_test_cases:
        case_dataset, case_image = get_test_case(case, DATA_ROOT)
        case_predictions = get_case_predictions(model, case_dataset, device)
        test_data[case] = {
            "case_id": case,
            "image": case_image,
            "predictions": case_predictions
        }

    plot_test_slices(test_data, output_path=f"{MODEL_PATH}/unetr_2d_test_predictions.png")




    