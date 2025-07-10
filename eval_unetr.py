from random import random
import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNETR

from tqdm import tqdm
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_load import get_data_list, get_case_dataset, get_test_case
from utils.transforms import get_2D_transforms, get_2D_test_transforms
from utils.inference import get_case_predictions, evaluate_predictions
from utils.viz import plot_predictions_3D, plot_test_slices

# from utils.viz import plot_predictions_3D, plot_test_slices
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
    MODEL_PATH = "/scratch/cv-course2025/lschind5/cv-project-ss25/"
    MODEL_NAME = "best_DiceCELoss_0.30-1.00-3.00_3e-05.pth"

    # load data
    all_cases = get_data_list(DATA_ROOT)
    print(f"   Found {len(all_cases)} cases", flush=True)

    # Split data
    train_data, val_data = train_test_split(
        all_cases, test_size=0.2, random_state=42
    )

    val_cases = [case["case_id"] for case in val_data]
    test_cases = [f"case_{i:05d}" for i in range(210, 299)]

    # Load the model
    model = UNETR(  # patch size fixed to 16x16 for 2D
    in_channels=1,
    out_channels=3, # background, kidney, tumor
    img_size=512,
    feature_size=16,
    norm_name='batch',
    spatial_dims=2).to(device)


    checkpoint = torch.load(os.path.join(MODEL_PATH, MODEL_NAME), map_location="cpu")
    model.load_state_dict(checkpoint)

    augment_transforms, no_aug_transforms= get_2D_transforms()

    val_data = {}
    for case in tqdm(val_cases):
            
            case_dataset, case_image, case_label = get_case_dataset(case, DATA_ROOT, no_aug_transforms, NUM_WORKERS)
            case_predicitions = get_case_predictions(model, case_dataset, device)

            val_data[case] = {
                "case_id": case,
                "image": case_image,
                "ground_truth": case_label,
                "predictions": case_predicitions
            }    
    

    # Get evaluation metrics
    metrics_standard_nobg, _ = evaluate_predictions(val_data, exclude_false_positives=True, slice_wise=False, exclude_background_slices=False)
    metrics_slices, _ = evaluate_predictions(val_data, exclude_false_positives=False, slice_wise=True, exclude_background_slices=False)
    metrics_slices_nofp, _ = evaluate_predictions(val_data, exclude_false_positives=True, slice_wise=True, exclude_background_slices=True)
    metrics_slices_nobg, _ = evaluate_predictions(val_data, exclude_false_positives=True, slice_wise=True, exclude_background_slices=True)
    metrics_standard, val_with_scores = evaluate_predictions(val_data, exclude_false_positives=False, slice_wise=False, exclude_background_slices=False)


    # save in .csv
    results = [metrics_standard, metrics_standard_nobg, metrics_slices, metrics_slices_nofp, metrics_slices_nobg]
    df = pd.DataFrame(results)
    os.makedirs(os.path.join(MODEL_PATH, "results"), exist_ok=True)
    df.to_csv(os.path.join(MODEL_PATH, "results", "eval_results.csv"), index=False)

    # select cases: good kidney dice, bad kidney dice, good tumor dice, bad tumor dice
    # sort cases by kidney dice
    sorted_cases_kidney = sorted(val_with_scores.items(), key=lambda x: x[1]["kidney_dice"], reverse=True)
    sorted_cases_tumor = sorted(val_with_scores.items(), key=lambda x: x[1]["tumor_dice"], reverse=True)

    # select best and worst cases
    best_kidney_cases = [case_id for case_id, data in sorted_cases_kidney[:5]]  # Top 5 kidney cases
    worst_kidney_cases = [case_id for case_id, data in sorted_cases_kidney[-5:]]  # Bottom 5 kidney cases
    best_tumor_cases = [case_id for case_id, data in sorted_cases_tumor[:5]]  # Top 5 tumor cases
    worst_tumor_cases = [case_id for case_id, data in sorted_cases_tumor[-5:]]  # Bottom 5 tumor cases

    # Combine all selected case IDs into a single set (removes duplicates)
    selected_case_ids = set(best_kidney_cases + worst_kidney_cases + best_tumor_cases + worst_tumor_cases)
    selected_val_data = {case_id: val_with_scores[case_id] for case_id in selected_case_ids}

    plot_predictions_3D(selected_val_data, output_path=os.path.join(MODEL_PATH, "results", "comparison"))

    # plot random test predictions
    random_test_cases = random.sample(test_cases, 25)
    test_transforms = get_2D_test_transforms()
    test_data = {}
    for case in random_test_cases:
        case_dataset, case_image = get_test_case(case, DATA_ROOT, test_transforms, NUM_WORKERS)
        case_predictions = get_case_predictions(model, case_dataset, device)
        test_data[case] = {
            "case_id": case,
            "image": case_image,
            "predictions": case_predictions
        }

    plot_test_slices(test_data, slice_selection='evenly_spaced', output_path=os.path.join(MODEL_PATH, "results", "slices"))




    