"""
Inference script for Kaggle submission (Image Classification).

This script is executed on Kaggle's environment.
It loads trained models and generates predictions.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import config
from augmentation import get_tta_transforms, get_valid_transforms
from dataset import ImageDataset
from model import load_model
from torch.utils.data import DataLoader


def inference_single(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> np.ndarray:
    """Run inference on a single model.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to use
        use_amp: Whether to use automatic mixed precision

    Returns:
        Numpy array of predictions
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Inference"):
            images = images.to(device)

            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            # Apply softmax
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def inference_with_tta(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    image_dir: Path,
    transforms_list: list,
    batch_size: int,
    num_workers: int,
    device: str,
    use_amp: bool = True,
) -> np.ndarray:
    """Run inference with Test Time Augmentation.

    Args:
        model: Trained model
        test_df: Test dataframe
        image_dir: Directory containing images
        transforms_list: List of transforms for TTA
        batch_size: Batch size
        num_workers: Number of workers
        device: Device to use
        use_amp: Whether to use AMP

    Returns:
        Averaged predictions across TTA transforms
    """
    all_predictions = []

    for i, transforms in enumerate(transforms_list):
        print(f"TTA {i + 1}/{len(transforms_list)}")

        dataset = ImageDataset(
            df=test_df,
            image_dir=image_dir,
            transforms=transforms,
            is_train=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        preds = inference_single(model, dataloader, device, use_amp)
        all_predictions.append(preds)

    # Average predictions
    return np.mean(all_predictions, axis=0)


def main():
    """Main inference function."""
    CFG = config.CFG

    # Device
    device = CFG.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test data
    # TODO: Update with your test data loading logic
    test_df = pd.read_csv(CFG.DATA_PATH / "test.csv")
    print(f"Test samples: {len(test_df)}")

    # Image directory
    # TODO: Update with your image directory
    image_dir = CFG.DATA_PATH / "images" / "test"

    # Transforms
    if CFG.TTA:
        transforms_list = get_tta_transforms(CFG.IMG_SIZE)
    else:
        transforms_list = [get_valid_transforms(CFG.IMG_SIZE)]

    # Model directory (from artifact)
    model_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"

    # Inference for each fold
    all_predictions = []

    for fold in range(CFG.N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")

        model_path = model_dir / f"model_fold{fold + 1}.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        # Load model
        model = load_model(
            model_path=str(model_path),
            model_name=CFG.MODEL_NAME,
            num_classes=CFG.NUM_CLASSES,
            device=device,
        )

        # Inference
        if CFG.TTA:
            preds = inference_with_tta(
                model=model,
                test_df=test_df,
                image_dir=image_dir,
                transforms_list=transforms_list,
                batch_size=CFG.BATCH_SIZE,
                num_workers=CFG.NUM_WORKERS,
                device=device,
                use_amp=CFG.USE_AMP,
            )
        else:
            dataset = ImageDataset(
                df=test_df,
                image_dir=image_dir,
                transforms=transforms_list[0],
                is_train=False,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=CFG.BATCH_SIZE,
                shuffle=False,
                num_workers=CFG.NUM_WORKERS,
                pin_memory=True,
            )
            preds = inference_single(model, dataloader, device, CFG.USE_AMP)

        all_predictions.append(preds)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average predictions across folds
    final_predictions = np.mean(all_predictions, axis=0)

    # Create submission
    # TODO: Update with your submission format
    submission = pd.read_csv(CFG.DATA_PATH / "sample_submission.csv")

    # For classification: use argmax
    submission[CFG.TARGET_COL] = final_predictions.argmax(axis=1)

    # Or for probability submission:
    # submission[CFG.TARGET_COL] = final_predictions[:, 1]

    # Save
    submission.to_csv(config.OUTPUT_DIR / "submission.csv", index=False)
    print(f"\nSubmission saved to: {config.OUTPUT_DIR / 'submission.csv'}")
    print(submission.head())


if __name__ == "__main__":
    main()
