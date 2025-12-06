"""Image augmentation utilities using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Get training augmentations.

    Args:
        img_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05),
            A.GridDistortion(distort_limit=0.05),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
            ),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 8,
            max_width=img_size // 8,
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = 224) -> A.Compose:
    """Get validation/test augmentations.

    Args:
        img_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224) -> list[A.Compose]:
    """Get Test Time Augmentation transforms.

    Args:
        img_size: Target image size

    Returns:
        List of Albumentations Compose objects for TTA
    """
    base_transform = [
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    return [
        # Original
        A.Compose(base_transform),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transform),
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)] + base_transform),
        # Both flips
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base_transform),
    ]
