"""Model definitions using timm."""

import timm
import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    """Image classifier using timm pretrained models.

    Args:
        model_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_model(
    model_name: str = "tf_efficientnet_b0_ns",
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """Create and return a model.

    Args:
        model_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to put the model on

    Returns:
        Model instance
    """
    model = ImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return model.to(device)


def load_model(
    model_path: str,
    model_name: str = "tf_efficientnet_b0_ns",
    num_classes: int = 2,
    device: str = "cuda",
) -> nn.Module:
    """Load a trained model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        model_name: Name of the timm model
        num_classes: Number of output classes
        device: Device to put the model on

    Returns:
        Loaded model instance
    """
    model = ImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


# Available models (popular choices for Kaggle)
RECOMMENDED_MODELS = [
    # EfficientNet
    "tf_efficientnet_b0_ns",
    "tf_efficientnet_b1_ns",
    "tf_efficientnet_b2_ns",
    "tf_efficientnet_b3_ns",
    "tf_efficientnet_b4_ns",
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_m",
    # ConvNeXt
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    # Swin Transformer
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    # MaxViT
    "maxvit_tiny_tf_224",
    "maxvit_small_tf_224",
    # CoAtNet
    "coatnet_0_rw_224",
    "coatnet_1_rw_224",
]
