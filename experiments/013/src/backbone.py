"""DINOv3 Backbone for feature extraction."""

import timm
import torch
from torch import nn


class DINOv3Backbone(nn.Module):
    """DINOv3 Backbone for extracting CLS and patch features.

    Output structure for 960x960 input:
        - Total sequence: [B, 3605, 1280]
        - CLS token: output[:, 0, :]  -> [B, 1280]
        - Register tokens: output[:, 1:5, :] -> [B, 4, 1280] (ignored)
        - Patch tokens: output[:, 5:, :] -> [B, 3600, 1280]
    """

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
    PATCH_SIZE = 16
    HIDDEN_DIM = 1280
    NUM_REGISTERS = 4

    def __init__(self, pretrained: bool = True):
        """Initialize DINOv3Backbone.

        Args:
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @property
    def hidden_dim(self) -> int:
        """Return hidden dimension size."""
        return self.HIDDEN_DIM

    @property
    def num_registers(self) -> int:
        """Return number of register tokens."""
        return self.NUM_REGISTERS

    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size.

        Args:
            img_size: Image size (must be divisible by patch_size).

        Returns:
            Number of patches.
        """
        assert img_size % self.PATCH_SIZE == 0, f"Image size must be divisible by {self.PATCH_SIZE}"
        num_patches_per_side = img_size // self.PATCH_SIZE
        return num_patches_per_side * num_patches_per_side

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Tuple of:
            - cls_token: [B, 1280]
            - patch_tokens: [B, num_patches, 1280]
        """
        output = self.model(x)
        cls_token = output[:, 0, :]
        patch_tokens = output[:, 1 + self.NUM_REGISTERS :, :]
        return cls_token, patch_tokens

    def train(self, mode: bool = True):
        """Always keep in eval mode (frozen)."""
        return super().train(False)


def build_backbone(pretrained: bool = True, device: str = "cuda") -> DINOv3Backbone:
    """Build DINOv3 backbone.

    Args:
        pretrained: Whether to use pretrained weights.
        device: Device to place model on.

    Returns:
        DINOv3Backbone instance.
    """
    backbone = DINOv3Backbone(pretrained=pretrained)
    return backbone.to(device)
