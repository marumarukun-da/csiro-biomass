"""Mixup data augmentation for dual input model."""

import numpy as np
import torch
from torch import Tensor


def mixup_data(
    images_left: Tensor,
    images_right: Tensor,
    targets: Tensor,
    alpha: float = 0.4,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample-wise Mixup for Dual Input model.

    Applies the same lambda to both left and right images from the same sample
    to maintain physical consistency (both halves come from the same original image).

    Uses the same shuffle indices for left, right, and targets to ensure
    proper pairing (e.g., sample A's left+right mixed with sample B's left+right).

    Args:
        images_left: Left half images [B, C, H, W]
        images_right: Right half images [B, C, H, W]
        targets: Target values [B, num_targets]
        alpha: Beta distribution parameter (default: 0.4)

    Returns:
        tuple of (mixed_left, mixed_right, mixed_targets, lam)
        - mixed_left: Mixed left images [B, C, H, W]
        - mixed_right: Mixed right images [B, C, H, W]
        - mixed_targets: Mixed target values [B, num_targets]
        - lam: Lambda values used for mixing [B]
    """
    batch_size = images_left.size(0)
    device = images_left.device

    # Generate same shuffle indices for left, right, and targets
    # This ensures proper pairing: A_left+B_left, A_right+B_right (same B)
    indices = torch.randperm(batch_size, device=device)

    # Sample-wise lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = torch.tensor(lam, dtype=torch.float32, device=device)

    # Broadcast lambda for images: [B] -> [B, 1, 1, 1]
    lam_img = lam.view(-1, 1, 1, 1)
    mixed_left = lam_img * images_left + (1 - lam_img) * images_left[indices]
    mixed_right = lam_img * images_right + (1 - lam_img) * images_right[indices]

    # Broadcast lambda for targets: [B] -> [B, 1]
    lam_target = lam.view(-1, 1)
    mixed_targets = lam_target * targets + (1 - lam_target) * targets[indices]

    return mixed_left, mixed_right, mixed_targets, lam
