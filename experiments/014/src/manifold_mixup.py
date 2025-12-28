"""Feature-level Mixup (ManifoldMixup) implementation."""

import numpy as np
import torch
from torch import Tensor


def feature_mixup(
    cls_a: Tensor,
    patches_a: Tensor,
    targets_a: dict[str, Tensor],
    cls_b: Tensor,
    patches_b: Tensor,
    targets_b: dict[str, Tensor],
    alpha: float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Apply ManifoldMixup to precomputed features.

    Args:
        cls_a: CLS tokens for samples A [B, D]
        patches_a: Patch tokens for samples A [B, N, D]
        targets_a: Dict of target tensors for samples A
        cls_b: CLS tokens for samples B [B, D]
        patches_b: Patch tokens for samples B [B, N, D]
        targets_b: Dict of target tensors for samples B
        alpha: Beta distribution parameter

    Returns:
        Tuple of:
        - mixed_cls: Mixed CLS tokens [B, D]
        - mixed_patches: Mixed patch tokens [B, N, D]
        - mixed_targets: Dict of mixed target tensors
        - lam: Lambda values [B]
    """
    batch_size = cls_a.size(0)
    device = cls_a.device

    # Sample lambda from Beta distribution (per-sample)
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = torch.tensor(lam, dtype=torch.float32, device=device)

    # Mix features
    lam_feat = lam.view(-1, 1)  # [B, 1]
    mixed_cls = lam_feat * cls_a + (1 - lam_feat) * cls_b

    lam_patch = lam.view(-1, 1, 1)  # [B, 1, 1]
    mixed_patches = lam_patch * patches_a + (1 - lam_patch) * patches_b

    # Mix targets
    mixed_targets = {}
    for key in targets_a:
        if targets_a[key].dim() == 1:
            # Scalar targets (e.g., height)
            mixed_targets[key] = lam * targets_a[key] + (1 - lam) * targets_b[key]
        else:
            # Vector targets (e.g., main_targets, aux_targets)
            lam_target = lam.view(-1, 1)
            mixed_targets[key] = lam_target * targets_a[key] + (1 - lam_target) * targets_b[key]

    return mixed_cls, mixed_patches, mixed_targets, lam


def mixup_batch(
    batch: dict[str, Tensor],
    alpha: float = 2.5,
) -> dict[str, Tensor]:
    """Apply ManifoldMixup to a batch (shuffle within batch).

    Args:
        batch: Dict containing cls_token, patch_tokens, and targets
        alpha: Beta distribution parameter

    Returns:
        Mixed batch dict
    """
    batch_size = batch["cls_token"].size(0)
    device = batch["cls_token"].device

    # Shuffle indices
    indices = torch.randperm(batch_size, device=device)

    # Get shuffled samples
    cls_b = batch["cls_token"][indices]
    patches_b = batch["patch_tokens"][indices]

    targets_a = {
        "main_targets": batch["main_targets"],
        "aux_targets": batch["aux_targets"],
    }
    targets_b = {
        "main_targets": batch["main_targets"][indices],
        "aux_targets": batch["aux_targets"][indices],
    }

    # Add optional targets
    if "height_value" in batch:
        targets_a["height_value"] = batch["height_value"]
        targets_b["height_value"] = batch["height_value"][indices]

    # Apply mixup
    mixed_cls, mixed_patches, mixed_targets, lam = feature_mixup(
        batch["cls_token"],
        batch["patch_tokens"],
        targets_a,
        cls_b,
        patches_b,
        targets_b,
        alpha=alpha,
    )

    # Handle state labels (soft labels for mixup)
    if "state_label" in batch:
        num_states = 4
        state_labels = batch["state_label"]
        # Convert to tensor if not already
        if not isinstance(state_labels, torch.Tensor):
            state_labels = torch.tensor(state_labels, device=device)
        state_onehot = torch.zeros(batch_size, num_states, device=device)
        state_onehot.scatter_(1, state_labels.view(-1, 1), 1.0)
        state_onehot_b = state_onehot[indices]
        lam_cls = lam.view(-1, 1)
        mixed_targets["state_label"] = lam_cls * state_onehot + (1 - lam_cls) * state_onehot_b

    result = {
        "cls_token": mixed_cls,
        "patch_tokens": mixed_patches,
        **mixed_targets,
        "lam": lam,
        "indices": indices,
    }

    return result
