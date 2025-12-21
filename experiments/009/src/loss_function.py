"""Loss function for biomass regression using density map approach."""

import torch
import torch.nn.functional as F
from torch import nn


class SimpleSmoothL1Loss(nn.Module):
    """Simple Smooth L1 Loss for multi-target regression.

    Calculates mean Smooth L1 loss across all targets.
    Used with density map model where predictions are summed pixel values.
    """

    def __init__(self, beta: float = 1.0):
        """Initialize SimpleSmoothL1Loss.

        Args:
            beta: Threshold for switching between L1 and L2 behavior.
                  - When |error| < beta: uses L2-like behavior (smooth)
                  - When |error| >= beta: uses L1 behavior (robust to outliers)
        """
        super().__init__()
        self.beta = beta

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate smooth L1 loss.

        Args:
            preds: Predictions [B, 3] with [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]
            targets: Ground truth [B, 3]

        Returns:
            Mean loss across all targets (scalar)
        """
        return F.smooth_l1_loss(preds, targets, beta=self.beta)


def build_loss_function(beta: float = 1.0, **kwargs) -> nn.Module:
    """Build SimpleSmoothL1Loss function.

    Args:
        beta: Smooth L1 beta parameter
        **kwargs: Ignored legacy parameters (component_weight, total_weight, gdm_weight)

    Returns:
        SimpleSmoothL1Loss module
    """
    return SimpleSmoothL1Loss(beta=beta)
