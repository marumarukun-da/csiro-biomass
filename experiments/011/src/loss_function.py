"""Loss function for biomass regression."""

import torch
import torch.nn.functional as F
from torch import nn


class WeightedSmoothL1Loss(nn.Module):
    """Weighted Smooth L1 Loss for multi-target regression.

    Calculates Smooth L1 loss for each target separately,
    then combines them with weighted sum.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        beta: float = 1.0,
    ):
        """Initialize WeightedSmoothL1Loss.

        Args:
            weights: Weights for each target [Dry_Dead_g, Dry_Green_g, Dry_Clover_g].
                     If None, uses equal weights [1.0, 1.0, 1.0].
            beta: Threshold for switching between L1 and L2 behavior.
        """
        super().__init__()
        self.beta = beta
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]

    def forward(self, preds, targets):
        """Calculate weighted smooth L1 loss.

        Args:
            preds: Predictions [B, 3]
            targets: Ground truth [B, 3]

        Returns:
            Weighted loss (scalar)
        """
        # Calculate mean loss for each target
        loss_0 = F.smooth_l1_loss(preds[:, 0], targets[:, 0], beta=self.beta)
        loss_1 = F.smooth_l1_loss(preds[:, 1], targets[:, 1], beta=self.beta)
        loss_2 = F.smooth_l1_loss(preds[:, 2], targets[:, 2], beta=self.beta)

        # Weighted sum
        loss = self.weights[0] * loss_0 + self.weights[1] * loss_1 + self.weights[2] * loss_2

        return loss


class TotalAwareLoss(nn.Module):
    """Loss function that explicitly optimizes for Total and GDM.

    This loss aligns with the competition metric where:
    - Dry_Total_g has weight 0.5
    - GDM_g has weight 0.2
    - Individual components (Dead, Green, Clover) have weight 0.1 each

    The loss computes:
    - Individual component losses (Dead, Green, Clover)
    - Total loss (Dead + Green + Clover)
    - GDM loss (Green + Clover)

    And combines them with configurable weights.
    """

    def __init__(
        self,
        component_weight: float = 0.3,
        total_weight: float = 0.5,
        gdm_weight: float = 0.2,
        beta: float = 1.0,
    ):
        """Initialize TotalAwareLoss.

        Args:
            component_weight: Total weight for 3 individual components (split equally).
                              Default 0.3 means each component gets ~0.1.
            total_weight: Weight for Total (Dead + Green + Clover) loss.
                          Default 0.5 (matching competition metric).
            gdm_weight: Weight for GDM (Green + Clover) loss.
                        Default 0.2 (matching competition metric).
            beta: Smooth L1 beta parameter.
        """
        super().__init__()
        self.component_weight = component_weight
        self.total_weight = total_weight
        self.gdm_weight = gdm_weight
        self.beta = beta

    def forward(self, preds, targets):
        """Calculate total-aware loss.

        Args:
            preds: Predictions [B, 3] with [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]
            targets: Ground truth [B, 3]

        Returns:
            Combined loss (scalar)
        """
        # Individual component losses
        loss_dead = F.smooth_l1_loss(preds[:, 0], targets[:, 0], beta=self.beta)
        loss_green = F.smooth_l1_loss(preds[:, 1], targets[:, 1], beta=self.beta)
        loss_clover = F.smooth_l1_loss(preds[:, 2], targets[:, 2], beta=self.beta)

        # Total = Dead + Green + Clover
        pred_total = preds[:, 0] + preds[:, 1] + preds[:, 2]
        true_total = targets[:, 0] + targets[:, 1] + targets[:, 2]
        loss_total = F.smooth_l1_loss(pred_total, true_total, beta=self.beta)

        # GDM = Green + Clover
        pred_gdm = preds[:, 1] + preds[:, 2]
        true_gdm = targets[:, 1] + targets[:, 2]
        loss_gdm = F.smooth_l1_loss(pred_gdm, true_gdm, beta=self.beta)

        # Component loss (average of 3 components)
        loss_components = (loss_dead + loss_green + loss_clover) / 3.0

        # Combined loss
        loss = self.component_weight * loss_components + self.total_weight * loss_total + self.gdm_weight * loss_gdm

        return loss


class BiomassLossWithAuxClassification(nn.Module):
    """Combined loss for biomass regression with auxiliary State classification.

    Loss composition:
    - Regression loss: Smooth L1 for each of 3 targets (Dead, Green, Clover) with equal weight 1.0
    - Classification loss: CrossEntropy for State prediction with weight 0.5
    - Total weight sum: 3.0 (regression) + 0.5 (classification) = 3.5

    Supports both hard labels (indices) and soft labels (probability distributions) for Mixup.
    """

    def __init__(
        self,
        beta: float = 1.0,
        classification_weight: float = 0.5,
    ):
        """Initialize BiomassLossWithAuxClassification.

        Args:
            beta: Smooth L1 beta parameter for regression loss.
            classification_weight: Weight for auxiliary classification loss (default 0.5).
        """
        super().__init__()
        self.beta = beta
        self.classification_weight = classification_weight

    def forward(
        self,
        regression_preds: torch.Tensor,
        classification_logits: torch.Tensor,
        regression_targets: torch.Tensor,
        classification_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined loss.

        Args:
            regression_preds: Regression predictions [B, 3] (Dry_Dead_g, Dry_Green_g, Dry_Clover_g)
            classification_logits: Classification logits [B, num_classes] (State prediction)
            regression_targets: Regression ground truth [B, 3]
            classification_targets: Classification ground truth, either:
                - [B] hard labels (State indices) for standard training
                - [B, num_classes] soft labels (probability distribution) for Mixup

        Returns:
            Combined loss (scalar)
        """
        # Regression loss: equal weight (1.0) for each of 3 targets
        loss_dead = F.smooth_l1_loss(regression_preds[:, 0], regression_targets[:, 0], beta=self.beta)
        loss_green = F.smooth_l1_loss(regression_preds[:, 1], regression_targets[:, 1], beta=self.beta)
        loss_clover = F.smooth_l1_loss(regression_preds[:, 2], regression_targets[:, 2], beta=self.beta)

        # Total regression loss (sum of 3 components, each with weight 1.0)
        regression_loss = loss_dead + loss_green + loss_clover

        # Classification loss: CrossEntropy for State prediction
        # Support both hard labels (1D) and soft labels (2D) for Mixup
        if classification_targets.dim() == 1:
            # Hard labels: use standard cross entropy
            classification_loss = F.cross_entropy(classification_logits, classification_targets)
        else:
            # Soft labels: use soft cross entropy (KL divergence style)
            log_probs = F.log_softmax(classification_logits, dim=1)
            classification_loss = -(classification_targets * log_probs).sum(dim=1).mean()

        # Combined loss: regression (3.0) + classification (0.5) = 3.5
        total_loss = regression_loss + self.classification_weight * classification_loss

        return total_loss


def build_loss_function(
    beta: float = 1.0,
    classification_weight: float = 0.5,
) -> nn.Module:
    """Build BiomassLossWithAuxClassification function.

    Args:
        beta: Smooth L1 beta parameter for regression loss.
        classification_weight: Weight for auxiliary classification loss (default 0.5).

    Returns:
        BiomassLossWithAuxClassification module
    """
    return BiomassLossWithAuxClassification(
        beta=beta,
        classification_weight=classification_weight,
    )
