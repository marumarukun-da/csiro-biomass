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


class MultiTaskBiomassLoss(nn.Module):
    """Multi-task loss for biomass regression with auxiliary tasks.

    Loss composition:
    - Regression loss: Smooth L1 for 3 targets (Total, GDM, Green) with configurable weights
    - State classification loss: CrossEntropy with configurable weight
    - Height regression loss: Smooth L1 with configurable weight

    Supports both hard labels (indices) and soft labels (probability distributions) for Mixup.
    """

    def __init__(
        self,
        beta: float = 1.0,
        regression_weights: list[float] | None = None,
        classification_weight: float = 0.1,
        height_weight: float = 0.1,
    ):
        """Initialize MultiTaskBiomassLoss.

        Args:
            beta: Smooth L1 beta parameter for regression loss.
            regression_weights: Weights for each regression target [Total, GDM, Green].
                               Default: [1.0, 0.6, 0.3]
            classification_weight: Weight for auxiliary State classification loss.
            height_weight: Weight for auxiliary Height regression loss.
        """
        super().__init__()
        self.beta = beta
        self.regression_weights = regression_weights if regression_weights is not None else [1.0, 0.6, 0.3]
        self.classification_weight = classification_weight
        self.height_weight = height_weight

    def forward(
        self,
        regression_preds: torch.Tensor,
        classification_logits: torch.Tensor,
        height_preds: torch.Tensor,
        regression_targets: torch.Tensor,
        classification_targets: torch.Tensor,
        height_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined multi-task loss.

        Args:
            regression_preds: Regression predictions [B, 3] (Dry_Total_g, GDM_g, Dry_Green_g)
            classification_logits: Classification logits [B, num_classes] (State prediction)
            height_preds: Height predictions [B, 1]
            regression_targets: Regression ground truth [B, 3]
            classification_targets: Classification ground truth, either:
                - [B] hard labels (State indices) for standard training
                - [B, num_classes] soft labels (probability distribution) for Mixup
            height_targets: Height ground truth [B] or [B, 1]

        Returns:
            Combined loss (scalar)
        """
        # Regression loss: weighted Smooth L1 for each target
        loss_total = F.smooth_l1_loss(regression_preds[:, 0], regression_targets[:, 0], beta=self.beta)
        loss_gdm = F.smooth_l1_loss(regression_preds[:, 1], regression_targets[:, 1], beta=self.beta)
        loss_green = F.smooth_l1_loss(regression_preds[:, 2], regression_targets[:, 2], beta=self.beta)

        regression_loss = (
            self.regression_weights[0] * loss_total
            + self.regression_weights[1] * loss_gdm
            + self.regression_weights[2] * loss_green
        )

        # Classification loss: CrossEntropy for State prediction
        # Support both hard labels (1D) and soft labels (2D) for Mixup
        if classification_targets.dim() == 1:
            classification_loss = F.cross_entropy(classification_logits, classification_targets)
        else:
            log_probs = F.log_softmax(classification_logits, dim=1)
            classification_loss = -(classification_targets * log_probs).sum(dim=1).mean()

        # Height regression loss: Smooth L1
        height_targets_flat = height_targets.view(-1)
        height_preds_flat = height_preds.view(-1)
        height_loss = F.smooth_l1_loss(height_preds_flat, height_targets_flat, beta=self.beta)

        # Combined loss
        total_loss = (
            regression_loss + self.classification_weight * classification_loss + self.height_weight * height_loss
        )

        return total_loss


def build_loss_function(
    beta: float = 1.0,
    regression_weights: list[float] | None = None,
    classification_weight: float = 0.2,
    height_weight: float = 0.2,
) -> nn.Module:
    """Build MultiTaskBiomassLoss function.

    Args:
        beta: Smooth L1 beta parameter for regression loss.
        regression_weights: Weights for each regression target [Total, GDM, Green].
                           Default: [1.5, 0.9, 0.3]
        classification_weight: Weight for auxiliary State classification loss.
        height_weight: Weight for auxiliary Height regression loss.

    Returns:
        MultiTaskBiomassLoss module
    """
    return MultiTaskBiomassLoss(
        beta=beta,
        regression_weights=regression_weights,
        classification_weight=classification_weight,
        height_weight=height_weight,
    )


class DINOv3MultiTaskLoss(nn.Module):
    """Multi-task loss for DINOv3 head with all auxiliary tasks.

    Tasks:
    - Main: Total, GDM, Green (Smooth L1, weighted)
    - Aux1: State classification (CrossEntropy)
    - Aux2: Height regression (Smooth L1)
    - Aux3: Dead, Clover regression (Smooth L1)
    """

    def __init__(
        self,
        beta: float = 1.0,
        main_weights: list[float] | None = None,
        state_weight: float = 0.1,
        height_weight: float = 0.1,
        aux_weight: float = 0.1,
    ):
        """Initialize loss function.

        Args:
            beta: Smooth L1 beta parameter.
            main_weights: Weights for [Total, GDM, Green]. Default: [1.0, 0.6, 0.3]
            state_weight: Weight for state classification loss.
            height_weight: Weight for height regression loss.
            aux_weight: Weight for Dead/Clover auxiliary loss.
        """
        super().__init__()
        self.beta = beta
        self.main_weights = main_weights or [1.0, 0.6, 0.3]
        self.state_weight = state_weight
        self.height_weight = height_weight
        self.aux_weight = aux_weight

    def forward(
        self,
        main_pred: torch.Tensor,
        state_pred: torch.Tensor,
        height_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        main_target: torch.Tensor,
        state_target: torch.Tensor,
        height_target: torch.Tensor,
        aux_target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined multi-task loss.

        Args:
            main_pred: [B, 3] predictions for Total, GDM, Green
            state_pred: [B, 4] logits for State classification
            height_pred: [B, 1] prediction for Height
            aux_pred: [B, 2] predictions for Dead, Clover
            main_target: [B, 3] targets for Total, GDM, Green
            state_target: [B] indices or [B, 4] soft labels for State
            height_target: [B] or [B, 1] target for Height
            aux_target: [B, 2] targets for Dead, Clover

        Returns:
            Combined loss scalar.
        """
        # Main loss: weighted Smooth L1
        main_loss = 0.0
        for i, w in enumerate(self.main_weights):
            main_loss += w * F.smooth_l1_loss(
                main_pred[:, i], main_target[:, i], beta=self.beta
            )

        # State classification loss
        if state_target.dim() == 1:
            state_loss = F.cross_entropy(state_pred, state_target)
        else:
            # Soft labels (for Mixup)
            log_probs = F.log_softmax(state_pred, dim=1)
            state_loss = -(state_target * log_probs).sum(dim=1).mean()

        # Height regression loss
        height_target_flat = height_target.view(-1)
        height_pred_flat = height_pred.view(-1)
        height_loss = F.smooth_l1_loss(height_pred_flat, height_target_flat, beta=self.beta)

        # Auxiliary loss: Dead, Clover
        aux_loss = F.smooth_l1_loss(aux_pred, aux_target, beta=self.beta)

        # Combined loss
        total_loss = (
            main_loss
            + self.state_weight * state_loss
            + self.height_weight * height_loss
            + self.aux_weight * aux_loss
        )

        return total_loss
