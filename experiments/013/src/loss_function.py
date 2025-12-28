"""Loss function for biomass regression."""

import torch
import torch.nn.functional as F
from torch import nn


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
            main_loss += w * F.smooth_l1_loss(main_pred[:, i], main_target[:, i], beta=self.beta)

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
            main_loss + self.state_weight * state_loss + self.height_weight * height_loss + self.aux_weight * aux_loss
        )

        return total_loss
