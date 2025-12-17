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


def build_loss_function(
    beta: float = 1.0,
    component_weight: float = 0.3,
    total_weight: float = 0.5,
    gdm_weight: float = 0.2,
) -> nn.Module:
    """Build TotalAwareLoss function.

    Args:
        beta: Smooth L1 beta parameter
        component_weight: Weight for individual components (Dead, Green, Clover).
        total_weight: Weight for Total (Dead + Green + Clover) loss.
        gdm_weight: Weight for GDM (Green + Clover) loss.

    Returns:
        TotalAwareLoss module
    """
    return TotalAwareLoss(
        component_weight=component_weight,
        total_weight=total_weight,
        gdm_weight=gdm_weight,
        beta=beta,
    )


class TotalAwareGaussianNLLLoss(nn.Module):
    """Gaussian NLL Loss that explicitly optimizes for Total and GDM.

    This loss uses Gaussian Negative Log-Likelihood instead of Smooth L1.
    It allows the model to learn both predictions (mean) and their uncertainties (variance).

    The weights align with the competition metric where:
    - Dry_Total_g has weight 0.5
    - GDM_g has weight 0.2
    - Individual components (Dead, Green, Clover) have weight 0.1 each

    For combined targets (Total, GDM), variance is computed assuming independence:
    Var(A + B) = Var(A) + Var(B)

    Key benefit:
    - Samples that are hard to predict will have larger variance, reducing their loss contribution
    - Easy samples will have smaller variance, forcing the model to predict them accurately
    """

    def __init__(
        self,
        component_weight: float = 0.3,
        total_weight: float = 0.5,
        gdm_weight: float = 0.2,
        eps: float = 1e-6,
    ):
        """Initialize TotalAwareGaussianNLLLoss.

        Args:
            component_weight: Total weight for 3 individual components (split equally).
                              Default 0.3 means each component gets ~0.1.
            total_weight: Weight for Total (Dead + Green + Clover) loss.
                          Default 0.5 (matching competition metric).
            gdm_weight: Weight for GDM (Green + Clover) loss.
                        Default 0.2 (matching competition metric).
            eps: Small value for numerical stability in variance.
        """
        super().__init__()
        self.component_weight = component_weight
        self.total_weight = total_weight
        self.gdm_weight = gdm_weight
        self.eps = eps

        # PyTorch's GaussianNLLLoss
        self.gaussian_nll = nn.GaussianNLLLoss(eps=eps, reduction="mean")

    def forward(
        self,
        means: torch.Tensor,
        variances: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate total-aware Gaussian NLL loss.

        Args:
            means: Predicted means [B, 3] with [Dry_Dead_g, Dry_Green_g, Dry_Clover_g]
            variances: Predicted variances [B, 3]
            targets: Ground truth [B, 3]

        Returns:
            Combined loss (scalar)
        """
        # Individual component losses
        loss_dead = self.gaussian_nll(means[:, 0], targets[:, 0], variances[:, 0])
        loss_green = self.gaussian_nll(means[:, 1], targets[:, 1], variances[:, 1])
        loss_clover = self.gaussian_nll(means[:, 2], targets[:, 2], variances[:, 2])

        # Total = Dead + Green + Clover
        # Mean: sum of means
        # Variance: sum of variances (independent assumption)
        mean_total = means.sum(dim=1)
        var_total = variances.sum(dim=1)
        true_total = targets.sum(dim=1)
        loss_total = self.gaussian_nll(mean_total, true_total, var_total)

        # GDM = Green + Clover
        mean_gdm = means[:, 1] + means[:, 2]
        var_gdm = variances[:, 1] + variances[:, 2]
        true_gdm = targets[:, 1] + targets[:, 2]
        loss_gdm = self.gaussian_nll(mean_gdm, true_gdm, var_gdm)

        # Component loss (average of 3 components)
        loss_components = (loss_dead + loss_green + loss_clover) / 3.0

        # Combined loss
        loss = self.component_weight * loss_components + self.total_weight * loss_total + self.gdm_weight * loss_gdm

        return loss


def build_gaussian_nll_loss_function(
    component_weight: float = 0.3,
    total_weight: float = 0.5,
    gdm_weight: float = 0.2,
    eps: float = 1e-6,
) -> nn.Module:
    """Build TotalAwareGaussianNLLLoss function.

    Args:
        component_weight: Weight for individual components (Dead, Green, Clover).
        total_weight: Weight for Total (Dead + Green + Clover) loss.
        gdm_weight: Weight for GDM (Green + Clover) loss.
        eps: Small value for numerical stability in variance.

    Returns:
        TotalAwareGaussianNLLLoss module
    """
    return TotalAwareGaussianNLLLoss(
        component_weight=component_weight,
        total_weight=total_weight,
        gdm_weight=gdm_weight,
        eps=eps,
    )
