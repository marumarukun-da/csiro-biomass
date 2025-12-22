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


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining main regression and auxiliary classification.

    Combines:
    - Main task: Biomass regression (TotalAwareLoss)
    - Auxiliary task: Species classification (CrossEntropyLoss)
    """

    def __init__(
        self,
        main_loss: nn.Module,
        lambda_species: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        """Initialize MultiTaskLoss.

        Args:
            main_loss: Loss function for main task (e.g., TotalAwareLoss)
            lambda_species: Weight for species classification loss
            class_weights: Optional class weights for species (for imbalanced data)
        """
        super().__init__()
        self.main_loss = main_loss
        self.lambda_species = lambda_species
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        species: torch.Tensor | None = None,
        species_b: torch.Tensor | None = None,
        lam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate multi-task loss.

        Args:
            outputs: Model outputs dict with "biomass" and optionally "species"
            targets: Ground truth for biomass [B, 3]
            species: Ground truth species labels [B] (optional, or species_a for Mixup)
            species_b: Shuffled species labels [B] for Mixup (optional)
            lam: Lambda values [B] for Mixup soft-label loss (optional)

        Returns:
            tuple of (total_loss, loss_dict)
            - total_loss: Combined loss for backpropagation
            - loss_dict: Individual loss values for logging
        """
        # Main task loss
        loss_main = self.main_loss(outputs["biomass"], targets)

        loss_dict = {"main": loss_main.item()}
        total_loss = loss_main

        # Auxiliary task loss (species classification)
        if species is not None and "species" in outputs:
            if species_b is not None and lam is not None:
                # Mixup: soft-label loss
                # loss = λ * CE(pred, species_a) + (1-λ) * CE(pred, species_b)
                ce_a = F.cross_entropy(
                    outputs["species"],
                    species,
                    weight=self.class_weights,
                    reduction="none",
                )
                ce_b = F.cross_entropy(
                    outputs["species"],
                    species_b,
                    weight=self.class_weights,
                    reduction="none",
                )
                loss_species = (lam * ce_a + (1 - lam) * ce_b).mean()
            else:
                # Normal: standard cross entropy
                loss_species = F.cross_entropy(
                    outputs["species"],
                    species,
                    weight=self.class_weights,
                )
            loss_dict["species"] = loss_species.item()
            total_loss = total_loss + self.lambda_species * loss_species

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


def compute_species_class_weights(
    species_counts: dict[str, int],
    species_list: list[str],
    power: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute class weights for species classification.

    Args:
        species_counts: Dict mapping species name to count
        species_list: Ordered list of species names
        power: Power for inverse frequency weighting (0.5 = sqrt, 1.0 = inverse)
        device: Device to place tensor on

    Returns:
        Class weights tensor [num_species]
    """
    weights = []
    for sp in species_list:
        count = species_counts.get(sp, 1)  # Default to 1 if not found
        weights.append(1.0 / (count**power))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Normalize so mean is 1
    weights = weights / weights.mean()

    return weights
