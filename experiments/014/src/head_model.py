"""CLS Gating Head Model for DINOv3 features."""

import torch
import torch.nn.functional as F
from torch import nn


class CLSGatingHead(nn.Module):
    """Head model with CLS-based gating and multi-task outputs.

    Uses CLS token to compute attention weights over patch tokens,
    then combines CLS and weighted patch features for prediction.

    Architecture:
        1. CLS Gating: cls_token -> MLP -> softmax -> attention weights
        2. Weighted Pooling: patch_tokens * attn_weights -> sum
        3. Shared Representation: concat(cls, weighted_pool) -> MLP
        4. Multi-Head Outputs: main, state, height, aux
    """

    def __init__(
        self,
        hidden_dim: int = 1280,
        num_patches: int = 3600,
        shared_hidden: int = 512,
        num_main_outputs: int = 3,  # Total, GDM, Green
        num_states: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize CLSGatingHead.

        Args:
            hidden_dim: DINOv3 hidden dimension (1280).
            num_patches: Number of patch tokens (3600 for 960x960 input).
            shared_hidden: Hidden dimension for shared representation.
            num_main_outputs: Number of main regression outputs (Total, GDM, Green).
            num_states: Number of state classes for classification.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # CLS Gating: cls -> attention weights over patches
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_patches),
        )

        # Shared representation: concat(cls, weighted_pool) -> shared features
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, shared_hidden),
            nn.LayerNorm(shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.main_head = nn.Linear(shared_hidden, num_main_outputs)  # Total, GDM, Green
        self.state_head = nn.Linear(shared_hidden, num_states)  # State classification
        self.height_head = nn.Linear(shared_hidden, 1)  # Height regression
        self.aux_head = nn.Linear(shared_hidden, 2)  # Dead, Clover

    def forward(
        self, cls_token: torch.Tensor, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            cls_token: [B, hidden_dim]
            patch_tokens: [B, num_patches, hidden_dim]

        Returns:
            Tuple of:
            - main_out: [B, 3] (Total, GDM, Green) - non-negative
            - state_out: [B, 4] logits
            - height_out: [B, 1] - non-negative
            - aux_out: [B, 2] (Dead, Clover) - non-negative
        """
        # CLS Gating: compute attention weights
        attn_weights = F.softmax(self.gate_mlp(cls_token), dim=-1)  # [B, num_patches]

        # Weighted pooling of patch tokens
        weighted_pool = (patch_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

        # Shared representation
        combined = torch.cat([cls_token, weighted_pool], dim=-1)  # [B, hidden_dim * 2]
        shared = self.shared_mlp(combined)  # [B, shared_hidden]

        # Output heads with non-negative constraints
        main_out = F.softplus(self.main_head(shared))
        state_out = self.state_head(shared)  # raw logits
        height_out = F.softplus(self.height_head(shared))
        aux_out = F.softplus(self.aux_head(shared))

        return main_out, state_out, height_out, aux_out


def build_head_model(
    hidden_dim: int = 1280,
    num_patches: int = 3600,
    shared_hidden: int = 512,
    num_main_outputs: int = 3,
    num_states: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> CLSGatingHead:
    """Build CLS Gating head model.

    Args:
        hidden_dim: DINOv3 hidden dimension.
        num_patches: Number of patch tokens.
        shared_hidden: Hidden dimension for shared representation.
        num_main_outputs: Number of main regression outputs.
        num_states: Number of state classes.
        dropout: Dropout rate.
        device: Device to place model on.

    Returns:
        CLSGatingHead instance.
    """
    model = CLSGatingHead(
        hidden_dim=hidden_dim,
        num_patches=num_patches,
        shared_hidden=shared_hidden,
        num_main_outputs=num_main_outputs,
        num_states=num_states,
        dropout=dropout,
    )
    return model.to(device)


def load_head_model(
    model_path: str,
    hidden_dim: int = 1280,
    num_patches: int = 3600,
    shared_hidden: int = 512,
    num_main_outputs: int = 3,
    num_states: int = 4,
    dropout: float = 0.1,
    device: str = "cuda",
) -> CLSGatingHead:
    """Load trained head model from checkpoint.

    Args:
        model_path: Path to model weights.
        hidden_dim: DINOv3 hidden dimension.
        num_patches: Number of patch tokens.
        shared_hidden: Hidden dimension for shared representation.
        num_main_outputs: Number of main regression outputs.
        num_states: Number of state classes.
        dropout: Dropout rate.
        device: Device to place model on.

    Returns:
        Loaded CLSGatingHead in eval mode.
    """
    model = build_head_model(
        hidden_dim=hidden_dim,
        num_patches=num_patches,
        shared_hidden=shared_hidden,
        num_main_outputs=num_main_outputs,
        num_states=num_states,
        dropout=dropout,
        device=device,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model
