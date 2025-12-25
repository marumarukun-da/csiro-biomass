"""Model definitions for biomass regression."""

import timm
import torch
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    """Generalized Mean Pooling.

    More flexible than average pooling, learnable parameter p controls pooling behavior:
    - p=1: average pooling
    - p=inf: max pooling
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, p_trainable: bool = True):
        """Initialize GeM pooling.

        Args:
            p: Initial power parameter
            eps: Small value for numerical stability
            p_trainable: Whether p is learnable
        """
        super().__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Pooled tensor [B, C, 1, 1]
        """
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x


def _ensure_bchw(x: torch.Tensor, num_features: int | None) -> torch.Tensor:
    """Ensure tensor is in BCHW format.

    Some backbones (e.g., Swin) return BHWC format.
    """
    if x.dim() == 4 and (num_features is not None) and x.shape[-1] == num_features:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


class DualInputRegressionNet(nn.Module):
    """Dual input regression model for left-right split images (Siamese-like).

    Architecture:
    - Shared timm backbone for both left and right images
    - GeM pooling for each
    - Concatenate features from both sides
    - MLP head for regression

    Input: Two images (left half, right half of original image)
    Output: Regression predictions
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_b0.in1k",
        num_outputs: int = 3,
        num_classes: int = 4,
        pretrained: bool = True,
        in_chans: int = 3,
        dropout: float = 0.1,
        hidden_size: int = 512,
    ):
        """Initialize DualInputRegressionNet.

        Args:
            model_name: timm model name for backbone
            num_outputs: Number of regression outputs (Total, GDM, Green)
            num_classes: Number of classes for auxiliary classification task (State)
            pretrained: Use pretrained weights
            in_chans: Number of input channels
            dropout: Dropout rate
            hidden_size: Hidden layer size in regression head
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.num_classes = num_classes

        # Shared backbone (weight sharing for left and right)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_chans, num_classes=0, drop_path_rate=0.2
        )
        self.num_features = self.backbone.num_features

        # Shared pooling
        self.global_pool = GeM(p_trainable=True)

        # Regression head (input: 2 * num_features from concatenated left+right)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs),
        )

        # Classification head for auxiliary State classification task
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # Height regression head for auxiliary Height_Avg_cm prediction
        self.height_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a single image.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            Pooled features [B, num_features]
        """
        batch_size = x.size(0)

        # Extract features
        features = self.backbone.forward_features(x)

        # Ensure BCHW format
        features = _ensure_bchw(features, self.num_features)

        # Global pooling
        pooled = self.global_pool(features).view(batch_size, -1)

        return pooled

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dual inputs.

        Args:
            x_left: Left half images [B, C, H, W]
            x_right: Right half images [B, C, H, W]

        Returns:
            Tuple of:
            - regression_output: [B, num_outputs] (non-negative via Softplus)
            - classification_logits: [B, num_classes] (raw logits for State classification)
            - height_pred: [B, 1] (non-negative via Softplus for Height_Avg_cm)
        """
        # Extract features from both sides (shared backbone)
        feat_left = self._extract_features(x_left)
        feat_right = self._extract_features(x_right)

        # Concatenate features
        feat_concat = torch.cat([feat_left, feat_right], dim=1)

        # Regression head
        regression_output = self.regression_head(feat_concat)
        # Apply Softplus for non-negative constraint (biomass values are always >= 0)
        regression_output = F.softplus(regression_output)

        # Classification head (raw logits, CrossEntropyLoss handles softmax)
        classification_logits = self.classification_head(feat_concat)

        # Height head (non-negative via Softplus)
        height_pred = self.height_head(feat_concat)
        height_pred = F.softplus(height_pred)

        return regression_output, classification_logits, height_pred


def build_model(
    model_name: str = "tf_efficientnetv2_b0.in1k",
    num_outputs: int = 3,
    num_classes: int = 4,
    pretrained: bool = True,
    in_chans: int = 3,
    dropout: float = 0.2,
    hidden_size: int = 512,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Build DualInputRegressionNet model.

    Args:
        model_name: timm model name
        num_outputs: Number of regression outputs
        num_classes: Number of classes for auxiliary classification task (State)
        pretrained: Use pretrained weights
        in_chans: Number of input channels
        dropout: Dropout rate
        hidden_size: Hidden layer size
        device: Device to place model on

    Returns:
        Model instance on specified device
    """
    model = DualInputRegressionNet(
        model_name=model_name,
        num_outputs=num_outputs,
        num_classes=num_classes,
        pretrained=pretrained,
        in_chans=in_chans,
        dropout=dropout,
        hidden_size=hidden_size,
    )

    return model.to(device)


def load_model(
    model_path: str,
    model_name: str = "tf_efficientnetv2_b0.in1k",
    num_outputs: int = 3,
    num_classes: int = 4,
    in_chans: int = 3,
    dropout: float = 0.2,
    hidden_size: int = 512,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model weights
        model_name: timm model name
        num_outputs: Number of regression outputs
        num_classes: Number of classes for auxiliary classification task (State)
        in_chans: Number of input channels
        dropout: Dropout rate
        hidden_size: Hidden layer size
        device: Device to place model on

    Returns:
        Loaded model in eval mode
    """
    model = build_model(
        model_name=model_name,
        num_outputs=num_outputs,
        num_classes=num_classes,
        pretrained=False,
        in_chans=in_chans,
        dropout=dropout,
        hidden_size=hidden_size,
        device=device,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model
