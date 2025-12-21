"""Model definitions for biomass regression using density map approach."""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F


class PreActivationResidualBlock(nn.Module):
    """Pre-activation Residual Block for bottleneck.

    Architecture: GN -> ReLU -> Conv -> GN -> ReLU -> Conv + skip
    Reference: He et al. "Identity Mappings in Deep Residual Networks"

    Uses GroupNorm instead of BatchNorm for stability with small batch sizes.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        """Initialize PreActivationResidualBlock.

        Args:
            channels: Number of input/output channels (same for residual)
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W] with residual added
        """
        residual = x
        out = F.relu(self.gn1(x))
        out = self.conv1(out)
        out = F.relu(self.gn2(out))
        out = self.conv2(out)
        return out + residual


class DecoderBlock(nn.Module):
    """Unet decoder block.

    Architecture:
    - Upsample 2x (bilinear)
    - Concatenate with skip connection
    - Conv 3x3 -> GN -> ReLU -> Conv 3x3 -> GN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_groups: int = 32,
    ):
        """Initialize DecoderBlock.

        Args:
            in_channels: Number of input channels from previous decoder stage
            skip_channels: Number of channels from encoder skip connection
            out_channels: Number of output channels
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # After concat: in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input from previous decoder stage [B, C_in, H, W]
            skip: Skip connection from encoder [B, C_skip, 2H, 2W]

        Returns:
            Decoded features [B, C_out, 2H, 2W]
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UnetDecoder(nn.Module):
    """Unet decoder for generating density maps from multi-scale features.

    Dynamically supports 4 or 5 level encoder features.
    - 5-level (e.g., MaxViT): produces 256x256 density map
    - 4-level (e.g., ConvNeXtV2): produces 128x128 density map
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        num_outputs: int = 3,
        num_groups: int = 32,
    ):
        """Initialize UnetDecoder.

        Args:
            encoder_channels: Channel counts from encoder
                             - 5-level: [64, 96, 192, 384, 768] for MaxViT
                             - 4-level: [128, 256, 512, 1024] for ConvNeXtV2
            decoder_channels: Channel counts for decoder stages
                             - 5-level: [512, 256, 128, 64] (4 stages)
                             - 4-level: [512, 256, 128] (3 stages)
            num_outputs: Number of output channels (target count)
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.num_encoder_stages = len(encoder_channels)

        # Bottleneck: Pre-activation residual block (at lowest resolution)
        self.bottleneck = PreActivationResidualBlock(encoder_channels[-1], num_groups)

        # Build decoder blocks dynamically
        self.decoder_blocks = nn.ModuleList()
        num_decoder_stages = len(decoder_channels)

        for i in range(num_decoder_stages):
            if i == 0:
                # First decoder block: from bottleneck
                in_ch = encoder_channels[-1]
            else:
                in_ch = decoder_channels[i - 1]

            # Skip connection from encoder (reversed order)
            skip_idx = -(i + 2)  # -2, -3, -4, ...
            skip_ch = encoder_channels[skip_idx]
            out_ch = decoder_channels[i]

            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, num_groups)
            )

        # Final 1x1 conv to get density map
        self.final = nn.Conv2d(decoder_channels[-1], num_outputs, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Generate density map from encoder features.

        Args:
            features: List of feature maps from encoder
                     - 5-level: [f1, f2, f3, f4, f5] at resolutions [256, 128, 64, 32, 16]
                     - 4-level: [f1, f2, f3, f4] at resolutions [128, 64, 32, 16]

        Returns:
            Density map [B, num_outputs, H, W]
            - 5-level: [B, num_outputs, 256, 256]
            - 4-level: [B, num_outputs, 128, 128]
        """
        # Bottleneck (at lowest resolution feature)
        x = self.bottleneck(features[-1])

        # Decode with skip connections (from low to high resolution)
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_idx = -(i + 2)  # -2, -3, -4, ...
            x = dec_block(x, features[skip_idx])

        # Final conv
        x = self.final(x)

        return x


class DensityMapModel(nn.Module):
    """Density map based regression model for biomass prediction.

    Architecture:
    - Shared encoder (timm backbone with features_only=True)
    - Shared decoder (UnetDecoder)
    - Process left and right images with weight sharing (Siamese)
    - Generate density maps and compute mean for final prediction

    Supported backbones:
    - MaxViT (5-level): produces 256x256 density map
    - ConvNeXtV2 (4-level): produces 128x128 density map

    Input: Two images (left half, right half of original 2000x1000 image)
    Output: Regression predictions (mean of density maps)
    """

    def __init__(
        self,
        backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        decoder_channels: list[int] | None = None,
        num_outputs: int = 3,
        pretrained: bool = True,
        num_groups: int = 32,
    ):
        """Initialize DensityMapModel.

        Args:
            backbone: timm model name for encoder
            decoder_channels: Channel counts for decoder stages
                             - For 5-level (MaxViT): [512, 256, 128, 64]
                             - For 4-level (ConvNeXtV2): [512, 256, 128]
                             If None, auto-detected based on backbone
            num_outputs: Number of regression outputs
            pretrained: Use pretrained weights for backbone
            num_groups: Number of groups for GroupNorm in decoder
        """
        super().__init__()

        self.num_outputs = num_outputs

        # Encoder (shared backbone)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )

        # Get encoder channel info
        encoder_channels = self.encoder.feature_info.channels()
        # e.g., [64, 96, 192, 384, 768] for maxvit_small_tf_512
        # e.g., [128, 256, 512, 1024] for convnextv2_base

        # Auto-detect decoder channels if not provided
        if decoder_channels is None:
            if len(encoder_channels) == 5:
                # 5-level encoder (e.g., MaxViT)
                decoder_channels = [512, 256, 128, 64]
            else:
                # 4-level encoder (e.g., ConvNeXtV2)
                decoder_channels = [512, 256, 128]

        # Decoder (shared)
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_outputs=num_outputs,
            num_groups=num_groups,
        )

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        return_density_maps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with dual inputs.

        Args:
            x_left: Left half images [B, C, H, W]
            x_right: Right half images [B, C, H, W]
            return_density_maps: If True, also return density maps for visualization

        Returns:
            If return_density_maps is False:
                Predictions [B, num_outputs] (non-negative via Softplus, summed)
            If return_density_maps is True:
                Tuple of (predictions, left_density, right_density)
        """
        # Encode both sides (shared weights)
        left_features = self.encoder(x_left)
        right_features = self.encoder(x_right)

        # Decode both sides (shared weights)
        left_density = self.decoder(left_features)  # [B, num_outputs, 256, 256]
        right_density = self.decoder(right_features)  # [B, num_outputs, 256, 256]

        # Apply Softplus for non-negative constraint
        left_density = F.softplus(left_density)
        right_density = F.softplus(right_density)

        # # Sum over spatial dimensions
        # left_sum = left_density.sum(dim=(2, 3))  # [B, num_outputs]
        # right_sum = right_density.sum(dim=(2, 3))  # [B, num_outputs]

        # Mean over spatial dimensions
        left_mean = left_density.mean(dim=(2, 3))  # [B, num_outputs]
        right_mean = right_density.mean(dim=(2, 3))  # [B, num_outputs]

        # Combine left and right
        # predictions = left_sum + right_sum  # [B, num_outputs]

        predictions = left_mean + right_mean  # [B, num_outputs]

        if return_density_maps:
            return predictions, left_density, right_density

        return predictions


def build_model(
    backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k",
    decoder_channels: list[int] | None = None,
    num_outputs: int = 3,
    pretrained: bool = True,
    num_groups: int = 32,
    device: torch.device | str = "cuda",
    **kwargs,  # Ignore legacy parameters (dropout, hidden_size, etc.)
) -> nn.Module:
    """Build DensityMapModel.

    Args:
        backbone: timm model name for encoder
        decoder_channels: Channel counts for decoder stages (auto-detected if None)
        num_outputs: Number of regression outputs
        pretrained: Use pretrained weights for backbone
        num_groups: Number of groups for GroupNorm
        device: Device to place model on
        **kwargs: Ignored legacy parameters

    Returns:
        Model instance on specified device
    """
    model = DensityMapModel(
        backbone=backbone,
        decoder_channels=decoder_channels,
        num_outputs=num_outputs,
        pretrained=pretrained,
        num_groups=num_groups,
    )

    return model.to(device)


def load_model(
    model_path: str,
    backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k",
    decoder_channels: list[int] | None = None,
    num_outputs: int = 3,
    num_groups: int = 32,
    device: torch.device | str = "cuda",
    **kwargs,  # Ignore legacy parameters
) -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model weights
        backbone: timm model name for encoder
        decoder_channels: Channel counts for decoder stages (auto-detected if None)
        num_outputs: Number of regression outputs
        num_groups: Number of groups for GroupNorm
        device: Device to place model on
        **kwargs: Ignored legacy parameters

    Returns:
        Loaded model in eval mode
    """
    model = build_model(
        backbone=backbone,
        decoder_channels=decoder_channels,
        num_outputs=num_outputs,
        pretrained=False,
        num_groups=num_groups,
        device=device,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model
