from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourceLoss(nn.Module):
    """
    Weighted multi-source loss function for spectrogram reconstruction.

    Args:
        weights: Relative weights for each source channel (will be normalized)
        distance: Distance metric ('l1' or 'l2')
        reduction: Loss reduction method ('mean' or 'sum')

    Inputs:
        outputs: (batch_size, num_sources, freq_bins, time_steps)
        targets: (batch_size, num_sources, freq_bins, time_steps)

    Returns:
        Weighted combination of per-source losses
    """

    def __init__(self, weights: list, distance: str = "l1"):
        super().__init__()
        self.weights = torch.tensor(
            [w / sum(weights) for w in weights], dtype=torch.float32
        )
        self.loss = nn.L1Loss() if distance.lower() == "l1" else nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        total_loss = 0.0
        for i, weight in enumerate(self.weights):
            if outputs.dim() == 3:
                total_loss += weight * self.loss(outputs[i], targets[i])
            elif outputs.dim() == 4:
                total_loss += weight * self.loss(outputs[:, i], targets[:, i])
            else:
                raise ValueError("Unsupported tensor dimensions.")
        return total_loss


class ResidualBlock(nn.Module):
    """
    Depthwise separable residual block with adaptive scaling.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        stride: Spatial scaling factor (1 for same size, 2 for down/upsample)
        mode: Network phase ('encode' or 'decode')
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, mode: str = "encode"):
        super().__init__()

        # Depthwise separable convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
        )

        # Residual connection
        self.res = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

        # Spatial scaling
        if mode == "encode":
            self.scale = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        else:
            self.scale = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(self.conv(x) + self.res(x))


class SpectralAE(nn.Module):
    """
    Efficient spectrogram autoencoder with configurable compression.

    Args:
        in_channels: Input spectrogram channels (default: 1)
        base_channels: Base channel count (default: 32)
        latent_ratio: Latent space compression ratio (default: 8)
        depth: Number of encoding/decoding blocks (default: 3)
        strides: Stride factors for each block (default: [2,2,2])

    Input shape: (batch, channels, freq_bins, time_steps)
    Output shape: (batch, channels, freq_bins, time_steps)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_ratio: int = 8,
        depth: int = 3,
        strides: Optional[list] = None,
    ):
        super().__init__()

        # Configuration
        self.strides = strides or [2] * depth
        if len(self.strides) != depth:
            raise ValueError("Strides list length must match depth")

        # Build encoder
        encoder_layers = []
        current_ch = in_channels
        for stride in self.strides:
            encoder_layers.append(
                ResidualBlock(current_ch, base_channels, stride, "encode")
            )
            current_ch = base_channels
            base_channels *= 2  # Double channels each layer

        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.Conv2d(current_ch, current_ch // latent_ratio, 1),  # Latent bottleneck
        )

        # Build decoder
        decoder_layers = []
        current_ch = current_ch // latent_ratio
        for stride in reversed(self.strides):
            base_channels //= 2  # Halve channels each layer
            decoder_layers.append(
                ResidualBlock(current_ch, base_channels, stride, "decode")
            )
            current_ch = base_channels

        self.decoder = nn.Sequential(
            nn.Conv2d(current_ch, current_ch * latent_ratio, 1),  # Expand from latent
            *decoder_layers,
            nn.Conv2d(base_channels, in_channels, 1),  # Final output
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class LatentSpaceLoss(nn.Module):
    """
    Composite loss with latent space regularization.

    Args:
        rec_weight: Reconstruction loss weight (default: 0.9)
        latent_weight: Latent regularization weight (default: 0.1)
        reg_type: Regularization type ('l2', 'l1', or 'variance') (default: 'l2')
    """

    def __init__(
        self, rec_weight: float = 0.9, latent_weight: float = 0.1, reg_type: str = "l2"
    ):
        super().__init__()
        if rec_weight + latent_weight != 1.0:
            raise ValueError("Loss weights must sum to 1.0")

        self.rec_weight = rec_weight
        self.latent_weight = latent_weight
        self.reg_type = reg_type.lower()

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        rec_loss = F.l1_loss(output, target)

        if self.reg_type == "l2":
            latent_loss = torch.mean(torch.norm(latent, p=2, dim=(1, 2, 3)))
        elif self.reg_type == "l1":
            latent_loss = torch.mean(torch.norm(latent, p=1, dim=(1, 2, 3)))
        elif self.reg_type == "variance":
            latent_loss = -torch.var(latent, dim=(0, 2, 3)).mean()  # Maximize variance
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")

        return (self.rec_weight * rec_loss) + (self.latent_weight * latent_loss)
