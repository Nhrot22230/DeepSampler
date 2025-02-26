import torch
import torch.nn as nn

from src.utils.train.tensor_logger import TensorLogger


class FreqAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5

        # Proyecciones QKV
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, F, T = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)  # [B,C,F,T] cada uno

        # Reshape para multi-head
        q = q.view(B, self.heads, C // self.heads, F, T)
        k = k.view(B, self.heads, C // self.heads, F, T)
        v = v.view(B, self.heads, C // self.heads, F, T)

        # Atención espacio-frecuencia
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,F,T,T]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(B, C, F, T)  # Re-combinar cabezas
        return self.to_out(out)


class ChannelSELayer(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    Implements style-based channel recalibration from:
    - SENet (Hu et al., 2018) [9]
    - SRM attention mechanisms [3]

    Args:
        channels: Input feature channels
        reduction: Channel reduction ratio (default: 16)

    Input Shape:
        (B, C, H, W)

    Output Shape:
        (B, C, H, W) (same as input)

    Example:
        >>> se = ChannelSELayer(channels=64)
        >>> x = torch.randn(4, 64, 128, 128)
        >>> x = se(x)  # Recalibrates channel importance

    Reference:
        [3] "Style-based Recalibration Module", Paperspace 2023
        [9] Hu et al. "Squeeze-and-Excitation Networks", CVPR 2018
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # X: [B,C,H,W]
        # SE(X): [B,C,1,1]
        return x * self.se(x)  # [B,C,H,W] * [B,C,1,1] = [B,C,H,W]


class TransformerModule(nn.Module):
    """Vision Transformer module for spectrogram sequence processing.

    Implements 2D spectrogram adaptation of:
    - Original Transformer (Vaswani et al., 2017) [7]
    - Vision Transformer principles [4]

    Args:
        dim: Feature dimension
        heads: Number of attention heads (default: 4)
        depth: Number of transformer layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        debug: Enable tensor shape logging

    Input Shape:
        (B, C, H, W)

    Output Shape:
        (B, C, H, W) (same as input)

    Example:
        >>> transformer = TransformerModule(dim=256)
        >>> latent = torch.randn(4, 256, 16, 16)
        >>> transformed = transformer(latent)  # Processes as (256, 256) sequence

    Reference:
        [4] "Transformer Models For Sequential Data", Restackio 2025
        [7] Vaswani et al. "Attention Is All You Need", NeurIPS 2017
    """

    def __init__(
        self,
        dim: int,
        layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        debug: bool = False,
    ):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dropout=dropout, batch_first=True
            ),
            num_layers=layers,
        )
        self.logger = TensorLogger(debug=debug, prefix=f"Transformer(dim={dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.log("input", x)
        b, c, h, w = x.shape
        x_reshaped = x.view(b, c, h * w).permute(0, 2, 1)
        self.logger.log("reshaped", x_reshaped)
        x_transformed = self.transformer(x_reshaped)
        self.logger.log("transformed", x_transformed)
        x_restored = x_transformed.permute(0, 2, 1).view(b, c, h, w)
        self.logger.log("output", x_restored)
        return x_restored
