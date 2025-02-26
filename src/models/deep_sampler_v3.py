import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple


class Encoder1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        self.pool = nn.MaxPool1d(2)
        self.res_path = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.res_path(x)
        x = self.conv(x) + residual
        return x, self.pool(x)


class Decoder1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        self.res_path = (
            nn.Conv1d(in_ch + out_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        residual = self.res_path(x)
        return self.conv(x) + residual


class DeepSamplerV3(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 32,
        depth=4,
        dropout=0.2,
        t_heads=4,
        t_layers=2,
    ):
        super().__init__()
        self.depth = depth
        channels = [base_ch * (2**i) for i in range(depth + 1)]

        # Encoder
        self.initial_conv = nn.Conv1d(in_ch, base_ch, kernel_size=1, padding=0)
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(Encoder1D(channels[i], channels[i + 1], dropout))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 3, padding=1),
            nn.GroupNorm(8, channels[-1]),
            nn.GELU(),
        )

        # Transformer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=channels[-1],
                nhead=t_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=t_layers,
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoders.append(Decoder1D(channels[i + 1], channels[i] // 2, dropout))

        # Output
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_ch // 2, out_ch, 3, padding=1),
            nn.Tanh(),  # Para salida en [-1, 1] si se usa normalización
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, L]
        orig_len = x.shape[2]
        pad_factor = 2**self.depth
        new_len = orig_len + (pad_factor - orig_len % pad_factor) % pad_factor
        padding = (0, new_len - orig_len)
        x = F.pad(x, padding)

        x = self.initial_conv(x)
        skips: List[torch.Tensor] = []

        # Encoder
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Transformer (B, C, L) -> (B, L, C) -> transformer -> (B, C, L)
        b, c, length = x.shape
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).reshape(b, c, length)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output
        x = self.final_conv(x)
        return x[:, :, :orig_len]
