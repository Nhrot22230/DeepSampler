import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple


class EnhancedEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),  # Replaced BatchNorm with GroupNorm
            nn.GELU(),  # Changed from ReLU
            nn.Dropout2d(dropout),
        )
        self.pool = nn.MaxPool2d(2)
        self.res_path = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.res_path(x)
        x = self.conv(x) + residual  # Added residual connection
        return x, self.pool(x)


class EnhancedDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.res_path = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Dynamic padding handling
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.res_path(
            x[:, : x.size(1) // 2, :, :]
        )  # Residual fusion


class PreLNTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None
    ) -> torch.Tensor:
        # Pre-LayerNorm implementation
        src2 = self.norm1(src)
        src = src + self._sa_block(src2, src_mask, src_key_padding_mask)
        src2 = self.norm2(src)
        src = src + self._ff_block(src2)
        return src


class DeepSamplerV2(nn.Module):
    def __init__(
        self, in_ch=1, out_ch=4, base_ch=32, depth=4, dropout=0.2, t_heads=4, t_layers=2
    ):
        super().__init__()
        self.depth = depth

        # Encoder with enhanced blocks
        self.encoders = nn.ModuleList()
        encoder_channels = []
        current_ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.encoders.append(EnhancedEncoderBlock(current_ch, out_ch, dropout))
            encoder_channels.append(out_ch)
            current_ch = out_ch

        # Bottleneck with expanded capacity
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_ch, current_ch * 2, 3, padding=1),
            nn.GroupNorm(8, current_ch * 2),
            nn.GELU(),
        )
        bottleneck_ch = current_ch * 2

        # Transformer with pre-LN
        self.transformer = TransformerEncoder(
            PreLNTransformerEncoderLayer(
                d_model=bottleneck_ch,
                nhead=t_heads,
                dim_feedforward=bottleneck_ch * 4,
                dropout=dropout,
            ),
            num_layers=t_layers,
        )

        # Decoder with enhanced blocks
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = encoder_channels[i]
            self.decoders.append(EnhancedDecoderBlock(bottleneck_ch, out_ch, dropout))
            bottleneck_ch = out_ch

        # Final output with multi-scale fusion
        self.final_conv = nn.Sequential(nn.Conv2d(base_ch, out_ch, 1), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic padding implementation
        orig_h, orig_w = x.shape[2:]
        pad_factor = 2**self.depth
        new_h = orig_h + (pad_factor - orig_h % pad_factor) % pad_factor
        new_w = orig_w + (pad_factor - orig_w % pad_factor) % pad_factor
        padding = (
            (new_w - orig_w) // 2,
            (new_w - orig_w) - (new_w - orig_w) // 2,
            (new_h - orig_h) // 2,
            (new_h - orig_h) - (new_h - orig_h) // 2,
        )
        x = F.pad(x, padding)

        # Encoder forward with skip connections
        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck processing
        x = self.bottleneck(x)

        # Transformer processing
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        # Decoder with enhanced skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Final output
        x = self.final_conv(x)
        return x[:, :, :orig_h, :orig_w]
