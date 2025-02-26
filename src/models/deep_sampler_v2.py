import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple


class Encoder(nn.Module):
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


class Decoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.res_path = (
            nn.Conv2d(in_ch + out_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Input: [batch, in_ch, h, w]
        # Skip: [batch, out_ch, h*2, w*2] (from corresponding encoder level)

        # Upsample input: [batch, in_ch, h, w] -> [batch, out_ch, h*2, w*2]
        x = self.up(x)

        # Concatenate with skip connection along channel dimension
        # [batch, out_ch, h*2, w*2] + [batch, out_ch, h*2, w*2] -> [batch, out_ch*2, h*2, w*2]
        x = torch.cat([x, skip], dim=1)

        # Apply residual connection
        # [batch, out_ch*2, h*2, w*2] -> [batch, out_ch, h*2, w*2]
        residual = self.res_path(x)

        # Apply convolutional block
        # [batch, out_ch*2, h*2, w*2] -> [batch, out_ch, h*2, w*2]
        conv = self.conv(x)

        # Add residual connection
        # [batch, out_ch, h*2, w*2] + [batch, out_ch, h*2, w*2] -> [batch, out_ch, h*2, w*2]
        return conv + residual


class DeepSamplerV2(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 4,
        base_ch: int = 32,
        depth=4,
        dropout=0.2,
        t_heads=4,
        t_layers=2,
    ):
        super().__init__()
        self.depth = depth

        # Channel configuration: [32, 64, 128, 256, 512] for depth=4
        channels = [base_ch * (2**i) for i in range(depth + 1)]

        # --- Encoder Path ---
        # Initial projection (1 -> 32)
        self.initial_conv = nn.Conv2d(
            in_ch, base_ch, kernel_size=1, padding=0
        )  # [1->32]

        # Encoder blocks (32->64->128->256 for depth=4)
        self.encoders = nn.ModuleList()
        for i in range(depth):  # [0,1,2,3 for depth=4]
            in_c = channels[i]
            out_c = channels[i + 1]
            self.encoders.append(
                Encoder(
                    in_c, out_c, dropout=dropout
                ),  # [32->64], [64->128], [128->256], [256->512]
            )

        # --- Bottleneck ---
        # Final encoder to bottleneck (512->1024)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(8, channels[-1]),
            nn.GELU(),
        )

        # --- Transformer ---
        # Requires (batch, seq_len, features) - handled in forward
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=channels[-1],  # 1024
                nhead=t_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=t_layers,
        )

        # --- Decoder Path ---
        # Reverse channel list: [1024, 256, 128, 64]
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):  # [3,2,1,0 for depth=4]
            in_c = channels[i + 1]  # [512, 256, 128, 64]
            out_c = in_c // 2  # [256, 128, 64, 32]
            self.decoders.append(
                Decoder(in_c, out_c, dropout=dropout)  # [512->256], [256->128], etc.
            )

        # --- Final Output ---
        # Project back to output channels (32->4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1), nn.GELU()  # [32->4]
        )

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

        x = self.initial_conv(x)

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
        cont = 0
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
            cont += 1

        # Final output
        x = self.final_conv(x)
        return x[:, :, :orig_h, :orig_w]
