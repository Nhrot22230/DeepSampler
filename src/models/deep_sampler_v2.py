from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpecEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ):
        super(SpecEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class SpecDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        dropout: float = 0.4,
    ):
        super(SpecDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=2
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels * 2, out_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Pad x to match the shape of skip
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x, skip], dim=1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class DeepSamplerV2(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 4,
        base_ch: int = 48,
        depth: int = 5,
        t_heads: int = 8,
        t_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        channels = [in_ch] + [base_ch * 2**i for i in range(depth + 1)]
        for i in range(depth):
            self.encoder.append(
                SpecEncoder(channels[i], channels[i + 1], groups=channels[i])
            )
            self.decoder.append(
                SpecDecoder(
                    channels[-i - 1],
                    channels[-i - 2],
                    groups=channels[i],
                    dropout=dropout,
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channels[-1]),
            nn.GELU(),
        )
        # Transformer operating in the latent space.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channels[-1],
                nhead=t_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=t_layers,
        )
        self.final = nn.Sequential(
            nn.Conv2d(channels[1], out_ch, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for i in range(self.depth):
            x, skip = self.encoder[i](x)
            skips.append(skip)

        x = self.bottleneck(x)
        b, c, h_lat, w_lat = x.shape
        # Flatten spatial dimensions: [B, C, h_lat * w_lat] then permute to [B, seq_len, C]
        x_flat = x.view(b, c, h_lat * w_lat).permute(0, 2, 1)
        x_trans = self.transformer(x_flat)
        # Reshape back to [B, C, h_lat, w_lat]

        x = x_trans.permute(0, 2, 1).view(b, c, h_lat, w_lat)
        for i in range(self.depth):
            x = self.decoder[i](x, skips[-i - 1])
        return self.final(x)
