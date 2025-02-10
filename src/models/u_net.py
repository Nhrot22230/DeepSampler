from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Encoder block: one convolution followed by BatchNorm, ReLU, and 2x2 max pooling.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_prob: float = 0.4,
    ):
        """
        Decoder block: Upsampling via a transposed convolution, followed by concatenation
        with skip features and then a 3x3 convolution.
        """
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )
        # After concatenation, the number of channels becomes
        # (out_channels + skip_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.upconv(x)
        # Ensure that skip and x_up have the same spatial size
        if x_up.size()[2:] != skip.size()[2:]:
            x_up = F.interpolate(
                x_up, size=skip.size()[2:], mode="bilinear", align_corners=True
            )
        x_cat = torch.cat([x_up, skip], dim=1)
        x_out = self.conv(x_cat)
        return x_out


class SpectrogramChannelsUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,  # Change to 4 for multi-instrument separation.
        base_channels: int = 16,
    ):
        """
        Spectrogram‑Channels U‑Net model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (one per separated source).
            base_channels (int): Number of channels in the first encoder block.
        """
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(
            in_channels, base_channels
        )  # Output: base_channels, H/2, W/2
        self.enc2 = EncoderBlock(
            base_channels, base_channels * 2
        )  # Output: base_channels*2, H/4, W/4
        self.enc3 = EncoderBlock(
            base_channels * 2, base_channels * 4
        )  # Output: base_channels*4, H/8, W/8
        self.enc4 = EncoderBlock(
            base_channels * 4, base_channels * 8
        )  # Output: base_channels*8, H/16, W/16

        # Bottleneck (no pooling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec4 = DecoderBlock(
            base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
        )
        self.dec3 = DecoderBlock(
            base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )
        self.dec2 = DecoderBlock(
            base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )
        self.dec1 = DecoderBlock(
            base_channels * 2, skip_channels=base_channels, out_channels=base_channels
        )

        # Final layer: 1x1 convolution to map to out_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),  # Ensures non-negative spectrogram output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, in_channels, H, W]
        skip1, x1 = self.enc1(x)  # x1: [batch, base_channels, H/2, W/2]
        skip2, x2 = self.enc2(x1)  # x2: [batch, base_channels*2, H/4, W/4]
        skip3, x3 = self.enc3(x2)  # x3: [batch, base_channels*4, H/8, W/8]
        skip4, x4 = self.enc4(x3)  # x4: [batch, base_channels*8, H/16, W/16]

        x_bottleneck = self.bottleneck(x4)  # [batch, base_channels*16, H/16, W/16]

        x = self.dec4(x_bottleneck, skip4)  # [batch, base_channels*8, H/8, W/8]
        x = self.dec3(x, skip3)  # [batch, base_channels*4, H/4, W/4]
        x = self.dec2(x, skip2)  # [batch, base_channels*2, H/2, W/2]
        x = self.dec1(x, skip1)  # [batch, base_channels, H, W]
        out = self.final_conv(x)  # [batch, out_channels, H, W]
        return out
