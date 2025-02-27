from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpecEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(SpecEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
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
    ):
        super(SpecDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=2
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels * 2, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
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
        return x


class SCUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super(SCUNet, self).__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channels: int = [in_channels] + [base_channels * 2**i for i in range(depth + 1)]
        for i in range(depth):
            self.encoder.append(SpecEncoder(channels[i], channels[i + 1]))
            self.decoder.append(SpecDecoder(channels[-i - 1], channels[-i - 2]))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channels[-1]),
            nn.ReLU(),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(32, channels[-1]),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(channels[1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for i in range(self.depth):
            x, skip = self.encoder[i](x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.decoder[i](x, skips[-i - 1])
        x = self.final(x)
        return x
