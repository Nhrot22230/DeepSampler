import torch.nn as nn
from torch import Tensor


class TFCBlock(nn.Module):
    """Time-Frequency Convolutions Block"""

    def __init__(self, in_channels: int, out_channels: int, kF: int, kT: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kF, kT), padding=(kF // 2, kT // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (kF, kT), padding=(kF // 2, kT // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (kF, kT), padding=(kF // 2, kT // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.residual(x)


class TDFBlock(nn.Module):
    """Time-Distributed Fully-connected Block"""

    def __init__(
        self, channels: int, f_dim: int, bn_factor: int, min_bn_units: int = 16
    ) -> None:
        super().__init__()
        reduced_f = max(min_bn_units, f_dim // bn_factor)

        self.tdf = nn.Sequential(
            nn.Conv1d(f_dim, reduced_f, 1, groups=channels),
            nn.ReLU(),
            nn.Conv1d(reduced_f, f_dim, 1, groups=channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, F, T = x.shape
        x_tdf = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        x_tdf = x_tdf.reshape(-1, C, F)  # [B*T, C, F]
        x_tdf = self.tdf(x_tdf)  # Apply TDF
        x_tdf = x_tdf.reshape(B, T, C, F)
        return x_tdf.permute(0, 2, 3, 1) + x  # Restore shape and add residual


class TFC_TDFBlock(nn.Module):
    """Combined TFC-TDF Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kF: int,
        kT: int,
        f_dim: int,
        bn_factor: int,
    ) -> None:
        super().__init__()
        self.tfc = TFCBlock(in_channels, out_channels, kF, kT)
        self.tdf = TDFBlock(out_channels, f_dim, bn_factor)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tfc(x)
        return self.tdf(x)
