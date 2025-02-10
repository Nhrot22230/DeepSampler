import torch
import torch.nn as nn


class FreqConvBlock(nn.Module):
    """Capa que aplica kernels diferentes a cada banda."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.low_conv = nn.Conv2d(
            in_channels, out_channels // 3, kernel_size=(5, 1), padding=(2, 0)
        )
        self.mid_conv = nn.Conv2d(
            in_channels, out_channels // 3, kernel_size=(3, 1), padding=(1, 0)
        )
        self.high_conv = nn.Conv2d(in_channels, out_channels // 3, kernel_size=(1, 1))

    def forward(self, x):
        low = self.low_conv(x[:, :, : self.low_bin, :])
        mid = self.mid_conv(x[:, :, self.low_bin : self.mid_bin, :])
        high = self.high_conv(x[:, :, self.mid_bin :, :])
        return torch.cat([low, mid, high], dim=1)
