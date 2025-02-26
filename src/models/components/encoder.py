from typing import Tuple

import torch
import torch.nn as nn

from src.utils.train.tensor_logger import TensorLogger


class EncoderBlock(nn.Module):
    """Encoder block for the U-Net architecture.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        dropout: Dropout probability
        debug: Whether to enable tensor size logging
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float, debug: bool = False):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)
        self.logger = TensorLogger(debug=debug, prefix=f"Encoder({in_ch}->{out_ch})")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.logger.log("input", x)
        x = self.logger.log("after_conv", self.conv(x))
        x = self.logger.log("after_norm", self.norm(x))
        x = self.logger.log("after_act", self.act(x))
        skip = x  # Save the output before pooling as skip connection
        x = self.logger.log("after_pool", self.pool(x))
        x = self.logger.log("after_dropout", self.dropout(x))
        return skip, x


class DepthwiseFreqConv(nn.Module):
    """Memory-efficient frequency processing"""

    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), groups=channels, padding=(1, 0)
        )

    def forward(self, x):
        return self.dw_conv(x)
