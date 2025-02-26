import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.train.tensor_logger import TensorLogger


class DecoderBlock(nn.Module):
    """Decoder block for the U-Net architecture.

    Args:
        in_ch: Input channels
        skip_ch: Skip connection channels
        out_ch: Output channels
        dropout: Dropout probability
        debug: Whether to enable tensor size logging
    """

    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, dropout: float, debug: bool = False
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2
        )
        # Adjust conv input channels to account for concatenation with skip connection
        self.conv = nn.Conv2d(
            in_channels=in_ch + skip_ch, out_channels=out_ch, kernel_size=3, padding=1
        )
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.logger = TensorLogger(
            debug=debug, prefix=f"Decoder({in_ch}+{skip_ch}->{out_ch})"
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        self.logger.log("input", x)
        self.logger.log("skip", skip)
        x = self.logger.log("after_upconv", self.upconv(x))

        # Handle potential size mismatches between upsampled feature and skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
            self.logger.log("after_resize", x)

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        self.logger.log("after_concat", x)

        x = self.logger.log("after_conv", self.conv(x))
        x = self.logger.log("after_norm", self.norm(x))
        x = self.logger.log("after_act", self.act(x))
        x = self.logger.log("after_dropout", self.dropout(x))
        return x
