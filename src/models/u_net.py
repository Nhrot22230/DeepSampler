import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import DecoderBlock, EncoderBlock


class SimpleUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 4,
        base_channels: int = 64,
        depth: int = 4,
        dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.depth = depth

        # Configuración de capas
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = input_channels

        # Construcción encoder
        for i in range(depth):
            out_channels = base_channels * (2**i)
            self.encoders.append(EncoderBlock(current_channels, out_channels))
            current_channels = out_channels

        # Cuello de botella
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, current_channels * 2, 3, padding=1),
            nn.BatchNorm2d(current_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )
        bottleneck_channels = current_channels * 2

        # Construcción decoder
        for i in reversed(range(depth)):
            skip_channels = base_channels * (2**i)
            self.decoders.append(
                DecoderBlock(
                    in_channels=(
                        bottleneck_channels if i == depth - 1 else skip_channels * 2
                    ),
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    dropout_prob=dropout_prob,
                )
            )

        # Capa final
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding dinámico para dimensiones divisibles por 2^depth
        h, w = x.size(2), x.size(3)
        pad_factor = 2**self.depth
        pad_h = (pad_factor - (h % pad_factor)) % pad_factor
        pad_w = (pad_factor - (w % pad_factor)) % pad_factor
        x = F.pad(x, (0, pad_w, 0, pad_h))  # Padding: (left, right, top, bottom)

        # Encoding
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        # Ajuste final de dimensiones
        x = self.activation(self.final_conv(x))
        return x[:, :, :h, :w]  # Crop a dimensiones originales

    @property
    def device(self):
        return next(self.parameters()).device
