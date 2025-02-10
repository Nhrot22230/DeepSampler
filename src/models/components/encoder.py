from typing import Tuple
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Bloque codificador para procesamiento de espectrogramas.

        Args:
            in_channels: Número de canales de entrada (e.g., 1 para mono)
            out_channels: Número de canales de salida después de la convolución
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Proceso de forward pass para el bloque codificador.

        Args:
            x: Tensor de entrada con forma (batch_size, in_channels, height, width)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tupla con:
                - Features antes del pooling (para skip connections)
                - Features después del pooling (para siguiente bloque)
        """
        x_conv = self.conv(x)  # Aplicar convolución + BatchNorm + ReLU
        x_pool = self.pool(x_conv)  # Reducción espacial
        return x_conv, x_pool
