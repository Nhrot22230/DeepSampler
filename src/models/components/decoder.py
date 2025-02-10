import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_prob: float = 0.3,
    ):
        """
        Bloque decodificador para arquitecturas encoder-decoder.

        Args:
            in_channels: Canales de entrada desde capa anterior
            skip_channels: Canales de la conexión residual (skip connection)
            out_channels: Canales de salida después de la convolución
            dropout_prob: Probabilidad de dropout (0.0-1.0)
        """
        super().__init__()

        # Capa de upsampling con cálculo automático de padding
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,  # Kernel más pequeño para mejor eficiencia
            stride=2,
            padding=1,
            output_padding=1,
        )

        # Capas de normalización y activación
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        # Mecanismo de atención para skip connections
        self.attention = nn.Sequential(
            nn.Conv2d(skip_channels, 1, kernel_size=1), nn.Sigmoid()
        )

        # Bloque convolucional post-concatenación
        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Paso 1: Upsample con transposed convolution
        x_up = self.upconv(x)
        x_up = self.norm(x_up)
        x_up = self.activation(x_up)

        # Paso 2: Mecanismo de atención espacial para skip connection
        attention_mask = self.attention(skip)
        skip_weighted = skip * attention_mask

        # Paso 3: Ajuste dimensional automático y concatenación
        if x_up.shape[-2:] != skip_weighted.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip_weighted.shape[-2:], mode="nearest")

        x_cat = torch.cat([x_up, skip_weighted], dim=1)

        # Paso 4: Convolución final
        return self.conv(x_cat)
