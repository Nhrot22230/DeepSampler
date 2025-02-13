import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Bloque encoder simplificado:
    - Una única convolución con BatchNorm y ReLU.
    - Se guarda el output antes del pooling para usarlo en el skip.
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return skip, x


class DecoderBlock(nn.Module):
    """
    Bloque decoder simplificado:
    - Se usa ConvTranspose2d para upsampling.
    - Se concatenan los skip connections y se aplica una convolución
      para fusionar las características.
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DeepSampler(nn.Module):
    """
    Red DeepSampler con transformador en el espacio latente.
    Arquitectura encoder-decoder (similar a U-Net) con:
      - 'depth' bloques en el encoder y decoder.
      - Un cuello de botella que duplica los canales.
      - Un transformador que procesa el espacio latente.
      - Capa final que mapea al número deseado de canales de salida.

    Se incluye padding dinámico para que las dimensiones sean divisibles por 2^depth.
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=4,
        base_channels=32,
        depth=4,
        dropout=0.2,
        transformer_heads=4,
        transformer_layers=2,
    ):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        encoder_channels = []

        in_ch = input_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(EncoderBlock(in_ch, out_ch, dropout=dropout))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
        )
        bottleneck_channels = in_ch * 2

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=bottleneck_channels,
                nhead=transformer_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )

        for i in reversed(range(depth)):
            out_ch = encoder_channels[i]
            self.decoders.append(
                DecoderBlock(bottleneck_channels, out_ch, dropout=dropout)
            )
            bottleneck_channels = out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pad_factor = 2**self.depth
        new_h = h if h % pad_factor == 0 else h + (pad_factor - h % pad_factor)
        new_w = w if w % pad_factor == 0 else w + (pad_factor - w % pad_factor)
        padding = (
            (new_w - w) // 2,
            (new_w - w) - (new_w - w) // 2,
            (new_h - h) // 2,
            (new_h - h) - (new_h - h) // 2,
        )
        x = F.pad(x, padding)

        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Cuello de botella
        x = self.bottleneck(x)

        # Transformador
        b, c, h_lat, w_lat = x.shape
        x_flat = x.view(b, c, h_lat * w_lat).permute(
            0, 2, 1
        )  # [batch, seq_len, feature]
        x_trans = self.transformer(x_flat)  # [batch, seq_len, feature]
        x = x_trans.permute(0, 2, 1).view(b, c, h_lat, w_lat)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        # Capa final
        x = self.final_conv(x)
        # Crop para volver a las dimensiones originales
        return x[:, :, :h, :w]
