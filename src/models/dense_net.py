import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dense Layer: capa individual dentro de un bloque denso.
# Usa una convolución 1x1 (bottleneck) seguida de una convolución 3x3.
# ---------------------------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, dropout_rate=0.2):
        """
        Args:
            in_channels (int): Número de canales de entrada.
            growth_rate (int): Número de canales que añade esta capa.
            bn_size (int): Factor de ampliación para el bottleneck.
            dropout_rate (float): Tasa de dropout.
        """
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growth_rate, kernel_size=1, bias=False
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Dense Block: agrupa varias DenseLayer.
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(
        self, num_layers, in_channels, growth_rate, bn_size=4, dropout_rate=0.2
    ):
        """
        Args:
            num_layers (int): Número de DenseLayer en el bloque.
            in_channels (int): Número de canales de entrada al bloque.
            growth_rate (int): Número de canales que añade cada DenseLayer.
            bn_size (int): Factor de ampliación para el bottleneck.
            dropout_rate (float): Tasa de dropout.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


# ---------------------------------------------------------------------------
# Transition Down: reduce la dimensión espacial y comprime el número de canales.
# ---------------------------------------------------------------------------
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)  # Reduce la resolución a la mitad

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# Transition Up: upsampling mediante transposed convolution.
# ---------------------------------------------------------------------------
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
        """
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

    def forward(self, x):
        return self.transp_conv(x)


# ---------------------------------------------------------------------------
# DenseNet para separación de fuentes a partir de espectrogramas.
# Entrada: espectrograma de 1 canal.
# Salida: 4 espectrogramas (4 canales).
# ---------------------------------------------------------------------------
class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        num_init_features=32,
        growth_rate=16,
        block_config=(4, 4, 4),
        bn_size=4,
        dropout_rate=0.2,
    ):
        """
        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            num_init_features (int): Número de filtros en la convolución inicial.
            growth_rate (int): Número de canales que añade cada DenseLayer.
            block_config (tuple): Número de capas en cada DenseBlock del encoder.
            bn_size (int): Factor de ampliación para el bottleneck.
            dropout_rate (float): Tasa de dropout.
        """
        super().__init__()
        # Convolución inicial
        self.conv0 = nn.Conv2d(
            in_channels,
            num_init_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder: varios DenseBlocks y TransitionDown
        num_features = num_init_features
        self.encoder_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_features, growth_rate, bn_size, dropout_rate
            )
            self.encoder_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            # No se aplica TransitionDown en el último bloque del encoder
            if i != len(block_config) - 1:
                trans = TransitionDown(num_features, num_features // 2)
                self.trans_downs.append(trans)
                num_features = num_features // 2

        # Bottleneck: un bloque denso adicional
        self.bottleneck = DenseBlock(
            4, num_features, growth_rate, bn_size, dropout_rate
        )
        num_features = num_features + 4 * growth_rate

        # Decoder: camino ascendente (upsampling + bloque denso)
        self.trans_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        # Para este ejemplo usamos dos etapas de upsampling
        num_decoder_stages = 2
        for i in range(num_decoder_stages):
            up = TransitionUp(num_features, num_features // 2)
            self.trans_ups.append(up)
            num_features = num_features // 2
            block = DenseBlock(4, num_features, growth_rate, bn_size, dropout_rate)
            self.decoder_blocks.append(block)
            num_features = num_features + 4 * growth_rate

        # Capa final: mapea a 4 canales de salida
        self.final_conv = nn.Conv2d(
            num_features, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.pool0(out)

        # Encoder
        for i, block in enumerate(self.encoder_blocks):
            out = block(out)
            if i < len(self.trans_downs):
                out = self.trans_downs[i](out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder
        for up, block in zip(self.trans_ups, self.decoder_blocks):
            out = up(out)
            out = block(out)

        # Capa final
        out = self.final_conv(out)
        # Opcional: aplicar alguna activación (ej. ReLU o sigmoide) según la tarea
        return out
