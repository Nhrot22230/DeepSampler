import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuración básica del #logger
logging.basicConfig(level=logging.DEBUG)
# logger = logging.get#logger(__name__)

# Parámetros de entrada y salida
in_channels = 1  # El espectrograma de la mezcla es de 1 canal (magnitud)
out_channels = 4  # Cada canal de salida corresponde a la magnitud de una fuente


class SCUNet(nn.Module):
    def __init__(self):
        super(SCUNet, self).__init__()

        # Bloque 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 4
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque inferior (bottom)
        self.bottom = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(
                256 + 256, 256, kernel_size=3, padding=1
            ),  # se concatena con e4 (256 canales)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder Bloque 3
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(
                128 + 128, 128, kernel_size=3, padding=1
            ),  # concatena con e3 (128 canales)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder Bloque 2
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(
                64 + 64, 64, kernel_size=3, padding=1
            ),  # concatena con e2 (64 canales)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder Bloque 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(
                32 + 32, 32, kernel_size=3, padding=1
            ),  # concatena con e1 (32 canales)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Capa final 1×1: cada canal de salida representa el espectrograma de una fuente
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # logger.debug(f"Input shape: {x.shape}")
        # -------------------------
        # Encoder
        # -------------------------
        e1 = self.enc1(x)  # e1: (B, 32, H, W)
        p1 = self.pool1(e1)  # p1: (B, 32, H/2, W/2)
        # logger.debug(f"e1: {e1.shape}, p1: {p1.shape}")

        e2 = self.enc2(p1)  # e2: (B, 64, H/2, W/2)
        p2 = self.pool2(e2)  # p2: (B, 64, H/4, W/4)
        # logger.debug(f"e2: {e2.shape}, p2: {p2.shape}")

        e3 = self.enc3(p2)  # e3: (B, 128, H/4, W/4)
        p3 = self.pool3(e3)  # p3: (B, 128, H/8, W/8)
        # logger.debug(f"e3: {e3.shape}, p3: {p3.shape}")

        e4 = self.enc4(p3)  # e4: (B, 256, H/8, W/8)
        p4 = self.pool4(e4)  # p4: (B, 256, H/16, W/16)
        # logger.debug(f"e4: {e4.shape}, p4: {p4.shape}")

        b = self.bottom(p4)  # b: (B, 512, H/16, W/16)
        # logger.debug(f"b: {b.shape}")

        # -------------------------
        # Decoder
        # -------------------------

        # logger.debug(f"Decoder input shape: {b.shape}")
        # Bloque 4 del decoder: upsample, concatena con e4 y convoluciona
        d4 = self.up4(b)  # d4: (B, 256, H/8, W/8)

        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="nearest")

        # logger.debug(f"d4: {d4.shape}")
        d4 = torch.cat([d4, e4], dim=1)  # Concatenación: (B, 256+256, H/8, W/8)
        d4 = self.dec4(d4)  # d4: (B, 256, H/8, W/8)

        # Bloque 3 del decoder: upsample, concatena con e3 y convoluciona
        d3 = self.up3(d4)  # d3: (B, 128, H/4, W/4)

        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="nearest")
        # logger.debug(f"d3: {d3.shape}")

        d3 = torch.cat([d3, e3], dim=1)  # (B, 128+128, H/4, W/4)
        d3 = self.dec3(d3)  # d3: (B, 128, H/4, W/4)

        # Bloque 2 del decoder: upsample, concatena con e2 y convoluciona
        d2 = self.up2(d3)  # d2: (B, 64, H/2, W/2)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="nearest")
        # logger.debug(f"d2: {d2.shape}")

        d2 = torch.cat([d2, e2], dim=1)  # (B, 64+64, H/2, W/2)
        d2 = self.dec2(d2)  # d2: (B, 64, H/2, W/2)

        # Bloque 1 del decoder: upsample, concatena con e1 y convoluciona
        d1 = self.up1(d2)  # d1: (B, 32, H, W)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="nearest")
        # logger.debug(f"d1: {d1.shape}")

        d1 = torch.cat([d1, e1], dim=1)  # (B, 32+32, H, W)
        d1 = self.dec1(d1)  # d1: (B, 32, H, W)

        # Capa final 1x1 para obtener las 4 salidas (una por fuente) y ReLU
        out = self.final_conv(d1)  # out: (B, 4, H, W)
        # logger.debug(f"Output shape: {out.shape}")
        return out
