import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


# ---------------------------------------------------------------------------
# Weighted L1 Loss (MultiSourceL1Loss)
# ---------------------------------------------------------------------------
class MultiSourceLoss(nn.Module):
    def __init__(self, weights: List[float], distance: str = "l1"):
        """
        Args:
            weights (list of float): A list of weights for each channel.
        """
        super().__init__()
        self.weights = [w / sum(weights) for w in weights]
        self.l1_loss = nn.L1Loss(reduction="mean")

        if distance == "l1":
            self.loss = nn.L1Loss(reduction="mean")
        elif distance == "l2":
            self.loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Invalid distance: {distance}")

    def forward(self, outputs, targets):
        total_loss = 0.0
        for i, weight in enumerate(self.weights):
            total_loss += weight * self.loss(outputs[i], targets[i])
        return total_loss


# ---------------------------------------------------------------------------
# Weighted Multi-Scale Spectral Loss (MultiSourceMultiScaleSpectralLoss)
# ---------------------------------------------------------------------------
class MultiScaleLoss(nn.Module):
    def __init__(
        self, weights: List[float], scales: List[int] = [1, 2, 4], distance: str = "l1"
    ):
        """
        Args:
            channel_weights (list of float): Pesos para cada canal (deben estar normalizados).
            scales (list of int): Escalas (factores de downsampling) en las que se calculará.
            reduction (str): Método de reducción para la pérdida (por defecto "mean").
        """
        super().__init__()
        self.channel_weights = [w / sum(weights) for w in weights]
        self.scales = scales

        if distance == "l1":
            self.loss = nn.L1Loss(reduction="mean")
        elif distance == "l2":
            self.loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Invalid distance: {distance}")

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tensor de salida con forma [C, H, W].
            targets: Tensor objetivo con forma [C, H, W].
        Returns:
            Pérdida total combinada en múltiples escalas y canales.
        """
        total_loss = 0.0

        for i, weight in enumerate(self.channel_weights):
            channel_loss = 0.0

            for scale in self.scales:
                if scale == 1:
                    out_scaled = outputs[i]
                    tar_scaled = targets[i]
                else:
                    out_scaled = F.avg_pool2d(
                        outputs[i].unsqueeze(0), kernel_size=scale, stride=scale
                    ).squeeze(0)
                    tar_scaled = F.avg_pool2d(
                        targets[i].unsqueeze(0), kernel_size=scale, stride=scale
                    ).squeeze(0)

                channel_loss += self.loss(out_scaled, tar_scaled)

            channel_loss /= len(self.scales)
            total_loss += weight * channel_loss
        return total_loss


# ---------------------------------------------------------------------------
# VGG Feature Extractor
# ---------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, selected_layers: List[int] = [3, 8, 17, 26]):
        """
        Se extraen las salidas de las capas indicadas (por índice) del VGG19.
        Por ejemplo, en este caso se extraen:
            - Después de relu1_2 (índice 3)
            - Después de relu2_2 (índice 8)
            - Después de relu3_4 (índice 17)
            - Después de relu4_4 (índice 26)
        """
        super().__init__()
        vgg11 = models.vgg11(weights="DEFAULT").features
        self.selected_layers = selected_layers
        self.vgg_layers = vgg11
        # Los parámetros de VGG no se actualizarán
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Parámetros de normalización que VGG espera (imagen en [0,1])
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        """
        x: tensor de forma [N, 3, H, W]. Se normaliza antes de extraer características.
        Devuelve una lista con las salidas de las capas seleccionadas.
        """
        # Normalización (si tus espectrogramas no están en [0,1] deberás ajustarlo)
        x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features


# ---------------------------------------------------------------------------
# Composite Spectrogram Loss (Pixel + VGG Feature + VGG Style)
# ---------------------------------------------------------------------------
class VGGFeatureLoss(nn.Module):
    def __init__(
        self,
        weights: List[float],
        pixel_weight: float = 0.5,
        feature_weight: float = 0.25,
        style_weight: float = 0.25,
        vgg_layers: List[int] = [3, 8, 17, 26],
    ):
        """
        Args:
            weights (list of float): Pesos para cada canal (se normalizan internamente).
            pixel_weight (float): Peso de la pérdida pixel-level L2.
            feature_weight (float): Peso de la pérdida de reconstrucción de características.
            style_weight (float): Peso de la pérdida de estilo.
            vgg_layers (list of int): Índices de capas de VGG de las cuales extraer características.
        """
        super().__init__()
        # Normalización de pesos de canales
        self.channel_weights = [w / sum(weights) for w in weights]
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.style_weight = style_weight

        self.mse_loss = nn.MSELoss()
        self.vgg = VGGFeatureExtractor(selected_layers=vgg_layers)
        self.vgg.eval()  # Usamos VGG en modo evaluación

    def gram_matrix(self, x):
        """
        Calcula la Gram matrix para un mapa de características x.
        x: tensor de forma [N, C, H, W]
        Devuelve: tensor de forma [N, C, C]
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalizamos por el número total de elementos
        return gram / (c * h * w)

    def prepare_for_vgg(self, img):
        """
        Prepara el tensor `img` para que tenga forma [N, 3, H, W].
        Puede recibir:
            - Un tensor sin batch: [H, W]
            - Un tensor batched: [B, H, W]
        """
        if img.dim() == 2:
            # Un solo ejemplo: [H, W] -> [1, 1, H, W]
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            # Batched: [B, H, W] -> [B, 1, H, W]
            img = img.unsqueeze(1)
        if img.size(1) != 3:
            # Replicamos el canal (imagen en escala de grises a 3 canales)
            img = img.repeat(1, 3, 1, 1)
        return img

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tensor de salida. Puede tener forma:
                        - Unbatched: [C, H, W]  (cada canal es un espectrograma)
                        - Batched:   [B, C, H, W]
            targets: Tensor objetivo con la misma forma que outputs.
        Devuelve:
            Pérdida total compuesta, sumando (por canal) la pérdida pixel-level,
            la pérdida de características y la pérdida de estilo, cada una ponderada.
        """
        total_loss = 0.0

        if outputs.dim() == 3:
            # Unbatched: forma [C, H, W]
            for i, weight in enumerate(self.channel_weights):
                # 1. Pixel-level loss (L2)
                pixel_loss = self.mse_loss(outputs[i], targets[i])

                # 2. Preparar la imagen para VGG:
                out_img = self.prepare_for_vgg(outputs[i])  # [1, 3, H, W]
                tar_img = self.prepare_for_vgg(targets[i])

                # 3. Extraer características de VGG
                out_features = self.vgg(out_img)
                tar_features = self.vgg(tar_img)

                feature_loss = 0.0
                style_loss = 0.0
                for out_feat, tar_feat in zip(out_features, tar_features):
                    # Pérdida de reconstrucción de características (L2 entre mapas)
                    feature_loss += self.mse_loss(out_feat, tar_feat)
                    # Pérdida de estilo: L2 entre las Gram matrices
                    out_gram = self.gram_matrix(out_feat)
                    tar_gram = self.gram_matrix(tar_feat)
                    style_loss += self.mse_loss(out_gram, tar_gram)

                composite_loss = (
                    self.pixel_weight * pixel_loss
                    + self.feature_weight * feature_loss
                    + self.style_weight * style_loss
                )
                total_loss += weight * composite_loss

        elif outputs.dim() == 4:
            # Batched: forma [B, C, H, W]
            for i, weight in enumerate(self.channel_weights):
                # 1. Pixel-level loss (L2) para el canal i en todos los ejemplos
                pixel_loss = self.mse_loss(outputs[:, i, :, :], targets[:, i, :, :])

                # 2. Preparar imágenes para VGG:
                out_img = self.prepare_for_vgg(outputs[:, i, :, :])  # [B, 3, H, W]
                tar_img = self.prepare_for_vgg(targets[:, i, :, :])

                # 3. Extraer características de VGG (se procesan en batch)
                out_features = self.vgg(out_img)
                tar_features = self.vgg(tar_img)

                feature_loss = 0.0
                style_loss = 0.0
                for out_feat, tar_feat in zip(out_features, tar_features):
                    feature_loss += self.mse_loss(out_feat, tar_feat)
                    out_gram = self.gram_matrix(out_feat)
                    tar_gram = self.gram_matrix(tar_feat)
                    style_loss += self.mse_loss(out_gram, tar_gram)

                composite_loss = (
                    self.pixel_weight * pixel_loss
                    + self.feature_weight * feature_loss
                    + self.style_weight * style_loss
                )
                total_loss += weight * composite_loss

        else:
            raise ValueError(
                "Dimensiones inesperadas: outputs debe tener 3D [C,H,W] o 4D [B,C,H,W]"
            )

        return total_loss
