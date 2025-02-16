from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weighted L1 Loss (MultiSourceL1Loss)
# ---------------------------------------------------------------------------
class MultiSourceLoss(nn.Module):
    """
    Weighted L1/L2 Loss for multi-source signals.

    Args:
        weights (List[float]): A list of weights for each channel.
        distance (str, optional): Distance metric to use ("l1" or "l2"). Defaults to "l1".
    """

    def __init__(self, weights: List[float], distance: str = "l1") -> None:
        super().__init__()
        self.weights = [w / sum(weights) for w in weights]

        if distance.lower() == "l1":
            self.loss = nn.L1Loss(reduction="mean")
        elif distance.lower() == "l2":
            self.loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Invalid distance: {distance}")

    def forward(
        self, outputs: List[torch.Tensor], targets: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the weighted loss for each channel and returns the total loss.

        Args:
            outputs (List[torch.Tensor]): List of predicted tensors.
            targets (List[torch.Tensor]): List of ground truth tensors.

        Returns:
            torch.Tensor: The computed weighted loss.
        """
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

        if distance.lower() == "l1":
            self.loss = nn.L1Loss(reduction="mean")
        elif distance.lower() == "l2":
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
