import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weighted L1 Loss (MultiSourceL1Loss)
# ---------------------------------------------------------------------------
class MultiSourceL1Loss(nn.Module):
    def __init__(self, weights):
        """
        Args:
            weights (list of float): A list of weights for each channel.
        """
        super().__init__()
        self.weights = weights
        self.l1_loss = nn.L1Loss(reduction="mean")

    def forward(self, outputs, targets):
        total_loss = 0.0
        for i, weight in enumerate(self.weights):
            total_loss += weight * self.l1_loss(outputs[i], targets[i])
        return total_loss


# ---------------------------------------------------------------------------
# Weighted Multi-Scale Spectral Loss (MultiSourceMultiScaleSpectralLoss)
# ---------------------------------------------------------------------------
class MultiSourceMultiScaleSpectralLoss(nn.Module):
    def __init__(self, weights, scales=[1, 2, 4], distance="l1"):
        """
        Args:
            weights (list of float): List of weights for each channel.
            scales (list of int): Downsampling factors to simulate multiple scales.
                                  A scale of 1 means no downsampling.
            distance (str): Distance metric to use ("l1" or "l2").
        """
        super().__init__()
        self.weights = weights
        self.scales = scales
        self.distance = distance.lower()
        if self.distance not in ["l1", "l2"]:
            raise ValueError("Distance must be either 'l1' or 'l2'.")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Predicted spectrogram, shape [channels, width, height].
            targets (torch.Tensor): Target spectrogram, shape [channels, width, height].
        Returns:
            torch.Tensor: Scalar loss.
        """
        if outputs.shape != targets.shape:
            raise ValueError("Output and target shapes must match.")

        num_channels = outputs.shape[0]
        total_loss = 0.0

        for c in range(num_channels):
            channel_loss = 0.0
            out_channel = outputs[c].unsqueeze(0).unsqueeze(0)
            tar_channel = targets[c].unsqueeze(0).unsqueeze(0)
            for scale in self.scales:
                if scale == 1:
                    out_scaled = out_channel
                    tar_scaled = tar_channel
                else:
                    out_scaled = F.avg_pool2d(
                        out_channel, kernel_size=scale, stride=scale
                    )
                    tar_scaled = F.avg_pool2d(
                        tar_channel, kernel_size=scale, stride=scale
                    )

                if self.distance == "l1":
                    scale_loss = torch.mean(torch.abs(out_scaled - tar_scaled))
                else:
                    scale_loss = torch.mean((out_scaled - tar_scaled) ** 2)
                channel_loss += scale_loss

            channel_loss /= len(self.scales)
            total_loss += self.weights[c] * channel_loss

        return total_loss
