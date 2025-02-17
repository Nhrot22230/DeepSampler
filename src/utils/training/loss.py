import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourceLoss(nn.Module):
    def __init__(self, weights, distance="l1"):
        super().__init__()
        self.weights = torch.tensor(
            [w / sum(weights) for w in weights], dtype=torch.float32
        )
        self.loss = nn.L1Loss() if distance.lower() == "l1" else nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        total_loss = 0.0
        for i, weight in enumerate(self.weights):
            if outputs.dim() == 3:
                total_loss += weight * self.loss(outputs[i], targets[i])
            elif outputs.dim() == 4:
                total_loss += weight * self.loss(outputs[:, i], targets[:, i])
            else:
                raise ValueError("Unsupported tensor dimensions.")
        return total_loss


class MultiScaleLoss(nn.Module):
    def __init__(self, weights, scales=[1, 2, 4], distance="l1"):
        super().__init__()
        self.weights = torch.tensor(
            [w / sum(weights) for w in weights], dtype=torch.float32
        )
        self.scales = scales
        self.loss = nn.L1Loss() if distance.lower() == "l1" else nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        total_loss = 0.0
        for i, weight in enumerate(self.weights):
            channel_loss = 0.0
            for scale in self.scales:
                if isinstance(outputs, list):
                    out_scaled = self._downsample(outputs[i], scale)
                    tar_scaled = self._downsample(targets[i], scale)
                elif outputs.dim() == 3:
                    out_scaled = self._downsample(outputs[i], scale)
                    tar_scaled = self._downsample(targets[i], scale)
                elif outputs.dim() == 4:
                    out_scaled = self._downsample(outputs[:, i], scale)
                    tar_scaled = self._downsample(targets[:, i], scale)
                else:
                    raise ValueError("Unsupported input format.")
                channel_loss += self.loss(out_scaled, tar_scaled)
            channel_loss /= len(self.scales)
            total_loss += weight * channel_loss
        return total_loss

    def _downsample(self, x: torch.Tensor, scale: int):
        if scale == 1:
            return x
        return F.avg_pool2d(
            x.unsqueeze(0) if x.dim() == 2 else x, kernel_size=scale, stride=scale
        ).squeeze(0)
