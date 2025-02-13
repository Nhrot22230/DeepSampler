import torch.nn as nn


class MultiSourceL1Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1_loss = nn.L1Loss(reduction="mean")

    def forward(self, outputs, targets):
        total_loss = 0
        for i, weight in enumerate(self.weights):
            total_loss += weight * self.l1_loss(outputs[:, i], targets[:, i])
        return total_loss
