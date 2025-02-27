import torch
import torch.nn as nn


class MultiSourceLoss(nn.Module):
    """
    Weighted multi-source loss function for spectrogram reconstruction.

    Args:
        weights: Relative weights for each source channel (will be normalized).
        distance: Distance metric ('l1' or 'l2').
        reduction: Loss reduction method over the batch ('mean' or 'sum').

    Inputs:
        outputs: (batch_size, num_sources, freq_bins, time_steps)
        targets: (batch_size, num_sources, freq_bins, time_steps)

    Returns:
        Weighted combination of per-source losses.
    """

    def __init__(self, weights: list, distance: str = "l1", reduction: str = "mean"):
        super().__init__()
        # Normalize the weights
        normalized_weights = [w / sum(weights) for w in weights]
        self.weights = torch.tensor(normalized_weights, dtype=torch.float32)
        self.reduction = reduction.lower()

        # Use loss functions with reduction="none" to compute per-element loss
        if distance.lower() == "l1":
            self.loss = nn.L1Loss(reduction="none")
        elif distance.lower() == "l2":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if outputs.shape != targets.shape:
            raise ValueError("Output and target shapes must match")

        # If there is only one source (3D tensor), add the channel dimension.
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

        # [BATCH_SIZE, NUM_SOURCES, FREQ_BINS, TIME_STEPS]
        loss = self.loss(outputs, targets)
        # Average loss over the frequency and time dimensions for each source:
        # new shape will be [BATCH_SIZE, NUM_SOURCES]
        loss = loss.mean(dim=(-2, -1))

        # Ensure weights are on the same device and apply per-source weighting.
        weighted_loss = loss * self.weights.to(loss.device)
        # Sum over sources to get a per-sample loss, shape [BATCH_SIZE]
        sample_loss = weighted_loss.sum(dim=1)

        # Reduce over the batch dimension
        if self.reduction == "mean":
            return sample_loss.mean()
        elif self.reduction == "sum":
            return sample_loss.sum()
        else:
            # If no reduction is specified, return the loss for each sample.
            return sample_loss
