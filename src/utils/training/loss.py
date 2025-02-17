import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


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


class MultiSourceSeparationLoss(nn.Module):
    def __init__(self, weights, frequency_weights=None):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights /= self.weights.sum()

        # Initialize loss components
        self.spectral_convergence = SpectralConvergenceLoss()
        self.magnitude_loss = nn.L1Loss()

        # Frequency weighting (perceptual emphasis)
        self.frequency_weights = (
            frequency_weights if frequency_weights is not None else 1.0
        )

    def forward(self, outputs, targets):
        """
        outputs: (batch, channels, freq_bins, time_frames) - predicted spectrograms
        targets: (batch, channels, freq_bins, time_frames) - target spectrograms
        """
        total_loss = 0.0

        for i, weight in enumerate(self.weights):
            pred = outputs[:, i] * self.frequency_weights.to(outputs.device)
            targ = targets[:, i] * self.frequency_weights.to(targets.device)

            channel_loss = 0.5 * self.magnitude_loss(
                pred, targ
            ) + 0.5 * self.spectral_convergence(pred, targ)

            total_loss += weight * channel_loss

        return total_loss


class SpectralConvergenceLoss(nn.Module):
    def forward(self, pred, target):
        return torch.norm(target - pred, p="fro") / (torch.norm(target, p="fro") + 1e-6)


def create_mel_weights(n_fft=2048, sr=44100, n_mels=128, device="cpu"):
    """Create perceptual frequency weighting matrix"""
    mel_fb = (
        torchaudio.transforms.MelScale(
            n_mels=n_mels, n_stft=n_fft // 2 + 1, sample_rate=sr
        )
        .fb.clone()
        .to(device)
    )
    return mel_fb.mean(dim=0)  # Average across mel bands
