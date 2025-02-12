import typing

import torch
import torchaudio


def load_audio(path: str, target_sr: int = 44100, mono: bool = True) -> torch.Tensor:
    """Load audio file to waveform tensor.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (default: 44100)
        mono: Convert to mono (default: True)

    Returns:
        waveform: (channels, samples) tensor
    """
    waveform, sr = torchaudio.load(path, normalize=True, channels_first=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def chunk_waveform(
    waveform: torch.Tensor, chunk_len: int, hop_len: int
) -> typing.List[torch.Tensor]:
    """Chunk waveform into overlapping segments.

    Args:
        waveform: (channels, samples) tensor
        chunk_len: Length of each chunk
        hop_len: Hop size between chunks

    Returns:
        chunks: List of (channels, chunk_len) tensors
    """
    chunks = []
    for i in range(0, waveform.shape[-1] - chunk_len + 1, hop_len):
        # pad chunk to chunk_len
        chunk = waveform[..., i : i + chunk_len]
        if chunk.shape[-1] < chunk_len:
            chunk = torch.cat(
                [chunk, torch.zeros_like(chunk[..., : chunk_len - chunk.shape[-1]])],
                dim=-1,
            )
        chunks.append(chunk)

    return chunks


def stft(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute STFT of waveform.

    Args:
        waveform: (..., samples) tensor
        n_fft: FFT size
        hop_length: Hop size between frames
        window: Optional window function

    Returns:
        spectrogram: (..., freq, time, 2) tensor (real, imaginary)
    """
    if window is None:
        window = torch.hann_window(n_fft, device=waveform.device)

    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        onesided=True,
        return_complex=False,
    )


def istft(
    spectrogram: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: typing.Optional[torch.Tensor] = None,
    length: typing.Optional[int] = None,
) -> torch.Tensor:
    """Compute inverse STFT to waveform.

    Args:
        spectrogram: (..., freq, time, 2) tensor
        n_fft: FFT size
        hop_length: Hop size between frames
        window: Window function
        length: Original waveform length

    Returns:
        waveform: (..., samples) tensor
    """
    if window is None:
        window = torch.hann_window(n_fft, device=spectrogram.device)

    return torch.istft(
        spectrogram.permute(0, 2, 1, 3).reshape(-1, n_fft // 2 + 1, 2),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        onesided=True,
        length=length,
    )


def si_sdr_loss(
    estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.

    Args:
        estimate: Separated signal (..., samples)
        target: Reference signal (..., samples)
        eps: Numerical stability term

    Returns:
        SI-SDR loss value
    """
    # Normalize inputs
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    # Compute target projection
    alpha = (target * estimate).sum(dim=-1, keepdim=True) / (
        target.norm(dim=-1, keepdim=True).pow(2) + eps
    )

    # Calculate SI-SDR
    projection = alpha * target
    noise = estimate - projection
    return (
        -10
        * torch.log10(
            (projection.norm(dim=-1).pow(2) / (noise.norm(dim=-1).pow(2) + eps) + eps)
        ).mean()
    )
