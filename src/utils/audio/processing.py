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
        chunk = waveform[..., i : i + chunk_len]
        if chunk.shape[-1] < chunk_len:
            chunk = torch.cat(
                [chunk, torch.zeros_like(chunk[..., : chunk_len - chunk.shape[-1]])],
                dim=-1,
            )
        chunks.append(chunk)

    return chunks


def log_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute log magnitude spectrogram.

    Args:
        waveform: (channels, samples) tensor
        n_fft: FFT size
        hop_length: Hop size between frames
        window: Window function

    Returns:
        spectrogram: (..., freq, time) tensor
    """
    if window is None:
        window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    return torch.log1p(torch.abs(stft))


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: typing.Optional[float] = None,
) -> torch.Tensor:
    """Compute mel spectrogram.

    Args:
        waveform: (channels, samples) tensor
        sample_rate: Sample rate of audio
        n_fft: FFT size
        hop_length: Hop size between frames
        n_mels: Number of mel bins
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        spectrogram: (..., freq, time) tensor
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    return mel_transform(waveform)


def inverse_log_spectrogram(
    spectrogram: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the inverse of a log-magnitude spectrogram.

    Args:
        spectrogram: (..., freq, time) tensor containing log-magnitude values.
        n_fft: FFT size.
        hop_length: Hop size between frames.
        window: Window function; if None, a Hann window is used.

    Returns:
        waveform: Reconstructed waveform (channels, samples) tensor.
    """
    if window is None:
        window = torch.hann_window(n_fft, device=spectrogram.device)

    # Invert the log transform: log(1 + x) --> x = expm1(log_value)
    magnitude = torch.expm1(spectrogram)

    # Reconstruct a complex spectrogram with zero phase.
    # Esto crea un tensor complejo donde la parte imaginaria es cero.
    complex_spec = torch.complex(magnitude, torch.zeros_like(magnitude))

    # Calcular la longitud estimada del waveform.
    # Se asume que el número de frames es spectrogram.shape[-1].
    length = spectrogram.shape[-1] * hop_length

    # Aplicar la ISTFT.
    waveform = torch.istft(
        complex_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=length
    )

    return waveform.float()
