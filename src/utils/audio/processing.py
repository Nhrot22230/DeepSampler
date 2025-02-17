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
        waveform = waveform.mean(dim=0)

    return waveform


def chunk_waveform(
    waveform: torch.Tensor, chunk_len: int, hop_len: int
) -> typing.List[torch.Tensor]:
    """
    Divide un waveform en segmentos (chunks) solapados utilizando torch.unfold.
    Args:
        waveform (torch.Tensor): Tensor que contiene el waveform.
        chunk_len (int): Longitud de cada chunk.
        hop_len (int): Paso (hop) entre el inicio de cada chunk.
    Returns:
        List[torch.Tensor]: Lista de chunks. En el caso de waveform 1D, cada chunk es 1D;
                            para 2D, cada chunk es un tensor de forma (channels, chunk_len).
    """
    if waveform.dim() == 1:
        chunks = waveform.unfold(dimension=0, size=chunk_len, step=hop_len)
    else:
        raise ValueError("waveform debe ser 1D o 2D.")

    return [chunk.clone() for chunk in chunks]


def mag_stft(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Compute log magnitude spectrogram.
    Args:
        waveform: (channels, samples) tensor
        n_fft: FFT size
        hop_length: Hop size between frames
    Returns:
        spectrogram: (..., freq, time) tensor
    """
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    return torch.abs(stft)


def i_mag_stft(
    spectrogram: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute the inverse of a log-magnitude spectrogram.
    Args:
        spectrogram: (..., freq, time) tensor containing log-magnitude values.
        n_fft: FFT size.
        hop_length: Hop size between frames.
    Returns:
        waveform: Reconstructed waveform (channels, samples) tensor.
    """
    window = torch.hann_window(n_fft, device=spectrogram.device)

    # Reconstruct a complex spectrogram with zero phase.
    complex_spec = torch.complex(spectrogram, torch.zeros_like(spectrogram))
    length = spectrogram.shape[-1] * hop_length

    waveform = torch.istft(
        complex_spec, n_fft=n_fft, hop_length=hop_length, window=window, length=length
    )

    return waveform.float()
