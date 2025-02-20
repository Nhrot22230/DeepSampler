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
    """
    Compute the log-magnitude spectrogram of a waveform.

    Args:
        waveform (torch.Tensor): Input waveform tensor of shape (channels, samples).
        n_fft (int): FFT size.
        hop_length (int): Hop size between frames.
        return_phase (bool): If True, returns a tuple (log_magnitude, phase).

    Returns:
        If return_phase is False:
            torch.Tensor: Log-magnitude spectrogram of shape (..., freq, time).
        If return_phase is True:
            Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is the log-magnitude
            spectrogram and the second element is the phase spectrogram.
    """
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=lambda x: torch.hann_window(x, device=waveform.device),
    )
    magnitude = stft(waveform)
    return magnitude


def i_mag_stft(
    spectrogram: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Reconstruct the waveform from a log-magnitude spectrogram and its phase.

    This function inverts the log1p operation (using expm1) to recover the magnitude,
    then combines it with the provided phase to form a complex spectrogram before computing
    the inverse STFT.

    Args:
        log_magnitude (torch.Tensor): Log-magnitude spectrogram of shape (..., freq, time).
        phase (torch.Tensor): Phase spectrogram of shape (..., freq, time).
        n_fft (int): FFT size.
        hop_length (int): Hop size between frames.

    Returns:
        torch.Tensor: Reconstructed waveform tensor (channels, samples), normalized to [-1, 1].
    """
    gm = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=lambda x: torch.hann_window(x, device=spectrogram.device),
    )

    return gm(spectrogram)


def wiener_filter(
    mix_spec: torch.Tensor, target_spec: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Wiener filtering using estimated source spectrogram.

    Args:
        mix_spec: (..., freq, time) complex mixture spectrogram
        target_spec: (..., freq, time) estimated target magnitude
    Returns:
        filtered_spec: (..., freq, time) complex filtered spectrogram
    """
    mask = target_spec / (torch.abs(mix_spec) + eps)
    return mix_spec * mask
