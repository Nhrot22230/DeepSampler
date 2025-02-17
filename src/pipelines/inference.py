from typing import Dict, List

import torch
from src.utils.audio.processing import (
    chunk_waveform,
    inverse_log_spectrogram,
    load_audio,
    log_spectrogram,
)
from tqdm import tqdm


def infer_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    overlap: float = 0,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Inference pipeline for source separation.

    This function loads a mixture audio file, splits it into overlapping or non-overlapping chunks,
    computes the log-spectrogram of each chunk, processes each chunk through the model to obtain the
    spectrograms for each source, converts the predicted spectrograms back to waveforms, and
    reconstructs the complete signal by concatenating the chunks.

    Args:
        model (torch.nn.Module): The source separation model for inference.
        mixture_path (str): Path to the mixture audio file.
        sample_rate (int, optional): Audio sample rate. Defaults to 44100.
        chunk_seconds (float, optional): Duration of each chunk in seconds. Defaults to 2.
        overlap (float, optional): Fraction of overlap between chunks (0.0 to <1.0). Defaults to 0.
        n_fft (int, optional): FFT size for computing the spectrogram. Defaults to 2048.
        hop_length (int, optional): Hop length for the STFT. Defaults to 512.
        device (torch.device, optional): Device for inference. Defaults to CPU.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping instrument names to their separated waveform.
    """
    chunk_len = int(chunk_seconds * sample_rate)
    chunk_hop = int(chunk_len * (1 - overlap))
    mixture_waveform = load_audio(mixture_path, sample_rate)
    mixture_chunks = chunk_waveform(mixture_waveform, chunk_len, chunk_hop)
    instruments = ["vocals", "drums", "bass", "other"]
    separated_chunks: Dict[str, List[torch.Tensor]] = {inst: [] for inst in instruments}

    model.eval()
    for chunk in tqdm(mixture_chunks, desc="Separating audio"):
        chunk = chunk.to(device)
        spec = log_spectrogram(chunk, n_fft, hop_length)
        spec = spec.to(device)
        spec = spec.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Model output should have shape: [1, C, F, T]
            pred = model(spec)
            # Remove the batch dimension: now shape [C, F, T]
            pred = pred.squeeze(0)

        for i, inst in enumerate(instruments):
            wav_chunk = inverse_log_spectrogram(pred[i], n_fft, hop_length)
            separated_chunks[inst].append(wav_chunk)

    separated_audio: Dict[str, torch.Tensor] = {}
    for inst in instruments:
        reconstructed = torch.cat(separated_chunks[inst])
        separated_audio[inst] = reconstructed

    return separated_audio
