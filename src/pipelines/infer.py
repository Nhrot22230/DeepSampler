from typing import Dict, List, Union

import torch
from tqdm.auto import tqdm

from src.utils.audio.processing import chunk_waveform, i_mag_stft, load_audio, mag_stft


def infer_pipeline(
    model: torch.nn.Module,
    mixture: Union[str, torch.Tensor],
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    overlap: float = 0,
    n_iter: int = 1024,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Inference pipeline for source separation.

    This function loads a mixture audio file, splits it into overlapping or non-overlapping chunks,
    computes the log-spectrogram of each chunk, processes each chunk through the model to obtain the
    spectrograms for each source, converts the predicted spectrograms back to waveforms using the
    inverse log-magnitude function and reconstructs the complete signal by concatenating the chunks.

    Args:
        model (torch.nn.Module): The source separation model for inference.
        mixture (Union[str, torch.Tensor]): Path to the mixture audio file or the waveform tensor.
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

    if isinstance(mixture, str):
        mixture_waveform = load_audio(mixture, sample_rate)
    else:
        mixture_waveform = mixture
    mixture_waveform = mixture_waveform.to(device)
    mixture_chunks = chunk_waveform(mixture_waveform, chunk_len, chunk_hop)
    instruments = ["vocals", "drums", "bass", "other"]
    separated_chunks: Dict[str, List[torch.Tensor]] = {inst: [] for inst in instruments}

    model.eval()
    model.to(device)

    with torch.no_grad():
        for chunk in tqdm(mixture_chunks, desc="Processing chunks"):
            spec = mag_stft(chunk, n_fft, hop_length)
            spec = spec.unsqueeze(0).unsqueeze(0)
            pred = model(spec)  # Expected shape: [1, C, F, T]
            pred = pred.squeeze(0)  # Now shape: [C, F, T]
            for i, inst in enumerate(instruments):
                wav_chunk = i_mag_stft(pred[i], n_fft, hop_length, n_iter=n_iter)
                separated_chunks[inst].append(wav_chunk)
                del wav_chunk
            del spec, pred

    separated_audio: Dict[str, torch.Tensor] = {}
    for inst in instruments:
        # Usamos torchaudio.functional.fade y concatenación inteligente
        chunks_to_join = []
        for i, chunk in enumerate(separated_chunks[inst]):
            chunks_to_join.append(chunk)

        # Concatenación final con torch.cat
        separated_audio[inst] = torch.cat(chunks_to_join, dim=-1)

    return separated_audio

    return separated_audio
