from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

from src.pipelines.inference import infer_pipeline
from src.utils.audio.processing import load_audio


def eval_pipeline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    overlap: float = 0.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Pipeline de evaluación que:
      - Separa la señal de mezcla utilizando el infer_pipeline.
      - Calcula el SI‑SDR entre las señales separadas y las señales de referencia (ground truth)
        para cada fuente.

    Se espera que cada muestra en el DataLoader sea un diccionario con las claves:
        - "mixture": Ruta al archivo mixture.wav.
        - "vocals", "drums", "bass", "other": Rutas a los archivos de ground truth para cada fuente.

    Args:
        model (torch.nn.Module): Modelo de separación.
        dataloader (DataLoader): DataLoader que provee muestras de evaluación.
        sample_rate (int, optional): Frecuencia de muestreo. Defaults a 44100.
        chunk_seconds (float, optional): Duración de cada chunk en segundos. Defaults a 2.
        overlap (float, optional): Fracción de solapamiento entre chunks (0.0 a <1.0). Defaults a 0.
        n_fft (int, optional): Número de puntos para la FFT. Defaults a 2048.
        hop_length (int, optional): Salto para la STFT. Defaults a 512.
        device (torch.device, optional): Dispositivo para la inferencia. Defaults a CPU.

    Returns:
        Tuple[Dict[str, float], Dict[str, List[float]]]:
            - Diccionario con el SI‑SDR promedio para cada fuente.
            - Diccionario con la lista de SI‑SDR para cada muestra por fuente.
    """
    instruments = ["vocals", "drums", "bass", "other"]
    all_scores: Dict[str, List[float]] = {inst: [] for inst in instruments}

    for sample in tqdm(dataloader, desc="Evaluating"):
        mixture_path = sample["mixture"]
        separated_audio = infer_pipeline(
            model=model,
            mixture_path=mixture_path,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device,
        )

        for inst in instruments:
            gt_path = sample[inst]
            gt_audio = load_audio(gt_path, sample_rate).to(device)
            pred_audio = torch.tensor(separated_audio[inst], device=device)
            min_len = min(gt_audio.shape[1], pred_audio.shape[1])
            gt_audio = gt_audio[:, :min_len]
            pred_audio = pred_audio[:, :min_len]
            si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
            score = si_sdr_metric(pred_audio, gt_audio)
            all_scores[inst].append(score.item())

    avg_scores = {
        inst: np.mean(scores) if scores else 0.0 for inst, scores in all_scores.items()
    }
    return avg_scores, all_scores
