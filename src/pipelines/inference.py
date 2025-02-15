from typing import Dict, List

import numpy as np
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
) -> Dict[str, np.ndarray]:
    """
    Pipeline de inferencia para separación de fuentes.
    Se carga un archivo mixture.wav, se lo segmenta en chunks (con o sin solapamiento),
    se calcula el espectrograma logarítmico de cada chunk, se procesa cada chunk con el modelo
    para obtener los espectrogramas de cada fuente, se invierten los espectrogramas a waveform y
    finalmente se reconstruye la señal completa concatenando los chunks.

    Args:
        model (torch.nn.Module): Modelo de separación para la inferencia.
        mixture_path (str): Ruta al archivo mixture.wav.
        sample_rate (int, optional): Frecuencia de muestreo del audio. Defaults a 44100.
        chunk_seconds (float, optional): Duración de cada chunk en segundos. Defaults a 2.
        overlap (float, optional): Fracción de solapamiento entre chunks (0.0 a <1.0). Defaults a 0.
        n_fft (int, optional): Número de puntos para la FFT. Defaults a 2048.
        hop_length (int, optional): Salto para la STFT. Defaults a 512.
        device (torch.device, optional): Dispositivo para la inferencia. Defaults a CPU.

    Returns:
        Dict[str, np.ndarray]: Diccionario con las señales separadas para cada instrumento.
    """
    # Calcular longitud de chunk y salto (hop) en muestras.
    chunk_len = int(chunk_seconds * sample_rate)
    chunk_hop = int(chunk_len * (1 - overlap))

    # Cargar la mezcla y dividirla en chunks.
    mixture_waveform = load_audio(mixture_path, sample_rate)
    mixture_chunks = chunk_waveform(mixture_waveform, chunk_len, chunk_hop)

    # Definir los nombres de las fuentes.
    instruments = ["vocals", "drums", "bass", "other"]
    separated_chunks: Dict[str, List[torch.Tensor]] = {inst: [] for inst in instruments}

    # Asegurarse de que el modelo esté en el dispositivo y en modo evaluación.
    model.to(device)
    model.eval()

    for chunk in tqdm(mixture_chunks, desc="Separating audio"):
        # Calcular el espectrograma logarítmico. Se asume que la forma de chunk es (C, samples)
        spec = log_spectrogram(chunk, n_fft, hop_length)  # forma: (C, F, T)
        # Asegurar que el tensor tenga batch: (B, C, F, T)
        spec = spec.to(device).unsqueeze(0)

        with torch.no_grad():
            # Se espera que el modelo devuelva un tensor de forma (B, num_instruments, F, T)
            pred = model(spec)
            # Remover la dimensión batch: forma (num_instruments, F, T)
            pred = pred.squeeze(0)

        # Para cada instrumento, reconstruir el waveform del chunk.
        for i, inst in enumerate(instruments):
            # Se asume que inverse_log_spectrogram recibe un tensor de forma (F, T)
            waveform_chunk = inverse_log_spectrogram(pred[i], n_fft, hop_length)
            separated_chunks[inst].append(waveform_chunk)

    # Concatenar los chunks a lo largo del eje temporal para cada fuente.
    separated_audio: Dict[str, np.ndarray] = {}
    for inst in instruments:
        # Se concatena en la dimensión del tiempo (dim=1)
        reconstructed = torch.cat(separated_chunks[inst])
        separated_audio[inst] = reconstructed.cpu().numpy()

    return separated_audio
