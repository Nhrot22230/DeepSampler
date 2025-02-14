from typing import Dict, List
import numpy as np
import torch
from src.utils.audio.processing import (
    chunk_waveform,
    load_audio,
    inverse_log_spectrogram,
)
from tqdm import tqdm


def overlap_add(
    chunks: List[torch.Tensor], chunk_len: int, hop: int, window: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruye el waveform completo a partir de chunks superpuestos
    utilizando overlap-add y aplicando una ventana.

    Args:
        chunks (List[torch.Tensor]): Lista de chunks (forma: (channels, chunk_len)).
        chunk_len (int): Longitud de cada chunk (en muestras).
        hop (int): Desplazamiento entre chunks.
        window (torch.Tensor): Ventana a aplicar (forma: (1, chunk_len)).

    Returns:
        torch.Tensor: Waveform reconstruido (forma: (channels, total_length)).
    """
    num_chunks = len(chunks)
    total_length = (num_chunks - 1) * hop + chunk_len
    device = chunks[0].device
    dtype = chunks[0].dtype
    channels = chunks[0].shape[0]

    # Inicializar tensores para la señal reconstruida y para acumular pesos de la ventana.
    output = torch.zeros((channels, total_length), device=device, dtype=dtype)
    weight = torch.zeros((channels, total_length), device=device, dtype=dtype)

    # Asegurarse de que la ventana tenga forma (1, chunk_len)
    if window.ndim == 1:
        window = window.unsqueeze(0)

    # Realizar overlap-add para cada chunk
    for i, chunk in enumerate(chunks):
        start = i * hop
        end = start + chunk_len
        output[:, start:end] += chunk * window
        weight[:, start:end] += window

    # Evitar división por cero y normalizar la señal reconstruida.
    weight[weight == 0] = 1.0
    output /= weight
    return output


def infer_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    overlap: float = 0.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, np.ndarray]:
    """
    Pipeline de inferencia para separación de fuentes.
    Se carga un archivo mixture.wav, se lo segmenta en chunks (con o sin solapamiento),
    se procesa cada chunk con el modelo para obtener los espectrogramas de cada fuente,
    se invierten los espectrogramas a waveform y finalmente se reconstruye la señal completa
    aplicando overlap-add con una ventana de Hann.

    Args:
        model (torch.nn.Module): Modelo de separación para la inferencia.
        mixture_path (str): Ruta al archivo mixture.wav.
        sample_rate (int, optional): Frecuencia de muestreo del audio. Defaults a 44100.
        chunk_seconds (float, optional): Duración de cada chunk en segundos. Defaults a 2.
        overlap (float, optional): Fracción de solapamiento entre chunks (0.0 a <1.0).
        n_fft (int, optional): Número de puntos para la FFT. Defaults a 2048.
        hop_length (int, optional): Salto para la STFT. Defaults a 512.
        device (torch.device, optional): Dispositivo para la inferencia. Defaults a CPU.

    Returns:
        Dict[str, np.ndarray]: Diccionario con las señales separadas para cada instrumento.
    """
    # Calcular la longitud del chunk en muestras y el hop para chunking considerando.
    chunk_len = int(chunk_seconds * sample_rate)
    chunk_hop = int(chunk_len * (1 - overlap))

    # Cargar la mezcla y segmentarla en chunks (posiblemente con solapamiento).
    mixture_waveform = load_audio(mixture_path, sample_rate)
    mixture_chunks = chunk_waveform(mixture_waveform, chunk_len, chunk_hop)

    # Definir las fuentes a separar.
    instruments = ["vocals", "drums", "bass", "other"]
    # Inicializar diccionario para almacenar los chunks predichos para cada instrumento.
    separated_chunks: Dict[str, List[torch.Tensor]] = {
        instrument: [] for instrument in instruments
    }

    # Procesar cada chunk a través del modelo.
    for chunk in tqdm(mixture_chunks, desc="Separating audio"):
        chunk = chunk.to(device)
        with torch.no_grad():
            # Se asume que el modelo devuelve un tensor donde la primera dimensión indexa
            # las predicciones para cada instrumento.
            pred: torch.Tensor = model(chunk)
        for i, instrument in enumerate(instruments):
            # Invertir el espectrograma logarítmico para obtener el waveform.
            waveform_chunk = inverse_log_spectrogram(pred[i], n_fft, hop_length)
            separated_chunks[instrument].append(waveform_chunk)

    # Reconstruir el waveform completo para cada instrumento utilizando overlap-add.
    separated_audio: Dict[str, np.ndarray] = {}
    window = torch.hann_window(chunk_len, device=device).unsqueeze(
        0
    )  # Forma: (1, chunk_len)
    for instrument in instruments:
        reconstructed = overlap_add(
            separated_chunks[instrument], chunk_len, chunk_hop, window
        )
        separated_audio[instrument] = reconstructed.cpu().numpy()

    return separated_audio
