import os
from typing import List, Optional

import torch
from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import chunk_waveform, load_audio
from src.utils.data.dataset import MUSDB18Dataset
from src.utils.logging import main_logger as logger
from tqdm import tqdm


def _load_and_chunk(
    file_path: str, chunk_len: int, hop_len: int, sample_rate: int
) -> Optional[List[torch.Tensor]]:
    """
    Función auxiliar que carga un archivo de audio y lo divide en chunks.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Archivo no encontrado: {file_path}")
        return None
    try:
        audio = load_audio(file_path, sample_rate)
        return chunk_waveform(audio, chunk_len, hop_len)
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {e}")
        return None


def process_track(
    track_folder: str,
    chunk_seconds: int = 2,
    overlap: float = 0,
    sample_rate: int = 44100,
    instruments: Optional[List[str]] = None,
) -> List[AudioChunk]:
    """
    Procesa una pista completa cargando y segmentando los archivos de audio.

    Args:
        track_folder (str): Ruta a la carpeta de la pista.
        chunk_seconds (int, optional): Duración de cada segmento en segundos.
        overlap (float, optional): Porcentaje de solapamiento entre segmentos.
        sample_rate (int, optional): Frecuencia de muestreo para cargar el audio.
        instruments (List[str], optional): Lista de nombres de instrumentos a procesar.
            Defaults a ["bass", "drums", "vocals", "other"].

    Returns:
        List[AudioChunk]: Lista de chunks procesados, donde cada chunk es un diccionario:
            "mixture", "bass", "drums", "vocals" y "other".
    """
    if instruments is None:
        instruments = ["bass", "drums", "vocals", "other"]

    chunk_len = chunk_seconds * sample_rate
    hop_len = int(chunk_len * (1 - overlap))

    mixture_path = os.path.join(track_folder, "mixture.wav")
    mixture_chunks = _load_and_chunk(mixture_path, chunk_len, hop_len, sample_rate)
    if mixture_chunks is None:
        logger.error(f"No se pudo cargar la mezcla en {track_folder}")
        return []

    chunks: List[AudioChunk] = [
        AudioChunk(mixture=chunk, bass=None, drums=None, vocals=None, other=None)
        for chunk in mixture_chunks
    ]

    for instrument in instruments:
        file_path = os.path.join(track_folder, f"{instrument}.wav")
        instrument_chunks = _load_and_chunk(file_path, chunk_len, hop_len, sample_rate)
        if instrument_chunks is None:
            continue
        num_chunks = min(len(chunks), len(instrument_chunks))
        for idx in range(num_chunks):
            chunks[idx][instrument] = instrument_chunks[idx]

    return chunks


def musdb_pipeline(
    musdb_path: str,
    window: torch.Tensor,
    chunk_seconds: int = 2,
    overlap: float = 0.0,
    sample_rate: int = 44100,
    nfft: int = 2048,
    hop_length: int = 512,
    max_samples: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """
    Procesa el dataset MUSDB18HQ para entrenar un modelo de separación de fuentes.

    Args:
        musdb_path (str): Ruta a la carpeta con las pistas de MUSDB18HQ.
        window (torch.Tensor): Ventana a utilizar en la STFT.
        chunk_seconds (int, optional): Duración de cada segmento en segundos.
        overlap (float, optional): Porcentaje de solapamiento entre segmentos.
        sample_rate (int, optional): Frecuencia de muestreo para cargar el audio.
        nfft (int, optional): Tamaño de la ventana para la transformada de Fourier.
        hop_length (int, optional): Tamaño del salto para la transformada de Fourier.

    Returns:
        torch.utils.data.Dataset: Dataset de PyTorch con los datos procesados.
    """
    all_chunks = []
    track_dirs = [
        os.path.join(musdb_path, d)
        for d in os.listdir(musdb_path)
        if os.path.isdir(os.path.join(musdb_path, d))
    ]

    for track_dir in tqdm(track_dirs, desc="Procesando pistas"):
        track_chunks = process_track(
            track_folder=track_dir,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            sample_rate=sample_rate,
        )
        all_chunks.extend(track_chunks)

        if max_samples is not None and len(all_chunks) >= max_samples:
            all_chunks = all_chunks[:max_samples]
            tqdm.write(f"Se han procesado {max_samples} segmentos.")
            break

    return MUSDB18Dataset(
        data=all_chunks,
        window=window,
        nfft=nfft,
        hop_length=hop_length,
    )
