# =============================================================================
# PREPROCESAMIENTO: Generar y almacenar chunks en archivos .npy
# =============================================================================

import logging
import os
from typing import Dict, List

import numpy as np
from src.utils.audio.processing import chunk_audio, load_audio
from tqdm import tqdm

SR = 44100  # Frecuencia de muestreo
N_FFT = 2048  # Tamaño de la FFT
HOP_LENGTH = 512  # Hop length para la STFT
CHUNK_DURATION = 5.0  # Duración de cada chunk en segundos
OVERLAP_FRACTION = 0.2  # Fracción de solapamiento entre chunks


class TqdmLoggingHandler(logging.Handler):
    """
    Handler de logging que escribe mensajes usando
    tqdm.write para evitar que se interfiera con la barra de progreso.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def process_song(song_folder: str) -> List[Dict[str, np.ndarray]]:
    """
    Procesa una carpeta de canción:
    - Carga y normaliza cada canal (mezcla y 4 stems).
    - Calcula los espectrogramas logarítmicos.
    - Realiza chunking con ventana deslizante.
    - Guarda cada chunk en un archivo .npy.

    Cada chunk se guarda como un diccionario con las claves:
        "mixture", "drums", "bass", "vocals", "other"
    """

    mixture_path = os.path.join(song_folder, "mixture.wav")
    drums_path = os.path.join(song_folder, "drums.wav")
    bass_path = os.path.join(song_folder, "bass.wav")
    vocals_path = os.path.join(song_folder, "vocals.wav")
    other_path = os.path.join(song_folder, "other.wav")

    try:
        mixture_audio = load_audio(mixture_path)
        drums_audio = load_audio(drums_path)
        bass_audio = load_audio(bass_path)
        vocals_audio = load_audio(vocals_path)
        other_audio = load_audio(other_path)
    except ValueError as e:
        logging.error(e)
        return

    mixture_chunks = chunk_audio(mixture_audio)
    drums_chunks = chunk_audio(drums_audio)
    bass_chunks = chunk_audio(bass_audio)
    vocals_chunks = chunk_audio(vocals_audio)
    other_chunks = chunk_audio(other_audio)

    chunk_list: List[Dict[str, np.ndarray]] = []

    for mixture_chunk, drums_chunk, bass_chunk, vocals_chunk, other_chunk in zip(
        mixture_chunks, drums_chunks, bass_chunks, vocals_chunks, other_chunks
    ):
        chunk_dict = {
            "mixture": mixture_chunk,
            "drums": drums_chunk,
            "bass": bass_chunk,
            "vocals": vocals_chunk,
            "other": other_chunk,
        }
        chunk_list.append(chunk_dict)

    return chunk_list


def process_dataset(root_dir: str, processed_dir: str) -> None:
    """
    Itera sobre todas las carpetas de canciones en el dataset de entrenamiento,
    aplica el preprocesamiento a cada una y almacena los chunks extraídos en un único
    archivo NPZ.

    Cada archivo NPZ contendrá múltiples arrays, uno por cada chunk, con claves del tipo:
        "chunk_000", "chunk_001", ...

    Args:
        root_dir (str): Ruta al directorio raíz con las carpetas de canciones.
        processed_dir (str): Ruta al directorio donde se almacenarán los archivos NPZ.
    """

    song_folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    else:
        logging.warning(f"El directorio {processed_dir} ya existe. Se sobreescribirá.")

        for song_folder in tqdm(song_folders, desc="Processing songs"):
            song_name = os.path.basename(song_folder)
            chunks = process_song(song_folder)
            if chunks is None or len(chunks) == 0:
                logging.warning(
                    f"No se generaron chunks para la canción {song_name}. Se omite."
                )
                continue

            chunks_dict = {}
            for i, chunk in enumerate(chunks):
                key = f"chunk_{i:03d}"
                chunks_dict[key] = chunk

            npz_path = os.path.join(processed_dir, f"{song_name}.npz")
            np.savez(npz_path, **chunks_dict)
            logging.info(
                f"Se guardaron {len(chunks)} chunks para {song_name} en {npz_path}"
            )


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.handlers = []
    handler = TqdmLoggingHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    process_dataset("data/external/train", "data/processed/train")
