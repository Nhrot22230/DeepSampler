# =============================================================================
# PREPROCESAMIENTO: Generar y almacenar chunks en archivos .npy
# =============================================================================

import os
import numpy as np
import librosa
import logging
from tqdm import tqdm

SR = 44100  # Frecuencia de muestreo
N_FFT = 2048  # Tamaño de la FFT
HOP_LENGTH = 512  # Hop length para la STFT
CHUNK_DURATION = 2.0  # Duración de cada chunk en segundos
OVERLAP_FRACTION = 0.5  # Fracción de solapamiento entre chunks


def load_and_normalize(file_path: str, sr: int = SR) -> np.ndarray:
    """
    Carga un archivo de audio y normaliza la señal para que su pico
    máximo sea 1.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def compute_log_spectrogram(
    y: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Calcula la STFT, extrae la magnitud y aplica una escala logarítmica.
    """
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S_complex)
    S_log = np.log1p(S_mag)
    return S_log


def chunk_spectrogram(spec: np.ndarray, chunk_frames: int, overlap_frames: int) -> list:
    """
    Divide el espectrograma en chunks a lo largo del eje temporal.

    Args:
        spec (np.ndarray): Espectrograma con forma (n_freq, n_time)
        chunk_frames (int): Número de frames por chunk.
        overlap_frames (int): Número de frames solapados entre chunks.

    Returns:
        List[np.ndarray]: Lista de chunks.
    """
    chunks = []
    n_time = spec.shape[1]
    step = chunk_frames - overlap_frames
    for start in range(0, n_time - chunk_frames + 1, step):
        chunk = spec[:, start : start + chunk_frames]
        chunks.append(chunk)
    return chunks


def process_song(song_folder: str, save_dir: str) -> None:
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
        mixture_audio = load_and_normalize(mixture_path)
        drums_audio = load_and_normalize(drums_path)
        bass_audio = load_and_normalize(bass_path)
        vocals_audio = load_and_normalize(vocals_path)
        other_audio = load_and_normalize(other_path)
    except ValueError as e:
        logging.error(e)
        return

    mixture_spec = compute_log_spectrogram(mixture_audio)
    drums_spec = compute_log_spectrogram(drums_audio)
    bass_spec = compute_log_spectrogram(bass_audio)
    vocals_spec = compute_log_spectrogram(vocals_audio)
    other_spec = compute_log_spectrogram(other_audio)

    n_time = mixture_spec.shape[1]
    if not (
        drums_spec.shape[1] == n_time
        and bass_spec.shape[1] == n_time
        and vocals_spec.shape[1] == n_time
        and other_spec.shape[1] == n_time
    ):
        logging.error("Dimensiones temporales inconsistentes en %s.", song_folder)
        return

    chunk_frames = int(np.ceil(CHUNK_DURATION * SR / HOP_LENGTH))
    overlap_frames = int(chunk_frames * OVERLAP_FRACTION)

    mixture_chunks = chunk_spectrogram(mixture_spec, chunk_frames, overlap_frames)
    drums_chunks = chunk_spectrogram(drums_spec, chunk_frames, overlap_frames)
    bass_chunks = chunk_spectrogram(bass_spec, chunk_frames, overlap_frames)
    vocals_chunks = chunk_spectrogram(vocals_spec, chunk_frames, overlap_frames)
    other_chunks = chunk_spectrogram(other_spec, chunk_frames, overlap_frames)

    song_name = os.path.basename(os.path.normpath(song_folder))
    os.makedirs(save_dir, exist_ok=True)
    num_chunks = min(
        len(mixture_chunks),
        len(drums_chunks),
        len(bass_chunks),
        len(vocals_chunks),
        len(other_chunks),
    )

    for i in range(num_chunks):
        data_dict = {
            "mixture": mixture_chunks[i],
            "drums": drums_chunks[i],
            "bass": bass_chunks[i],
            "vocals": vocals_chunks[i],
            "other": other_chunks[i],
        }
        save_path = os.path.join(save_dir, f"{song_name}_chunk_{i}.npy")
        np.save(save_path, data_dict)

    logging.info("Procesada canción: %s | Chunks generados: %d", song_name, num_chunks)


def process_dataset(root_dir: str, processed_dir: str) -> None:
    """
    Itera sobre todas las carpetas de canciones en el dataset de entrenamiento
    y aplica el preprocesamiento a cada una.
    """
    song_folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for song_folder in tqdm(song_folders, desc="Procesando canciones"):
        process_song(song_folder, processed_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_dataset("data/external/train", "data/processed/train")
