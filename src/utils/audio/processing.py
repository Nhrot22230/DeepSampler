import librosa
import numpy as np

SR = 44100  # Frecuencia de muestreo
N_FFT = 2048  # Tamaño de la FFT
HOP_LENGTH = 512  # Hop length para la STFT
CHUNK_DURATION = 5.0  # Duración de cada chunk en segundos
OVERLAP_FRACTION = 0.2  # Fracción de solapamiento entre chunks


def normalize_audio(sample: np.ndarray) -> np.ndarray:
    """
    Normaliza una señal de audio para que su pico máximo sea 1.

    Args:
        sample (np.ndarray): Señal de audio 1D.

    Returns:
        np.ndarray: Señal normalizada.
    """
    max_val = np.max(np.abs(sample))
    if max_val > 0:
        return sample / max_val
    return sample


def load_audio(file_path: str, sr: int = SR) -> np.ndarray:
    """
    Carga un archivo de audio y normaliza la señal para que su pico máximo sea 1.

    Args:
        file_path (str): Ruta al archivo de audio.
        sr (int): Frecuencia de muestreo deseada.

    Returns:
        np.ndarray: Señal de audio normalizada.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y = normalize_audio(y)
        return y
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def chunk_audio(
    y: np.ndarray,
    chunk_duration: float = CHUNK_DURATION,
    overlap_fraction: float = OVERLAP_FRACTION,
    sr: int = SR,
) -> np.ndarray:
    """
    Divide una señal de audio en chunks de duración fija.
    Si la señal es más corta que un chunk, se rellena con ceros.

    Args:
        y (np.ndarray): Señal de audio 1D.
        chunk_duration (float): Duración de cada chunk en segundos.
        overlap_fraction (float): Fracción de solapamiento entre chunks.
        sr (int): Frecuencia de muestreo.

    Returns:
        np.ndarray: Array 2D de chunks con shape (n_chunks, chunk_samples).
    """
    chunk_samples = int(chunk_duration * sr)

    # Si la señal es más corta que el tamaño de un chunk, la rellenamos
    if len(y) < chunk_samples:
        y = np.pad(y, (0, chunk_samples - len(y)), mode="constant")
        return np.array([y])

    overlap_samples = int(chunk_samples * overlap_fraction)
    hop_samples = chunk_samples - overlap_samples
    n_chunks = 1 + (len(y) - chunk_samples) // hop_samples

    chunks = []
    for i in range(n_chunks):
        start = i * hop_samples
        end = start + chunk_samples
        chunk = y[start:end]
        # En caso de que el último chunk sea más corto, se rellena con ceros.
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
        chunks.append(chunk)
    return np.array(chunks)


def apply_stft(
    sample: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Aplica la STFT a un chunk de audio y retorna el espectrograma logarítmico en dB.

    Args:
        sample (np.ndarray): Chunk de audio (1D array).
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Hop length para la STFT.

    Returns:
        np.ndarray: Espectrograma logarítmico en dB.
    """
    stft_matrix = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    return log_spectrogram


def split_frequency_bands(spectrogram: np.ndarray, sr: int = SR) -> dict:
    """
    Divide el espectrograma en bandas de frecuencia: baja, media y alta.

    Args:
        spectrogram (np.ndarray): Espectrograma con shape (n_bins, time_frames).
        sr (int): Frecuencia de muestreo.

    Returns:
        dict: Diccionario con las claves "low", "mid" y "high" que contienen cada banda.
    """
    num_bins = spectrogram.shape[0]
    # Cada bin corresponde a una cantidad de Hz
    freq_per_bin = (sr / 2) / (num_bins - 1)

    low_cutoff_hz = 200.0
    mid_cutoff_hz = 2000.0

    cutoff_low = int(low_cutoff_hz / freq_per_bin)
    cutoff_mid = int(mid_cutoff_hz / freq_per_bin)

    low_band = spectrogram[:cutoff_low, :]
    mid_band = spectrogram[cutoff_low:cutoff_mid, :]
    high_band = spectrogram[cutoff_mid:, :]

    return {"low": low_band, "mid": mid_band, "high": high_band}


def merge_frequency_bands(bands: dict) -> np.ndarray:
    """
    Une las bandas de frecuencia en un solo espectrograma.
    Se utiliza una concatenación vertical (a lo largo del eje de frecuencias).

    Args:
        bands (dict): Diccionario con las claves "low", "mid" y "high" que contienen
        cada banda.

    Returns:
        np.ndarray: Espectrograma concatenado.
    """
    merged = np.vstack((bands["low"], bands["mid"], bands["high"]))
    return merged


def apply_stft_inverse(
    spectrogram: np.ndarray, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Aplica la transformada inversa de la STFT a un espectrograma y retorna el audio
    reconstruido.
    Se utiliza una fase aleatoria, por lo que la reconstrucción puede no ser exacta.

    Args:
        spectrogram (np.ndarray): Espectrograma logarítmico en dB.
        hop_length (int): Hop length utilizado en la STFT.

    Returns:
        np.ndarray: Audio reconstruido.
    """
    magnitude = librosa.db_to_amplitude(spectrogram)
    random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=magnitude.shape))
    stft_matrix = magnitude * random_phase
    audio = librosa.istft(stft_matrix, hop_length=hop_length)
    return audio


def apply_mel_spectrogram(
    sample: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Calcula el espectrograma mel de un chunk de audio y lo convierte a dB.

    Args:
        sample (np.ndarray): Chunk de audio (1D array).
        sr (int): Frecuencia de muestreo.
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Hop length para la STFT.
        n_mels (int): Número de bandas mel.

    Returns:
        np.ndarray: Espectrograma mel en dB.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def time_stretch(sample: np.ndarray, rate: float) -> np.ndarray:
    """
    Aplica time stretching a un chunk de audio.

    Args:
        sample (np.ndarray): Chunk de audio (1D array).
        rate (float): Factor de estiramiento. rate > 1 aumenta la duración.

    Returns:
        np.ndarray: Chunk de audio estirado en el tiempo.
    """
    return librosa.effects.time_stretch(sample, rate)


def pitch_shift(sample: np.ndarray, sr: int = SR, n_steps: float = 4.0) -> np.ndarray:
    """
    Aplica pitch shifting a un chunk de audio.

    Args:
        sample (np.ndarray): Chunk de audio (1D array).
        sr (int): Frecuencia de muestreo.
        n_steps (float): Número de semitonos a desplazar.

    Returns:
        np.ndarray: Chunk de audio con el pitch modificado.
    """
    return librosa.effects.pitch_shift(sample, sr, n_steps)


def add_white_noise(sample: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Agrega ruido blanco a un chunk de audio.

    Args:
        sample (np.ndarray): Chunk de audio (1D array).
        noise_factor (float): Factor de escala para el ruido.

    Returns:
        np.ndarray: Chunk de audio con ruido agregado, clippeado en el rango [-1, 1].
    """
    noise = np.random.randn(len(sample))
    augmented = sample + noise_factor * noise
    return np.clip(augmented, -1.0, 1.0)
