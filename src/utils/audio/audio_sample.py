from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as T


class AudioSample:
    """
    Representa la unidad mínima de entrenamiento en un proyecto de separación de audio.
    Cada instancia contiene un chunk de audio descompuesto en sus diferentes stems:
    "mixture", "drums", "bass", "vocals" y "other".
    """

    def __init__(self, data: Dict[str, np.ndarray]) -> None:
        """
        Inicializa una instancia de AudioSample.

        Args:
            data (Dict[str, np.ndarray]): Diccionario con claves "mixture", "drums",
            "bass", "vocals" y "other".
        """
        self.mixture = data["mixture"]
        self.drums = data["drums"]
        self.bass = data["bass"]
        self.vocals = data["vocals"]
        self.other = data["other"]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Convierte la instancia en un diccionario.

        Returns:
            Dict[str, np.ndarray]: Diccionario con claves "mixture", "drums", "bass",
            "vocals" y "other".
        """
        return {
            "mixture": self.mixture,
            "drums": self.drums,
            "bass": self.bass,
            "vocals": self.vocals,
            "other": self.other,
        }

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """
        Convierte la instancia en un diccionario de tensores.

        Returns:
            Dict[str, torch.Tensor]: Diccionario con claves "mixture", "drums", "bass",
            "vocals" y "other".
        """
        return {
            "mixture": torch.from_numpy(self.mixture).float(),
            "drums": torch.from_numpy(self.drums).float(),
            "bass": torch.from_numpy(self.bass).float(),
            "vocals": torch.from_numpy(self.vocals).float(),
            "other": torch.from_numpy(self.other).float(),
        }

    def plot_waveform(self, stem: str = "mixture") -> None:
        """
        Grafica la forma de onda del stem indicado.

        Args:
            stem (str): Nombre del stem a graficar (por defecto "mixture").
        """
        if not hasattr(self, stem):
            raise ValueError(f"Stem '{stem}' no existe en la muestra.")
        data = getattr(self, stem)
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(f"Waveform - {stem.capitalize()}")
        plt.xlabel("Tiempo")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        cmap: str = "magma",
    ) -> None:
        """
        Genera y grafica un espectrograma en dB a partir de un chunk de audio.

        Args:
            data (np.ndarray): Chunk de audio (vector 1D).
            title (str): Título del gráfico.
            n_fft (int): Tamaño de la FFT (por defecto 2048).
            hop_length (int): Hop length para la STFT (por defecto 512).
            cmap (str): Mapa de colores para la visualización (por defecto "magma").
        """
        waveform = torch.from_numpy(self.mixture).float()
        spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        db_transform = T.AmplitudeToDB(top_db=80)
        spectrogram = spec_transform(waveform)
        spectrogram_db = db_transform(spectrogram)
        spectrogram_db = spectrogram_db.numpy()

        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db, aspect="auto", origin="lower", cmap=cmap)
        plt.xlabel("Frames")
        plt.ylabel("Bins de Frecuencia")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
