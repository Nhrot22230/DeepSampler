import torch
from typing import Dict

import numpy as np


class AudioSample:
    """
    Representa la unidad mínima de entrenamiento en un proyecto de separación de audio.
    Cada instancia contiene un chunk del audio descompuesto en sus diferentes stems:
    "mixture", "drums", "bass", "vocals" y "other".
    """

    def __init__(
        self,
        mixture: np.ndarray,
        drums: np.ndarray,
        bass: np.ndarray,
        vocals: np.ndarray,
        other: np.ndarray,
    ) -> None:
        """
        Inicializa una instancia de AudioSample.

        Args:
            mixture (np.ndarray): Espectrograma o señal de la mezcla.
            drums (np.ndarray): Espectrograma o señal de la batería.
            bass (np.ndarray): Espectrograma o señal del bajo.
            vocals (np.ndarray): Espectrograma o señal de las voces.
            other (np.ndarray): Espectrograma o señal de los demás componentes.
        """
        self.mixture = mixture
        self.drums = drums
        self.bass = bass
        self.vocals = vocals
        self.other = other

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
            "mixture": torch.from_numpy(self.mixture),
            "drums": torch.from_numpy(self.drums),
            "bass": torch.from_numpy(self.bass),
            "vocals": torch.from_numpy(self.vocals),
            "other": torch.from_numpy(self.other),
        }
