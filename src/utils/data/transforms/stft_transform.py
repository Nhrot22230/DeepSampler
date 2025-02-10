from typing import Dict

import torch
import torchaudio


class STFTTransform:

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        power: float = 2.0,
        return_db: bool = False,
        top_db: int = 80,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.return_db = return_db

        # Crear la transformación STFT
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,  # Si power es None, se preservan los valores complejos.
        )
        # Si se solicita, se configura la transformación para convertir a decibelios.
        if return_db and (power is not None):
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        else:
            self.db_transform = None

    def __call__(self, sample_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed = {}
        for key, waveform in sample_dict.items():
            # Verifica si el tensor es 1D; en ese caso se agrega la dimensión de canal.
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            # Aplica la STFT al tensor (se espera forma [channel, time]).
            spectrogram = self.stft(waveform)
            # Si se ha solicitado la conversión a dB, se aplica la transformación.
            if self.db_transform is not None:
                spectrogram = self.db_transform(spectrogram)
            transformed[key] = spectrogram
        return transformed
