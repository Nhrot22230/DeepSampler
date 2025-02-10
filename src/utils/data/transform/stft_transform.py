from typing import Dict
import torch
import torchaudio


class STFTTransform:
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,  # Conserva los valores complejos
        )

    def __call__(self, sample_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed = {}
        for key, waveform in sample_dict.items():
            spectrogram = self.stft(waveform.unsqueeze(0)).squeeze(0)
            transformed[key] = spectrogram
        return transformed
