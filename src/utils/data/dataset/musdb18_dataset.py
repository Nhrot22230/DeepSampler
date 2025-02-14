from typing import List

import torch
from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import log_spectrogram
from torch.utils.data import Dataset


# Definición de la clase Dataset para MUSDB18
class MUSDB18Dataset(Dataset):
    def __init__(
        self,
        data: List[AudioChunk],
        window: torch.Tensor,
        nfft: int = 2048,
        hop_length: int = 512,
    ):
        """
        Args:
            data (List[AudioChunk]): Lista de AudioChunks.
            nfft (int, optional): Tamaño de la ventana de la STFT. Defaults to 2048.
            hop_length (int, optional): Tamaño del salto de la STFT. Defaults to 512.
            window (torch.Tensor, optional): Ventana a utilizar en la STFT.
        """
        self.data = data
        self.nfft = nfft
        self.hop_length = hop_length
        self.window = window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        mixture = self.data[index]["mixture"]
        vocals = self.data[index]["vocals"]
        drums = self.data[index]["drums"]
        bass = self.data[index]["bass"]
        other = self.data[index]["other"]

        mixture_spec = log_spectrogram(mixture, self.nfft, self.hop_length, self.window)
        vocals_spec = log_spectrogram(vocals, self.nfft, self.hop_length, self.window)
        drums_spec = log_spectrogram(drums, self.nfft, self.hop_length, self.window)
        bass_spec = log_spectrogram(bass, self.nfft, self.hop_length, self.window)
        other_spec = log_spectrogram(other, self.nfft, self.hop_length, self.window)

        targets = torch.stack([vocals_spec, drums_spec, bass_spec, other_spec])

        return mixture_spec, targets.squeeze(1)
