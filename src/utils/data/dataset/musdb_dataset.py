from typing import List, Union
import torch
from torch.utils.data import Dataset
from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import log_spectrogram


class MUSDBDataset(Dataset):
    """
    Dataset class for MUSDB18.

    Args:
        data (List[Union[AudioChunk, str]]): List of AudioChunks or file paths to .pt files.
        window (torch.Tensor): Window function to use for the STFT.
        n_fft (int, optional): Size of the FFT window for the STFT. Defaults to 2048.
        hop_length (int, optional): Hop length for the STFT. Defaults to 512.
    """

    def __init__(
        self,
        data: List[Union[AudioChunk, str]],
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.data: List[Union[AudioChunk, str]] = data
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        audio_chunk: AudioChunk = (
            AudioChunk.from_file(item) if isinstance(item, str) else item
        )

        target_keys = ["mixture", "vocals", "drums", "bass", "other"]

        for key in target_keys:
            audio_chunk[key] = log_spectrogram(
                audio_chunk[key],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

        # Ensure that the mixture spectrogram has a channel dimension (unsqueezed).
        input_spec = torch.stack([audio_chunk["mixture"]])
        # Stack the target spectrograms for vocals, drums, bass, and other.
        target_spec = torch.stack(
            [audio_chunk[key] for key in target_keys if key != "mixture"]
        )

        return input_spec, target_spec
