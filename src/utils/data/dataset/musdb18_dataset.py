import os

import torch
from torch.utils.data import Dataset


class MUSDB18Dataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.track_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]

        self.n_fft = 2048
        self.hop_length = 512
        self.window = torch.hann_window(self.n_fft)

    def __len__(self):
        return len(self.track_files)

    def _compute_spectrogram(self, waveform):
        # waveform shape: (1, samples)
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        stft = torch.abs(stft)  # Magnitude spectrogram
        # log spectrogram
        return torch.log1p(stft)

    def __getitem__(self, idx):
        chunks = torch.load(os.path.join(self.processed_dir, self.track_files[idx]))
        chunk = chunks[0]
        mixture_mag = self._compute_spectrogram(chunk["mixture"]).unsqueeze(
            0
        )  # (1, freq, time)

        sources = ["vocals", "drums", "bass", "other"]
        targets = torch.stack(
            [
                self._compute_spectrogram(chunk[source]).unsqueeze(0)
                for source in sources
            ]
        )  # (4, 1, freq, time)

        return mixture_mag, targets.squeeze(1)  # Shapes: (1, 1025, 169), (4, 1025, 169)
