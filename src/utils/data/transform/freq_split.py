from typing import Dict
import torch


class FreqSplit:
    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 2048,
        low_cutoff: int = 500,
        mid_cutoff: int = 4000,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.bin_freq = sr / n_fft
        self.low_bin = int(low_cutoff / self.bin_freq)
        self.mid_bin = int(mid_cutoff / self.bin_freq)

    def __call__(self, sample_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformed = {}
        for key, spectrogram in sample_dict.items():
            low = spectrogram[: self.low_bin, :]
            mid = spectrogram[self.low_bin : self.mid_bin, :]
            high = spectrogram[self.mid_bin :, :]
            transformed[f"{key}_low"] = low
            transformed[f"{key}_mid"] = mid
            transformed[f"{key}_high"] = high
        return transformed
