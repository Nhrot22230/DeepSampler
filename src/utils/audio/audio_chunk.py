from typing import TypedDict

import torch


class AudioChunk(TypedDict):
    mixture: torch.Tensor
    bass: torch.Tensor
    drums: torch.Tensor
    other: torch.Tensor
    vocals: torch.Tensor
