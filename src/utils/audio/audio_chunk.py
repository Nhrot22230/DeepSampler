import torch
from typing import TypedDict


class AudioChunk(TypedDict):
    mixture: torch.Tensor
    bass: torch.Tensor
    drums: torch.Tensor
    other: torch.Tensor
    vocals: torch.Tensor
