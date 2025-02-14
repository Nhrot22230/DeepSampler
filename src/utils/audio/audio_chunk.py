from typing import TypedDict

import torch


class AudioChunk(TypedDict):
    mixture: torch.Tensor
    vocals: torch.Tensor
    drums: torch.Tensor
    bass: torch.Tensor
    other: torch.Tensor
