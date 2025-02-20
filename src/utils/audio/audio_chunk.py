from dataclasses import dataclass
from typing import List, TypedDict

import torch


@dataclass
class AudioChunk(TypedDict):
    """
    Data class representing an audio chunk with multiple channels.

    Attributes:
        mixture (torch.Tensor): The mixture tensor.
        vocals (torch.Tensor): The vocals tensor.
        drums (torch.Tensor): The drums tensor.
        bass (torch.Tensor): The bass tensor.
        other (torch.Tensor): The tensor for other instruments.
    """

    mixture: torch.Tensor
    vocals: torch.Tensor
    drums: torch.Tensor
    bass: torch.Tensor
    other: torch.Tensor

    @staticmethod
    def from_file(pt_file: str) -> "AudioChunk":
        """
        Load an AudioChunk from a .pt file.

        Args:
            pt_file (str): The path to the .pt file.

        Returns:
            AudioChunk: The loaded AudioChunk instance.
        """
        loaded = torch.load(pt_file)
        return AudioChunk(
            mixture=loaded["mixture"],
            vocals=loaded["vocals"],
            drums=loaded["drums"],
            bass=loaded["bass"],
            other=loaded["other"],
        )

    @staticmethod
    def isolate_channels(chunk: "AudioChunk", channels: List[str]) -> "AudioChunk":
        """
        Create a new AudioChunk with only the specified channels.

        For channels not specified, the corresponding tensor is replaced by a zero tensor
        having the same shape as the mixture.

        Args:
            channels (List[str]): List of channels to include. Valid options include
                                  "vocals", "drums", "bass", and "other".

        Returns:
            AudioChunk: A new AudioChunk containing only the specified channels.
        """
        new_vocals = (
            chunk["vocals"]
            if "vocals" in channels
            else torch.zeros_like(chunk["mixture"])
        )
        new_drums = (
            chunk["drums"]
            if "drums" in channels
            else torch.zeros_like(chunk["mixture"])
        )
        new_bass = (
            chunk["bass"] if "bass" in channels else torch.zeros_like(chunk["mixture"])
        )
        new_other = (
            chunk["other"]
            if "other" in channels
            else torch.zeros_like(chunk["mixture"])
        )
        new_mixture = new_vocals + new_drums + new_bass + new_other

        return AudioChunk(
            mixture=new_mixture,
            vocals=new_vocals,
            drums=new_drums,
            bass=new_bass,
            other=new_other,
        )
