from typing import TypedDict

import torch


class AudioChunk(TypedDict):
    mixture: torch.Tensor
    vocals: torch.Tensor
    drums: torch.Tensor
    bass: torch.Tensor
    other: torch.Tensor

    @staticmethod
    def from_file(pt_file: str):
        """
        Carga un AudioChunk a partir de un archivo .pt.
        Args:
            pt_file (str): Ruta al archivo .pt.
        Returns:
            AudioChunk: AudioChunk cargado.
        """

        loaded = torch.load(pt_file)
        return AudioChunk(
            mixture=loaded["mixture"],
            vocals=loaded["vocals"],
            drums=loaded["drums"],
            bass=loaded["bass"],
            other=loaded["other"],
        )
