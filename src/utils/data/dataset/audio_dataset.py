from typing import Any, List, Optional

from src.utils.data.transforms import TransformChain
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self, data: List[Any], transform: Optional[TransformChain] = None
    ) -> None:
        """
        Dataset genérico con tipado fuerte y estructura modular.

        Args:
            data (List[Any]): Lista de muestras o rutas a archivos.
            transform (Optional[SampleTransformer]): Objeto que aplica
            transformaciones a cada muestra.
        """
        super().__init__()
        self.transform: Optional[TransformChain] = transform
        self.samples: List[Any] = data

    def __len__(self) -> int:
        """
        Retorna el número total de muestras en el dataset.

        Returns:
            int: Número de muestras.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        """
        Retorna la muestra correspondiente al índice `idx`, aplicando la
        transformación en caso de que esté definida.

        Args:
            idx (int): Índice de la muestra a retornar.

        Returns:
            Any: Muestra (transformada si se especificó un objeto
            transformador).

        Raises:
            IndexError: Si el índice está fuera del rango válido.
        """
        sample_len: int = len(self.samples)

        if idx < 0 or idx >= sample_len:
            raise IndexError(
                f"El índice {idx} está fuera del rango (0, {sample_len-1})."
            )

        sample: Any = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
