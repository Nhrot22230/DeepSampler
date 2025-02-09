from typing import Any, List, Optional
from torch.utils.data import Dataset
from src.data_utils.transform.sample_tranformer import SampleTransformer

class CustomDataset(Dataset):
    def __init__(self, data: List[Any], transform: Optional[SampleTransformer] = None) -> None:
        """
        Dataset genérico con tipado fuerte y estructura modular.

        Args:
            data (List[Any]): Lista de muestras o rutas a archivos.
            transform (Optional[SampleTransformer]): Objeto que aplica transformaciones a cada muestra.
        """
        super().__init__()
        self.transform: Optional[SampleTransformer] = transform
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
        Retorna la muestra correspondiente al índice `idx`, aplicando la transformación
        en caso de que esté definida.

        Args:
            idx (int): Índice de la muestra a retornar.

        Returns:
            Any: Muestra (transformada si se especificó un objeto transformador).

        Raises:
            IndexError: Si el índice está fuera del rango válido.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"El índice {idx} está fuera del rango (0, {len(self.samples)-1}).")
        
        sample: Any = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
