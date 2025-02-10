from typing import Any, Callable, List, Optional


class TransformChain:
    """
    Clase para transformar una muestra. Permite encadenar
    múltiples transformaciones.
    """

    def __init__(self, transforms: Optional[List[Callable[[Any], Any]]] = None) -> None:
        """
        Args:
            transforms (Optional[List[Callable[[Any], Any]]]):
            Lista de funciones de transformación.

        Cada función debe aceptar una muestra y retornar la muestra
        transformada.
        """
        self.transforms: List[Callable[[Any], Any]] = (
            transforms if transforms is not None else []
        )

    def add_transform(self, transform: Callable[[Any], Any]) -> None:
        """
        Agrega una transformación a la lista.

        Args:
            transform (Callable[[Any], Any]): Función de
            transformación que se aplicará a la muestra.
        """
        self.transforms.append(transform)

    def __call__(self, sample: Any) -> Any:
        """
        Aplica secuencialmente todas las transformaciones a la muestra.

        Args:
            sample (Any): La muestra a transformar.

        Returns:
            Any: La muestra transformada tras aplicar
            todas las transformaciones.
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample
