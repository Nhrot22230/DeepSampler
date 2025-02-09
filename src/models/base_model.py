from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class BaseModel(Module, ABC):
    """
    Clase base abstracta para modelos de Deep Learning en PyTorch.

    Esta clase define la interfaz que debe implementar cualquier modelo derivado,
    asegurando la consistencia en la definición de la propagación hacia adelante,
    pasos de entrenamiento, validación y configuración de optimizadores.
    """

    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Define la propagación hacia adelante del modelo.

        Args:
            x (torch.Tensor): Entrada del modelo.

        Returns:
            torch.Tensor: Salida generada por el modelo.
        """
        raise NotImplementedError(
            "El método forward debe ser implementado en la subclase."
        )

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """
        Define el paso de entrenamiento para un batch de datos.

        Args:
            batch (Any): Batch de datos de entrenamiento.
            batch_idx (int): Índice del batch.

        Returns:
            Dict[str, Any]: Diccionario con la información del paso de entrenamiento
            (por ejemplo, la pérdida).
        """
        raise NotImplementedError(
            "El método training_step debe ser implementado en la subclase."
        )

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """
        Define el paso de validación para un batch de datos.

        Args:
            batch (Any): Batch de datos de validación.
            batch_idx (int): Índice del batch.

        Returns:
            Dict[str, Any]: Diccionario con la información del paso de validación
            (por ejemplo, pérdida y métricas).
        """
        raise NotImplementedError(
            "El método validation_step debe ser implementado en la subclase."
        )

    @abstractmethod
    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        Configura y retorna los optimizadores (y opcionalmente los schedulers) del modelo.

        Returns:
            Union[torch.optim.Optimizer, Dict[str, Any]]:
                El optimizador o un diccionario con la configuración de optimizadores
                y schedulers.
        """
        raise NotImplementedError(
            "El método configure_optimizers debe ser implementado en la subclase."
        )
