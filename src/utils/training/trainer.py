from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader


def collate_dict(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Convierte una lista de diccionarios en un diccionario de tensores apilados.

    Args:
        batch (List[Dict[str, torch.Tensor]]): Lista de muestras.

    Returns:
        Dict[str, torch.Tensor]: Diccionario con cada clave asociada a un
                                 tensor de dimensión [batch_size, ...].
    """
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated


class Trainer:
    """
    Clase Trainer que integra el DataLoader de PyTorch, permite transformar los datos
    y ejecuta el ciclo de entrenamiento.

    Se asume que el modelo recibe como entrada el tensor "mixture
    y retorna un diccionario con las salidas para cada stem.

    La función de pérdida se aplica a cada par (output, target) y se promedia.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> None:
        """
        Inicializa el Trainer.

        Args:
            model (nn.Module): Modelo de PyTorch.
            optimizer (optim.Optimizer): Optimizador.
            loss_fn (Callable): Función de pérdida que reciba (output, target).
            dataset (data.Dataset): Dataset que retorna diccionarios de tensores.
            device (torch.device): Dispositivo para entrenamiento (CPU o CUDA).
            batch_size (int): Tamaño de batch para el DataLoader.
            num_workers (int): Número de procesos para cargar datos.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_dict,
        )

    def train_epoch(self) -> float:
        """
        Ejecuta una época de entrenamiento sobre el dataset.

        Returns:
            float: Pérdida promedio de la época.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.dataloader:
            batch = {key: value.to(self.device) for key, value in batch.items()}

            inputs = batch["mixture"]
            targets = {k: v for k, v in batch.items() if k != "mixture"}

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = 0.0
            for key in targets.keys():
                loss += self.loss_fn(outputs[key], targets[key])
            loss = loss / len(targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(self, num_epochs: int) -> None:
        """
        Ejecuta el ciclo de entrenamiento durante un número dado de épocas.

        Args:
            num_epochs (int): Número de épocas de entrenamiento.
        """
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")
