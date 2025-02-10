import logging
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_dict(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Converts a list of dictionaries into a dictionary of stacked tensors.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of samples.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with each key associated to a tensor
                                 of shape [batch_size, ...].
    """
    collated = {}
    # Assumes all samples share the same keys
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated


class Trainer:
    """
    Trainer class that integrates the PyTorch DataLoader, processes the data,
    and runs the training loop. It uses tqdm to display progress during each epoch
    and logging to record epoch information.

    Assumptions:
      - The model receives an input tensor named "mixture" and returns a tensor of shape
        [batch, out_channels, H, W].
      - The target dictionary contains, for each sample, the tensors corresponding
        to each source (e.g., "vocal", "drum", "bass", "other").
      - The `target_keys` parameter defines the order of output channels.
      - The loss function is applied per channel and the losses are averaged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        target_keys: List[str],
    ) -> None:
        """
        Initializes the Trainer and moves the model to the available device.

        Args:
            model (torch.nn.Module): The PyTorch model.
            optimizer (torch.optim.Optimizer): The optimizer.
            loss_fn (Callable): The loss function that receives (output, target).
            target_keys (List[str]): List of keys in the order that the output channels
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.target_keys = target_keys

        # Set up logging (adjust logging configuration as needed)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Simple console handler if no handler is configured yet
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def train_epoch(self) -> float:
        """
        Runs one training epoch over the dataset.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Wrap the dataloader with tqdm for progress display
        for batch in tqdm(self.dataloader, desc="Training batches", leave=False):
            # Move each tensor in the batch to the correct device
            batch = {key: value.to(self.device) for key, value in batch.items()}

            # Input is assumed to be under the key "mixture"
            inputs = batch["mixture"]
            # Targets are all keys except "mixture"
            targets = {k: v for k, v in batch.items() if k != "mixture"}

            self.optimizer.zero_grad()
            # The model is expected to return a tensor
            # with shape [batch, out_channels, H, W]
            outputs = self.model(inputs)

            # Ensure the output channels match the number of target keys
            if outputs.size(1) != len(self.target_keys):
                raise ValueError(
                    f"Output channels ({outputs.size(1)}) do not match "
                    f"the number of target keys ({len(self.target_keys)})."
                )

            loss = 0.0
            # Calculate loss for each channel based on the ordering in target_keys
            for i, key in enumerate(self.target_keys):
                # Extract channel i from the output tensor
                loss += self.loss_fn(outputs[:, i, :, :], targets[key])
            loss = loss / len(self.target_keys)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        epochs: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> None:
        """
        Trains the model for a given number of epochs.

        Args:
            dataset (torch.utils.data.Dataset): The training dataset.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size.
            num_workers (int): Number of processes for loading data.
            shuffle (bool): Whether to shuffle the data each epoch.
        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_dict,
            shuffle=shuffle,
        )

        self.logger.info(f"Starting training for {epochs} epochs.")
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    def validate(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """
        Validates the model on a given dataset.

        Args:
            dataset (torch.utils.data.Dataset): The validation dataset.
            batch_size (int): Batch size.
            num_workers (int): Number of processes for loading data.
        """
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_dict,
            shuffle=False,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation batches", leave=False):
                batch = {key: value.to(self.device) for key, value in batch.items()}

                inputs = batch["mixture"]
                targets = {k: v for k, v in batch.items() if k != "mixture"}

                outputs = self.model(inputs)

                if outputs.size(1) != len(self.target_keys):
                    raise ValueError(
                        f"Output channels ({outputs.size(1)}) do not match "
                        f"the number of target keys ({len(self.target_keys)})."
                    )

                loss = 0.0
                for i, key in enumerate(self.target_keys):
                    loss += self.loss_fn(outputs[:, i, :, :], targets[key])
                loss = loss / len(self.target_keys)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Loss: {avg_loss:.4f}")
