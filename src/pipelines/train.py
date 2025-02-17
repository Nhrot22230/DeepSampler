import time
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function module
        optimizer: Optimization algorithm
        device: Computation device (CPU/GPU)
        scheduler: Learning rate scheduler

    Returns:
        Tuple containing average epoch loss and current learning rate
    """
    model.train()
    total_loss = 0.0
    current_lr = optimizer.param_groups[0]["lr"]

    with tqdm(dataloader, unit="batch", leave=False) as batch_iter:
        for inputs, targets in batch_iter:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss calculation with shape validation
            try:
                loss = criterion(outputs, targets)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Loss calculation failed: {e}\n"
                    f"Shapes - Input: {inputs.shape},"
                    f"Target: {targets.shape}, Output: {outputs.shape}"
                ) from e

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update tracking
            total_loss += loss.item()
            batch_iter.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.1e}"}
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss, current_lr


def train_pipeline(
    model: torch.nn.Module,
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, list]:
    """Main training pipeline.

    Args:
        model: The neural network model
        epochs: Number of training epochs
        dataloader: Training data loader
        criterion: Loss function module
        optimizer: Optimization algorithm
        device: Computation device (CPU/GPU)
        scheduler: Learning rate scheduler

    Returns:
        Training history dictionary
    """
    history = {"epoch_loss": [], "learning_rate": [], "training_time": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train for one epoch
        avg_loss, current_lr = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Update scheduler if provided
        if scheduler:
            scheduler.step()

        # Record metrics
        epoch_time = time.time() - epoch_start
        history["epoch_loss"].append(avg_loss)
        history["learning_rate"].append(current_lr)
        history["training_time"].append(epoch_time)

        # Print epoch summary
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    return history
