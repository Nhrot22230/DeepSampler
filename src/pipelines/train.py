import os
import time
from typing import Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: Optional[float] = 1.0,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Train the model for one epoch with memory-efficient optimizations.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function module
        optimizer: Optimization algorithm
        device: Computation device (CPU/GPU)
        gradient_clip: Maximum gradient norm (None to disable)
        use_amp: Enable automatic mixed precision

    Returns:
        Tuple of (average epoch loss, current learning rate)
    """
    model.train()
    total_loss = 0.0
    current_lr = optimizer.param_groups[0]["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with tqdm(dataloader, unit="batch", leave=False) as batch_iter:
        for inputs, targets in batch_iter:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More memory-efficient

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()

            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_iter.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.1e}",
                device=str(device),
            )

    return total_loss / len(dataloader), current_lr


def train_pipeline(
    model: torch.nn.Module,
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_name: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_dir: str = "checkpoints",
    use_amp: bool = False,
    gradient_clip: Optional[float] = 1.0,
) -> Dict[str, list]:
    """Optimized training pipeline with checkpointing and monitoring.

    Args:
        model: Neural network model
        epochs: Number of training epochs
        dataloader: Training data loader
        criterion: Loss function module
        optimizer: Optimization algorithm
        device: Computation device (CPU/GPU)
        scheduler: Learning rate scheduler
        checkpoint_every: Save checkpoint every N epochs
        checkpoint_dir: Directory to save checkpoints
        use_amp: Enable automatic mixed precision
        gradient_clip: Maximum gradient norm (None to disable)

    Returns:
        Training history dictionary with metrics
    """
    history = {
        "loss": [],
        "learning_rate": [],
    }
    if checkpoint_every is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    start_time = time.time()

    with tqdm(range(1, epochs + 1), unit="epoch") as epoch_iter:
        for epoch in epoch_iter:
            epoch_start = time.time()

            # Train one epoch
            avg_loss, current_lr = train_one_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                gradient_clip=gradient_clip,
            )

            # Update learning rate schedule
            if scheduler:
                scheduler.step()

            # Record metrics
            epoch_time = time.time() - epoch_start
            history["loss"].append(avg_loss)
            history["learning_rate"].append(current_lr)

            # Update progress bar
            epoch_iter.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.1e}",
                time=f"{epoch_time:.1f}s",
            )

            # Save checkpoint
            if checkpoint_every and epoch % checkpoint_every == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{checkpoint_name if checkpoint_name else 'model'}_epoch_{epoch:03d}.pth",
                )
                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )
                tqdm.write(f"Saved checkpoint to {checkpoint_path}")

    tqdm.write(f"\nTraining completed in {time.time()-start_time:.1f} seconds")
    return history
