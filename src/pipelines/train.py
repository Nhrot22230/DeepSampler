import time
from typing import Optional

import torch
from tqdm import tqdm


def train_pipeline(
    model: torch.nn.Module,
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    history = {"epoch_loss": [], "learning_rate": [], "batch_losses": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        batch_losses = []

        batch_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, (inputs, targets) in enumerate(batch_iter, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            try:
                loss = criterion(outputs, targets)
                loss.backward()
            except Exception as e:
                print(f"Error: {e}")
                print(f"Batch index: {batch_idx}")
                print(f"Inputs shape: {inputs.shape}")
                print(f"Targets shape: {targets.shape}")
                print(f"Outputs shape: {outputs.shape}")
                raise e

            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            avg_loss = epoch_loss / batch_idx

            batch_iter.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "batch_loss": f"{batch_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                }
            )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        remaining_time = (time.time() - start_time) / epoch * (epochs - epoch)

        tqdm.write(
            f"Epoch {epoch}/{epochs} - "
            f"Avg Loss: {epoch_loss / len(dataloader):.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} - "
            f"Time: {epoch_time:.2f}s - "
            f"ETA: {remaining_time:.2f}s"
        )

        history["epoch_loss"].append(epoch_loss / len(dataloader))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["batch_losses"].extend(batch_losses)

    print("\nTraining complete. Total time:", time.time() - start_time, "seconds.")
    return history
