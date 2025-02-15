import os
import time
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


def train_pipeline(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    total_epochs: int = 40,
    phase1_epochs: int = 20,
):
    history = {"epoch_loss": [], "learning_rate": [], "batch_losses": []}
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        batch_losses = []

        if epoch == phase1_epochs + 1:
            print("\nStarting second training phase (lr=1e-4)")

        batch_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, (inputs, targets) in enumerate(batch_iter, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            avg_loss = epoch_loss / batch_idx

            # Calculate moving average of loss
            window_size = min(50, len(batch_losses))
            moving_avg_loss = np.mean(batch_losses[-window_size:])

            batch_iter.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "batch_loss": f"{batch_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "mov_avg_loss": f"{moving_avg_loss:.4f}",
                    "grad_norm": f"{grad_norm:.2f}",
                }
            )

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        remaining_time = (time.time() - start_time) / epoch * (total_epochs - epoch)

        print(
            f"Epoch {epoch}/{total_epochs} - "
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


if __name__ == "__main__":
    from src.models import SCUNet
    from src.utils.data import MUSDB18Dataset
    from src.utils.training import MultiSourceL1Loss
    from torch.utils.data import DataLoader

    project_root = os.getcwd()
    while "src" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)
    data_root = os.path.join(project_root, "data")
    musdb_path = os.path.join(project_root, "data", "musdb18hq", "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MUSDB18Dataset(os.path.join(data_root, "processed", "train"))
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    mixture, _ = train_dataset.__getitem__(0)
    print("Sample input shape:", mixture.shape)

    model = SCUNet()
    model.to(device)
    criterion = MultiSourceL1Loss(weights=[0.297, 0.262, 0.232, 0.209])  # Sum to 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    total_epochs = 40
    phase1_epochs = 20

    model = train_pipeline(
        model=model,
        dataloader=train_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        total_epochs=total_epochs,
        phase1_epochs=phase1_epochs,
    )

    torch.save(
        model.state_dict(),
        os.path.join(project_root, "experiments", "checkpoints", "scunet.pth"),
    )
    print("Model checkpoint saved.")
