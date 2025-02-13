import os
import sys

import torch
from src.models import SCUNet
from src.utils.data import MUSDB18Dataset
from src.utils.training import MultiSourceL1Loss
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_pipeline(
    model,
    dataloader,
    device,
    criterion,
    optimizer,
    scheduler,
    total_epochs=40,
    phase1_epochs=20,
):
    """
    Ejecuta la pipeline de entrenamiento sobre el modelo dado.

    Args:
        model (torch.nn.Module): Modelo a entrenar.
        dataloader (torch.utils.data.DataLoader): Dataloader que provee batches de datos.
        device (torch.device): Dispositivo en el que se ejecuta el entrenamiento.
        criterion (torch.nn.Module): Función de pérdida.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los pesos.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler para ajustar el lr.
        total_epochs (int, optional): Número total de épocas. Default es 40.
        phase1_epochs (int, optional): Épocas de la primera fase antes de cambiar el LR.

    Returns:
        torch.nn.Module: Modelo entrenado.
    """
    for epoch in tqdm(range(total_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        epoch_loss = 0.0

        if epoch == phase1_epochs:
            print("\nStarting second training phase (lr=1e-4)")

        batch_iter = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=False,
            unit="batch",
        )

        for inputs, targets in batch_iter:
            # Mover datos al dispositivo seleccionado
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation y actualización de parámetros
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Acumular pérdida para mostrar métricas
            epoch_loss += loss.item()
            batch_iter.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{epoch_loss/(batch_iter.n+1):.4f}",
                }
            )

        # Actualizar learning rate tras cada época
        scheduler.step()
        tqdm.write(
            f"Epoch {epoch+1} completed"
            f" - Avg Loss: {epoch_loss/len(dataloader):.4f}"
            f" - LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

    return model


if __name__ == "__main__":
    project_root = os.getcwd()
    while "src" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)
    sys.path.append(project_root)

    data_root = os.path.join(project_root, "data")
    musdb_path = os.path.join(project_root, "data", "musdb18hq", "train")

    # 1) Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MUSDB18Dataset(os.path.join(data_root, "processed", "train"))
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    # Get random sample shape just for debug
    mixture, _ = train_dataset.__getitem__(0)
    print("Sample input shape:", mixture.shape)

    model = SCUNet()
    model.to(device)  # 2) Move model to GPU (if available)

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

    # Save the model checkpoint
    torch.save(
        model.state_dict(),
        os.path.join(project_root, "experiments", "checkpoints", "scunet.pth"),
    )
    print("Model checkpoint saved.")
