import os
import sys

import torch
from src.models import SCUNet
from src.utils.data import MUSDB18Dataset
from src.utils.training import MultiSourceL1Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.getcwd()
while "src" not in os.listdir(project_root):
    project_root = os.path.dirname(project_root)
sys.path.append(project_root)

data_root = os.path.join(project_root, "data")
musdb_path = os.path.join(project_root, "data", "musdb18hq", "train")


train_dataset = MUSDB18Dataset(os.path.join(data_root, "processed", "train"))
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)

# get random sample shape
mixture, _ = train_dataset.__getitem__(0)
print(mixture.shape)

model = SCUNet()
criterion = MultiSourceL1Loss(weights=[0.297, 0.262, 0.232, 0.209])  # Sum to 1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

total_epochs = 40
phase1_epochs = 20

for epoch in tqdm(range(total_epochs), desc="Training Progress", unit="epoch"):
    model.train()
    epoch_loss = 0.0

    if epoch == phase1_epochs:
        print("\nStarting second training phase (lr=1e-4)")

    batch_iter = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False, unit="batch"
    )

    for mixture_mag, targets in batch_iter:
        outputs = model(mixture_mag)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_iter.set_postfix(
            {
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss/(batch_iter.n+1):.4f}",
            }
        )

    scheduler.step()
    tqdm.write(
        f"Epoch {epoch+1} completed - "
        f"Avg Loss: {epoch_loss/len(train_loader):.4f} "
        f"LR: {optimizer.param_groups[0]['lr']:.1e}"
    )

torch.save(model.state_dict(), os.path.join(project_root, "checkpoints", "scunet.pth"))
