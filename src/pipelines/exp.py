import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from src.models import DeepSampler
from src.pipelines.data import musdb_pipeline
from src.pipelines.train import train_pipeline
from src.utils.data.dataset import MUSDBDataset
from src.utils.train.losses import MultiSourceLoss
from torch.utils.data import DataLoader


def training_experiment(config):
    """Main training experiment function with configurable parameters"""
    instruments = ["vocals", "drums", "bass", "other"]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Audio parameters
    audio_params = config["audio_params"]
    SR = audio_params["sr"]
    NFFT = audio_params["n_fft"]
    HOP = audio_params["hop_length"]
    CHUNK_DUR = audio_params["chunk_duration"]
    OVERLAP = audio_params["overlap"]

    # Model parameters
    model_params = config["model_params"]
    model = DeepSampler(
        output_channels=model_params["n_sources"],
        base_channels=model_params["base_channels"],
        depth=model_params["depth"],
        dropout=model_params["drop_rate"],
        transformer_heads=model_params["transformer_heads"],
        transformer_layers=model_params["transformer_layers"],
    ).to(device)

    # Dataset parameters
    dataset_params = config["dataset_params"]
    train_params = config["training_params"]

    # Initialize datasets and dataloaders
    isolated_datasets = {}
    for inst in instruments:
        isolated_datasets[inst] = musdb_pipeline(
            musdb_path=Path(config["paths"]["musdb_train"]),
            isolated=[inst],
            sample_rate=SR,
            n_fft=NFFT,
            hop_length=HOP,
            chunk_duration=CHUNK_DUR,
            overlap=OVERLAP,
            max_chunks=dataset_params["isolated_max_samples"],
        )

    # Training pipeline
    try:
        # Phase 1: Isolated training
        if train_params["isolated_epochs"] > 0:
            combined_data = []
            for inst in instruments:
                combined_data.extend(isolated_datasets[inst].data)

            isolated_loader = DataLoader(
                MUSDBDataset(combined_data, NFFT, HOP),
                batch_size=train_params["mixed_batch_size"],
                shuffle=True,
            )

            train_pipeline(
                model=model,
                criterion=MultiSourceLoss(weights=train_params["loss_weights"]),
                optimizer=torch.optim.Adam(
                    model.parameters(), lr=train_params["isolated_lr"]
                ),
                dataloader=isolated_loader,
                epochs=train_params["isolated_epochs"],
                device=device,
            )

        # Phase 2: Mixed training
        train_dataset = musdb_pipeline(
            musdb_path=Path(config["paths"]["musdb_train"]),
            sample_rate=SR,
            n_fft=NFFT,
            hop_length=HOP,
            chunk_duration=CHUNK_DUR,
            overlap=OVERLAP,
            max_chunks=dataset_params["mixed_max_samples"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=train_params["mixed_batch_size"], shuffle=True
        )

        history = train_pipeline(
            model=model,
            criterion=MultiSourceLoss(weights=train_params["loss_weights"]),
            optimizer=torch.optim.AdamW(
                model.parameters(), lr=train_params["mixed_lr"]
            ),
            dataloader=train_loader,
            epochs=train_params["mixed_epochs"],
            checkpoint_dir=Path(config["paths"]["checkpoints"]),
            checkpoint_every=train_params["checkpoint_interval"],
            device=device,
        )

        # Plot training results
        plt.figure(figsize=(10, 5))
        plt.plot(history["loss"], label="Training Loss")
        plt.title("Training Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(Path(config["paths"]["results"]) / "training_curve.png")
        plt.close()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            print("Recovered from CUDA OOM error")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Run training experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up paths
    project_root = Path(__file__).resolve().parents[1]
    config["paths"] = {k: str(project_root / v) for k, v in config["paths"].items()}

    training_experiment(config)


if __name__ == "__main__":
    main()
