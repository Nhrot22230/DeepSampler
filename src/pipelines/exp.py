import argparse
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from src.models import DeepSampler
from src.pipelines.data import musdb_pipeline
from src.pipelines.train import train_pipeline
from src.utils.data.dataset import MUSDBDataset
from src.utils.train.losses import MultiSourceLoss


def training_experiment(config):
    # Decorative header for experiment start
    print("\n" + "=" * 60)
    print("🚀  STARTING TRAINING EXPERIMENT  🚀".center(60))
    print("=" * 60 + "\n")

    # Print the experiment configuration using pprint with decorations
    print("✨  EXPERIMENT CONFIG  ✨".center(60))
    pprint(config)
    print("\n" + "=" * 60)

    # Decorative device setup information
    print("🔧  DEVICE SETUP  🔧".center(60))
    print("=" * 60)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'Device:':20} {device_type}")
    device_count = torch.cuda.device_count()
    print(f"{'Device count:':20} {device_count}")
    if torch.cuda.is_available() and device_count > 0:
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{'Device name:':20} {device_name}")
        print(f"{'Device memory:':20} {device_memory:.2f} GB")
    print("=" * 60)

    # Calculate and print the spectrogram shape with decoration
    spectrogram_shape = (
        config["audio_params"]["n_fft"] // 2 + 1,
        config["audio_params"]["chunk_duration"]
        * config["audio_params"]["sr"]
        // config["audio_params"]["hop_length"]
        + 1,
    )
    print(
        f"Calculated Spectrogram shape: {spectrogram_shape[0]} x {spectrogram_shape[1]}"
    )
    print("\n" + "=" * 60 + "\n")

    """Main training experiment function with configurable parameters"""
    instruments = ["vocals", "drums", "bass", "other"]

    # Device setup
    device = torch.device(device_type)

    # Model parameters and instantiation
    model_params = config["model_params"]
    model = DeepSampler(
        out_ch=model_params["n_sources"],
        base_ch=model_params["base_channels"],
        depth=model_params["depth"],
        dropout=model_params["drop_rate"],
        t_heads=model_params["transformer_heads"],
        t_layers=model_params["transformer_layers"],
    )

    # Enable multi-GPU training if multiple GPUs are available.
    if device_count > 1:
        device_ids = list(range(device_count))
        print(f"Using GPUs: {device_ids} for training")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # Audio parameters
    audio_params = config["audio_params"]
    SR = audio_params["sr"]
    NFFT = audio_params["n_fft"]
    HOP = audio_params["hop_length"]
    CHUNK_DUR = audio_params["chunk_duration"]
    OVERLAP = audio_params["overlap"]

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
                    model.parameters(), lr=float(train_params["isolated_lr"])
                ),
                dataloader=isolated_loader,
                epochs=train_params["isolated_epochs"],
                device=device,
            )
        del isolated_datasets, isolated_loader
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
                model.parameters(), lr=float(train_params["mixed_lr"])
            ),
            dataloader=train_loader,
            epochs=train_params["mixed_epochs"],
            checkpoint_name=config["training_params"]["checkpoint_name"],
            checkpoint_dir=Path(config["paths"]["checkpoints"]),
            checkpoint_every=train_params["checkpoint_interval"],
            device=device,
        )
        del train_dataset, train_loader
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

    # create directories for checkpoints logs and results
    Path(config["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["logs"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["results"]).mkdir(parents=True, exist_ok=True)

    # verify if data directory exists and is not empty
    if not Path(config["paths"]["musdb_train"]).exists() or not list(
        Path(config["paths"]["musdb_train"]).glob("*")
    ):
        raise FileNotFoundError(
            f"Data directory {config['paths']['musdb_train']} does not exist or is empty"
        )
    if not Path(config["paths"]["musdb_test"]).exists() or not list(
        Path(config["paths"]["musdb_test"]).glob("*")
    ):
        raise FileNotFoundError(
            f"Data directory {config['paths']['musdb_test']} does not exist or is empty"
        )

    training_experiment(config)


if __name__ == "__main__":
    main()
