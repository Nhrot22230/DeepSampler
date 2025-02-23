import os
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)
from src.pipelines.infer import infer_pipeline
from src.utils.audio.processing import load_audio

METRICS = {
    "si_sdr": scale_invariant_signal_distortion_ratio,
    "si_snr": scale_invariant_signal_noise_ratio,
    "sdr": signal_distortion_ratio,
    "snr": signal_noise_ratio,
}


def eval_one_file(
    model: torch.nn.Module,
    sample_path: str,
    instruments: List[str],
    sample_rate: int,
    chunk_seconds: float,
    overlap: float,
    n_fft: int,
    hop_length: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate separation quality for a single audio sample with multiple metrics.

    Args:
        model: Separation model to evaluate
        sample_path: Path to directory containing mixture and sources
        instruments: List of source instruments to evaluate
        sample_rate: Audio sampling rate
        chunk_seconds: Inference chunk duration
        overlap: Chunk overlap ratio
        n_fft: STFT window size
        hop_length: STFT hop size
        device: Computation device

    Returns:
        Dictionary with metrics for each instrument
    """
    mixture_path = os.path.join(sample_path, "mixture.wav")

    # Run inference with memory optimization
    with torch.inference_mode(), torch.amp.autocast(device.type):
        separated = infer_pipeline(
            model=model,
            mixture=mixture_path,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            overlap=overlap,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device,
        )

    metrics = {}
    for inst in instruments:
        gt_path = os.path.join(sample_path, f"{inst}.wav")

        # Load with memory-mapped I/O for large files
        gt_audio = load_audio(gt_path, sample_rate, mmap=True).to(device)
        pred_audio = separated[inst].to(device, non_blocking=True)

        # Align lengths using non-copy slicing
        min_len = min(gt_audio.shape[-1], pred_audio.shape[-1])
        gt_audio = gt_audio[..., :min_len]
        pred_audio = pred_audio[..., :min_len]

        # Compute all metrics in single forward pass
        inst_metrics = {}
        for name, metric_fn in METRICS.items():
            try:
                inst_metrics[name] = metric_fn(pred_audio, gt_audio).item()
            except RuntimeError as e:
                print(f"Error computing {name} for {inst}: {str(e)}")
                inst_metrics[name] = float("nan")

        metrics[inst] = inst_metrics

    return metrics


def eval_pipeline(
    model: torch.nn.Module,
    dataset_path: str,
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    overlap: float = 0.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, List[float]]]]:
    """Optimized evaluation pipeline with multi-metric tracking and memory management.

    Args:
        model: Separation model to evaluate
        dataset_path: Path to evaluation dataset
        sample_rate: Audio sampling rate
        chunk_seconds: Inference chunk duration
        overlap: Chunk overlap ratio
        n_fft: STFT window size
        hop_length: STFT hop size
        device: Computation device

    Returns:
        (average_metrics, all_metrics) tuple containing:
        - average_metrics: Dictionary of average metric values per instrument
        - all_metrics: Dictionary containing all individual sample metrics
    """
    instruments = ["vocals", "drums", "bass", "other"]
    samples = [
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    # Initialize metric storage
    all_metrics = {
        metric: {inst: [] for inst in instruments} for metric in METRICS.keys()
    }
    all_metrics["samples"] = []

    model = model.to(device).eval()

    for sample_path in tqdm(samples, desc="Evaluating", unit="file"):
        try:
            sample_metrics = eval_one_file(
                model=model,
                sample_path=sample_path,
                instruments=instruments,
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                overlap=overlap,
                n_fft=n_fft,
                hop_length=hop_length,
                device=device,
            )

            # Aggregate metrics
            for inst in instruments:
                for metric in METRICS.keys():
                    all_metrics[metric][inst].append(sample_metrics[inst][metric])

            all_metrics["samples"].append(sample_path)

        except Exception as e:
            print(f"Skipping {sample_path} due to error: {str(e)}")
            continue

    # Compute averages excluding failed samples
    avg_metrics = {
        metric: {inst: np.nanmean(values) for inst, values in inst_metrics.items()}
        for metric, inst_metrics in all_metrics.items()
        if metric != "samples"
    }

    return avg_metrics, all_metrics
