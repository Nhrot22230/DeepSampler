import os
from typing import Dict, List, Optional

import numpy as np
import torch
from src.pipelines.infer import infer_pipeline
from src.utils.audio.processing import load_audio
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)
from tqdm.auto import tqdm

METRICS = {
    "si_sdr": scale_invariant_signal_distortion_ratio,
    "si_snr": scale_invariant_signal_noise_ratio,
    "sdr": signal_distortion_ratio,
    "snr": signal_noise_ratio,
}


def compute_audio_metrics(
    pred_audio: torch.Tensor,
    gt_audio: torch.Tensor,
    metrics: Dict[str, callable],
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """Compute audio separation metrics between predicted and ground truth signals.

    Args:
        pred_audio: Separated audio tensor of shape (batch_size?, channels?, time)
        gt_audio: Ground truth audio tensor of shape (batch_size?, channels?, time)
        metrics: Dictionary of metric functions to compute
        epsilon: Small value to prevent division by zero

    Returns:
        Dictionary of computed metric values
    """
    # Ensure audio tensors have batch dimension
    if pred_audio.ndim == 1:
        pred_audio = pred_audio.unsqueeze(0)
    if gt_audio.ndim == 1:
        gt_audio = gt_audio.unsqueeze(0)

    # Add channel dimension if missing
    if pred_audio.ndim == 2:
        pred_audio = pred_audio.unsqueeze(1)
    if gt_audio.ndim == 2:
        gt_audio = gt_audio.unsqueeze(1)

    results = {}
    for name, metric_fn in metrics.items():
        try:
            # Some metrics require additional constraints
            results[name] = metric_fn(pred_audio, gt_audio).item()
        except Exception as e:
            print(f"Metric {name} computation failed: {str(e)}")
            results[name] = float("nan")

    return results


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
    metrics: Optional[Dict[str, callable]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate audio separation quality for a single sample using multiple metrics.

    Args:
        model: Separation model to evaluate
        sample_path: Path to directory containing audio files
        instruments: List of source instruments to evaluate
        sample_rate: Audio sampling rate in Hz
        chunk_seconds: Duration of processing chunks in seconds
        overlap: Overlap ratio between chunks (0.0 to 1.0)
        n_fft: STFT window size in samples
        hop_length: STFT hop size in samples
        device: Computation device (cpu/cuda)
        metrics: Dictionary of metric functions to compute

    Returns:
        Nested dictionary containing metrics for each instrument:
        {
            "vocals": {"si_sdr": 12.3, "snr": 15.2, ...},
            "drums": {...},
            ...
        }
    """
    metrics = metrics or METRICS
    mixture_path = os.path.join(sample_path, "mixture.wav")

    # Run inference with optimized memory management
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

    results = {}
    for inst in instruments:
        gt_path = os.path.join(sample_path, f"{inst}.wav")

        try:
            # Memory-mapped loading for large files
            gt_audio = load_audio(gt_path, sample_rate)
            pred_audio = separated[inst].clone().detach()

            # Align audio lengths without copying
            min_len = min(gt_audio.shape[-1], pred_audio.shape[-1])
            gt_audio = gt_audio[..., :min_len].to(device, non_blocking=True)
            pred_audio = pred_audio[..., :min_len].to(device, non_blocking=True)

            results[inst] = compute_audio_metrics(pred_audio, gt_audio, metrics)

        except Exception as e:
            print(f"Error processing {inst} in {sample_path}: {str(e)}")
            results[inst] = {name: float("nan") for name in metrics.keys()}

    return results


def eval_pipeline(
    model: torch.nn.Module,
    dataset_path: str,
    device: torch.device,
    sample_rate: int = 44100,
    chunk_seconds: float = 5.0,
    overlap: float = 0.25,
    n_fft: int = 2048,
    hop_length: int = 512,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Optimized evaluation pipeline for music source separation models.

    Args:
        model: Separation model to evaluate
        dataset_path: Path to evaluation dataset directory
        device: Computation device (cpu/cuda)
        sample_rate: Audio sampling rate in Hz (default: 44100)
        chunk_seconds: Inference chunk duration in seconds (default: 5.0)
        overlap: Chunk overlap ratio (0.0-1.0) (default: 0.25)
        n_fft: STFT window size in samples (default: 2048)
        hop_length: STFT hop size in samples (default: 512)
        metrics: List of metrics to compute (default: all available)

    Returns:
        avg_metrics: Dictionary of average metrics per instrument

    Raises:
        ValueError: If dataset directory structure is invalid
    """
    instruments = ["vocals", "drums", "bass", "other"]
    selected_metrics = {k: v for k, v in METRICS.items() if not metrics or k in metrics}

    # Validate dataset structure
    samples = []
    for entry in os.listdir(dataset_path):
        sample_dir = os.path.join(dataset_path, entry)
        if os.path.isdir(sample_dir):
            required_files = ["mixture.wav"] + [f"{inst}.wav" for inst in instruments]
            if not all(
                os.path.exists(os.path.join(sample_dir, f)) for f in required_files
            ):
                raise ValueError(f"Invalid sample directory structure in {sample_dir}")
            samples.append(sample_dir)

    # Initialize metric storage
    all_metrics = {
        metric: {inst: [] for inst in instruments} for metric in selected_metrics.keys()
    }
    model = model.to(device).eval()

    # Process samples with progress tracking
    for sample_path in tqdm(samples, desc="Evaluating", unit="sample"):
        try:
            sample_results = eval_one_file(
                model=model,
                sample_path=sample_path,
                instruments=instruments,
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                overlap=overlap,
                n_fft=n_fft,
                hop_length=hop_length,
                device=device,
                metrics=selected_metrics,
            )

            # Aggregate results
            for inst in instruments:
                for metric in selected_metrics:
                    all_metrics[metric][inst].append(sample_results[inst][metric])
        except Exception as e:
            print(f"Skipping {sample_path} due to error: {str(e)}")
            continue

    avg_metrics = {
        metric: {inst: np.nanmean(values) for inst, values in inst_data.items()}
        for metric, inst_data in all_metrics.items()
        if metric != "samples"
    }

    return avg_metrics
