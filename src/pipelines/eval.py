import os

import torch
from src.utils.audio import load_audio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


def evaluate_si_sdr_for_folder(
    model: torch.nn.Module,
    folder_path: str,
    device: torch.device,
    sample_rate: int = 44100,
) -> dict:
    """
    Evaluates the SI-SDR for separated sources using a model on a folder
    containing the audio files:
      - mixture.wav
      - bass.wav
      - drums.wav
      - vocals.wav
      - other.wav

    The model is expected to take the mixture as input and produce a tensor of shape [4, time],
    with the channels corresponding (in order) to vocals, drums, bass, and other.

    Args:
        model (torch.nn.Module): The source separation model.
        folder_path (str): Path to the folder containing the audio files.
        device (torch.device): Device for inference.
        sample_rate (int): Sampling rate to use.

    Returns:
        dict: SI-SDR values for each source.
    """
    mixture_path = os.path.join(folder_path, "mixture.wav")
    gt_paths = {
        "vocals": os.path.join(folder_path, "vocals.wav"),
        "drums": os.path.join(folder_path, "drums.wav"),
        "bass": os.path.join(folder_path, "bass.wav"),
        "other": os.path.join(folder_path, "other.wav"),
    }

    mixture = load_audio(mixture_path, target_sr=sample_rate).to(device)
    if mixture.shape[0] == 1:
        mixture = mixture.squeeze(0)

    gt_signals = {}
    for source, path in gt_paths.items():
        gt = load_audio(path, target_sr=sample_rate)
        gt_signals[source] = gt.squeeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(mixture)

    min_len = predictions.shape[-1]
    for source in gt_signals:
        min_len = min(min_len, gt_signals[source].shape[-1])
    predictions = predictions[..., :min_len]
    for source in gt_signals:
        gt_signals[source] = gt_signals[source][..., :min_len]

    si_sdr_metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)

    si_sdr_results = {}
    source_order = ["vocals", "drums", "bass", "other"]
    for idx, source in enumerate(source_order):
        pred_source = predictions[idx]
        target_source = gt_signals[source]
        si_sdr_value = si_sdr_metric(preds=pred_source, target=target_source)
        si_sdr_results[source] = si_sdr_value.item()

    avg_si_sdr = sum(si_sdr_results.values()) / len(si_sdr_results)
    si_sdr_results["average"] = avg_si_sdr

    return si_sdr_results
