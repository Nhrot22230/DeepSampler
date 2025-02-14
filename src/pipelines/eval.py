import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


def evaluation_pipeline(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Evaluate the model on the test dataset using SI-SDR metric.

    Args:
        model (torch.nn.Module): Source separation model.
        test_dataloader (DataLoader): Dataloader for the test dataset.
        device (torch.device): Device on which to run evaluation.

    Returns:
        dict: Average SI-SDR values per source and overall average.
    """
    model.eval()
    si_sdr_metric = ScaleInvariantSignalDistortionRatio(zero_mean=True)

    si_sdr_totals = [0.0, 0.0, 0.0, 0.0]
    sample_count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)  # Expected shape: [batch, time]
            targets = targets.to(device)  # Expected shape: [batch, 4, time]

            outputs = model(inputs)  # Expected shape: [batch, 4, time]
            batch_size = outputs.shape[0]
            sample_count += batch_size

            # Compute SI-SDR for each source channel per sample.
            for ch in range(4):
                for i in range(batch_size):
                    pred = outputs[i, ch]
                    target = targets[i, ch]
                    si_sdr_val = si_sdr_metric(preds=pred, target=target)
                    si_sdr_totals[ch] += si_sdr_val.item()

    # Average SI-SDR values over all samples
    avg_si_sdr = [total / sample_count for total in si_sdr_totals]
    avg_results = {
        "vocals": avg_si_sdr[0],
        "drums": avg_si_sdr[1],
        "bass": avg_si_sdr[2],
        "other": avg_si_sdr[3],
        "average": sum(avg_si_sdr) / len(avg_si_sdr),
    }
    return avg_results
