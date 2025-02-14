import argparse
import os

import numpy as np
import torch
import torchaudio
from src.models import SCUNet
from src.utils.audio import chunk_waveform, load_audio
from tqdm import tqdm


def load_SCUNET(model_path, device):
    model = SCUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    output_path: str,
    device: torch.device = torch.device("cpu"),
    sample_rate: int = 44100,
    chunk_seconds: float = 2,
    n_fft: int = 2048,
    hop_length: int = 512,
):
    """
    Run the inference pipeline on the given audio mixture.

    Parameters:
      model (torch.nn.Module): Loaded SCUNet model.
      mixture_path (str): Path to the input mixture wav file.
      output_path (str): Directory to save the separated source wav files.
      device (str): Device to perform computations ('cuda' or 'cpu').
      sample_rate (int): Sampling rate for audio.
      chunk_seconds (int or float): Duration of each audio chunk in seconds.
      n_fft (int): FFT window size.
      hop_length (int): Hop length for STFT.
    """
    # Create the Hann window for both STFT and ISTFT, and send it to the device.
    window = torch.hann_window(n_fft, device=device)

    # 1. Load the audio and divide it into chunks.
    waveform = load_audio(mixture_path, target_sr=sample_rate, mono=True)
    chunk_len = int(chunk_seconds * sample_rate)
    chunks = chunk_waveform(waveform, chunk_len, chunk_len)

    # Prepare a list to collect separated sources (assuming 4 output channels).
    separated_sources = [[] for _ in range(4)]

    # 2. Process each chunk.
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk = chunk.squeeze(0).to(device)
            stft = torch.stft(
                chunk, n_fft, hop_length, window=window, return_complex=True
            )
            mag = torch.abs(stft).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, freq, time)
            phase = torch.angle(stft).cpu().numpy()

            # Run the model.
            pred = model(mag)

            # Reconstruct each source.
            for i in range(pred.shape[1]):
                source_mag = pred[0, i].cpu().numpy()  # (freq, time)
                source_stft = source_mag * np.exp(1j * phase)
                # Convert back to tensor and perform inverse STFT.
                source_stft_tensor = torch.tensor(source_stft, device=device)
                source_wav = torch.istft(
                    source_stft_tensor,
                    n_fft,
                    hop_length,
                    window=window,
                    length=chunk_len,
                )
                separated_sources[i].append(source_wav.cpu().numpy())

    # 3. Concatenate the chunks for each source along the time axis.
    final_sources = []
    for i in range(len(separated_sources)):
        final_wave = np.concatenate(separated_sources[i], axis=-1)
        final_sources.append(final_wave)

    # 4. Save each source as a WAV file.
    sources = {
        "vocals": final_sources[0],
        "drums": final_sources[1],
        "bass": final_sources[2],
        "other": final_sources[3],
    }

    os.makedirs(output_path, exist_ok=True)
    for name, audio in sources.items():
        torchaudio.save(
            os.path.join(output_path, f"{name}.wav"),
            torch.tensor(audio).unsqueeze(0),
            sample_rate,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference pipeline for DeepSampler")
    parser.add_argument(
        "--mixture", type=str, required=True, help="Path to the mixture audio file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the separated sources",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Audio sample rate (default: 44100)",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=2,
        help="Duration of each chunk in seconds (default: 2)",
    )
    parser.add_argument(
        "--n_fft", type=int, default=2048, help="FFT window size (default: 2048)"
    )
    parser.add_argument(
        "--hop_length", type=int, default=512, help="Hop length for STFT (default: 512)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.getcwd()
    while "src" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)

    model_path = os.path.join(project_root, "experiments", "checkpoints", "scunet.pth")
    model = load_SCUNET(model_path, device)

    inference_pipeline(
        model,
        args.mixture,
        args.output_dir,
        device=device,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
