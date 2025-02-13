import argparse
import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from src.models.u_net import SimpleUNet  # or SCUNet if you have that model instead
from src.utils.audio import chunk_audio, load_audio
from tqdm import tqdm

SR = 44100        # Sample rate
N_FFT = 2048      # FFT size
HOP_LENGTH = 512  # Hop length for STFT
CHUNK_DURATION = 5.0  # Duration (in seconds) for each chunk to process


def prepare_model_input(complex_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From a complex STFT, extract magnitude & phase, convert magnitude to dB,
    and prepare the input tensor for the model (adding a channel dimension).

    Args:
        complex_spec (torch.Tensor): Complex STFT of shape [1, F, T].

    Returns:
        model_input (torch.Tensor): [1, 1, F, T] (dB magnitude).
        phase (torch.Tensor): Phase of shape [1, F, T].
    """
    magnitude = torch.abs(complex_spec)  # [1, F, T]
    phase = torch.angle(complex_spec)    # [1, F, T]

    # Convert amplitude to decibels
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    magnitude_db = amp_to_db(magnitude)  # [1, F, T]

    # Model expects [batch, channels=1, freq, time]
    model_input = magnitude_db.unsqueeze(1)  # => [1, 1, F, T]
    return model_input, phase


def reconstruct_chunk(
    outputs: torch.Tensor,
    phase: torch.Tensor,
    chunk_length: int
) -> List[np.ndarray]:
    """
    Reconstruct waveforms for a chunk using the model outputs (in dB) + phase.

    Args:
        outputs (torch.Tensor): [1, out_channels, F, T] (predicted magnitudes in dB).
        phase (torch.Tensor): [1, F, T] (phase for that chunk).
        chunk_length (int): Number of samples in the original chunk.

    Returns:
        List[np.ndarray]: One waveform per output source.
    """
    # Remove batch dim => [out_channels, F, T]
    outputs_db = outputs.squeeze(0).cpu().numpy()
    # Convert dB to linear amplitude
    outputs_linear = librosa.db_to_amplitude(outputs_db, ref=1.0)  # [out_channels, F, T]

    phase_np = phase.squeeze(0).cpu().numpy()  # [F, T]

    num_channels = outputs_linear.shape[0]
    reconstructed = []

    for i in range(num_channels):
        # Complex = mag * exp(j * phase)
        complex_spec_est = outputs_linear[i] * np.exp(1j * phase_np)
        # iSTFT with librosa, ensuring we keep the chunk length consistent
        waveform = librosa.istft(
            complex_spec_est,
            hop_length=HOP_LENGTH,
            length=chunk_length
        )
        reconstructed.append(waveform)

    return reconstructed


def inference_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    stft_params: Dict[str, int],
    device: torch.device,
    output_dir: Optional[str] = None,
    chunk_duration: float = CHUNK_DURATION,
) -> Dict[str, np.ndarray]:
    """
    Full inference pipeline for source separation:
      1. Load audio (mixture).
      2. Split audio into non-overlapping chunks.
      3. Process each chunk through the model.
      4. Each chunk yields out_channels => waveforms.
      5. Concatenate these chunked waveforms per-source.
      6. Save the final waveforms if an output directory is provided.

    Args:
        model (torch.nn.Module): Trained model.
        mixture_path (str): Path to mixture.wav
        stft_params (Dict[str, int]): e.g. {"n_fft": 2048, "hop_length": 512}
        device (torch.device): 'cuda' or 'cpu'
        output_dir (Optional[str]): Where to save results if provided
        chunk_duration (float): Duration of each chunk in seconds

    Returns:
        Dict[str, np.ndarray]: Maps source name => separated waveform array
    """
    # 1) Load full mixture audio
    mixture = load_audio(mixture_path, sr=SR)  # shape [num_samples], or stereo

    # 2) Split into chunks
    chunks = chunk_audio(mixture, chunk_duration=chunk_duration, sr=SR)
    print(f"Total chunks: {len(chunks)}")

    # We'll assume the model has out_channels = 4 => [vocal, drum, bass, other]
    source_names = ["vocal", "drum", "bass", "other"]
    reconstructed_chunks = {name: [] for name in source_names}

    model.eval()
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # 3) STFT with librosa (CPU)
            complex_spec_np = librosa.stft(
                chunk, n_fft=stft_params["n_fft"], hop_length=stft_params["hop_length"]
            )  # shape [F, T]

            # Move to torch complex64 => shape [1, F, T]
            complex_spec = torch.tensor(complex_spec_np, dtype=torch.complex64).unsqueeze(0)

            # Prepare input => get dB magnitude + phase
            model_input, phase = prepare_model_input(complex_spec)
            # Move model_input to device (for GPU inference)
            model_input = model_input.to(device)

            # 4) Inference => shape [1, out_channels, F, T]
            outputs = model(model_input)

            # 5) Reconstruct chunk waveforms for each source
            rec_chunks = reconstruct_chunk(outputs, phase, chunk_length=len(chunk))

            # 6) Save them in a list to be concatenated
            for i, name in enumerate(source_names):
                reconstructed_chunks[name].append(rec_chunks[i])

    # 7) Concatenate all chunks per source
    full_reconstructed = {
        name: np.concatenate(reconstructed_chunks[name]) for name in source_names
    }

    # 8) Save to disk if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for name, waveform in full_reconstructed.items():
            out_path = os.path.join(output_dir, f"{name}.wav")
            # Save via torchaudio => waveforms need shape [channels, time]
            torchaudio.save(
                out_path,
                torch.tensor(waveform).unsqueeze(0),  # shape [1, time]
                sample_rate=SR
            )
            print(f"Saved {name} to {out_path}")

    return full_reconstructed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference pipeline for audio source separation.")
    parser.add_argument("--mixture", type=str, required=True, help="Path to mixture.wav")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--model_checkpoint", type=str, default="checkpoints/simple_unet.pth")
    args = parser.parse_args()

    # 1) Create model with the same architecture used in training
    #    For example, your SimpleUNet or SCUNet class
    model = SimpleUNet(input_channels=1, output_channels=4, depth=1)
    
    # 2) Load checkpoint
    #    Use map_location='cpu' if your checkpoint was saved on CPU, but we'll move it to GPU anyway.
    state_dict = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    # 3) Pick device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Inference device: {device}")

    # 4) STFT parameters
    stft_params = {"n_fft": N_FFT, "hop_length": HOP_LENGTH}

    # 5) Run inference
    separated = inference_pipeline(
        model=model,
        mixture_path=args.mixture,
        stft_params=stft_params,
        device=device,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
    )
    print("Inference completed.")
