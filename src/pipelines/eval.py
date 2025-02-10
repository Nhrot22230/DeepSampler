import argparse
import os
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torchaudio
from src.models.u_net import SimpleUNet

# Inference STFT parameters (should match training)
STFT_PARAMS = {
    "n_fft": 2048,
    "hop_length": 512,
}

SR = 44100  # Sampling rate (should match training)


def load_mixture(mixture_path: str, sr: int = SR) -> np.ndarray:
    """
    Loads a mixture.wav file, converting it to mono if necessary.

    Args:
        mixture_path (str): Path to the mixture.wav file.
        sr (int): Sampling rate.

    Returns:
        np.ndarray: 1D numpy array containing the audio waveform.
    """
    y, _ = librosa.load(mixture_path, sr=sr, mono=True)
    return y


def compute_complex_stft(
    waveform: np.ndarray, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    Computes the complex-valued STFT of a waveform using torchaudio.

    Args:
        waveform (np.ndarray): 1D audio waveform.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.

    Returns:
        torch.Tensor: Complex-valued STFT with shape [1, F, T].
    """
    # Convert waveform to tensor and add batch dimension
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # shape: [1, T]
    # Use torchaudio's Spectrogram with power=None to obtain complex numbers.
    # (If using torchaudio >= 0.7, you can set return_complex=True.)
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=None,  # returns complex-valued spectrogram
    )
    complex_spec = stft_transform(waveform_tensor)  # shape: [1, F, T]
    return complex_spec


def prepare_model_input(
    complex_spec: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    Prepares the model input by converting the magnitude to dB.

    Args:
        complex_spec (torch.Tensor): Complex STFT, shape [1, F, T].
        n_fft (int): FFT window size.
        hop_length (int): Hop length.

    Returns:
        torch.Tensor: Model input with shape [1, 1, F, T] (dB magnitude).
    """
    # Extract magnitude and phase
    magnitude = torch.abs(complex_spec)  # [1, F, T]
    phase = torch.angle(complex_spec)  # [1, F, T]

    # Convert magnitude to dB
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    magnitude_db = amp_to_db(magnitude)  # [1, F, T]

    # The model was trained with a single input channel; add channel dim.
    model_input = magnitude_db.unsqueeze(1)  # shape: [1, 1, F, T]
    return model_input, phase


def reconstruct_waveforms(
    model_outputs: torch.Tensor,
    mixture_phase: torch.Tensor,
    stft_params: Dict[str, int],
    waveform_length: int,
) -> Dict[str, np.ndarray]:
    """
    Reconstructs audio waveforms for each separated source by combining the estimated
    magnitude (converted from dB to linear) with the original mixture phase.

    Args:
        model_outputs (torch.Tensor): Model output tensor of shape [1, out_channels, F, T]
          in dB.
        mixture_phase (torch.Tensor): Mixture phase, shape [1, F, T].
        stft_params (Dict[str, int]): Dictionary with keys "n_fft" and "hop_length".
        waveform_length (int): Desired length of the reconstructed waveform.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping source names (e.g., "vocal", "drum")
                               to reconstructed waveforms.
    """
    # Assume model_outputs is on CPU and shape [1, out_channels, F, T]
    outputs_db = model_outputs.squeeze(0).cpu().numpy()  # shape: [out_channels, F, T]
    # Convert from dB to linear amplitude
    outputs_linear = librosa.db_to_amplitude(
        outputs_db, ref=1.0
    )  # shape: [out_channels, F, T]

    # Get the mixture phase as a numpy array
    phase = mixture_phase.squeeze(0).cpu().numpy()  # shape: [F, T]

    # Define source names for each channel; adjust as needed.
    source_names = ["vocal", "drum", "bass", "other"]
    reconstructed = {}
    for i, source in enumerate(source_names):
        # Multiply the estimated magnitude by the phase to obtain a complex spectrogram
        estimated_complex = outputs_linear[i] * np.exp(1j * phase)
        # Inverse STFT using librosa; ensure that the output length m
        # atches the original waveform length
        reconstructed_waveform = librosa.istft(
            estimated_complex,
            hop_length=stft_params["hop_length"],
            length=waveform_length,
        )
        reconstructed[source] = reconstructed_waveform
    return reconstructed


def inference_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    stft_params: Dict[str, int],
    device: torch.device,
    output_dir: Optional[str] = None,
) -> None:
    """
    Complete inference pipeline: loads a mixture, processes it through the model,
    reconstructs separated audio signals, and optionally saves the results to WAV files.

    Args:
        model (torch.nn.Module): Trained model.
        mixture_path (str): Path to the mixture.wav file.
        stft_params (Dict[str, int]): STFT parameters (n_fft and hop_length).
        device (torch.device): Device on which to run inference.
        output_dir (Optional[str]): If provided, saves the reconstructed sources as WAV.
    """
    # Load mixture waveform
    mixture = load_mixture(mixture_path, sr=SR)
    waveform_length = len(mixture)

    # Compute complex STFT of the mixture to obtain phase information
    complex_spec = compute_complex_stft(
        mixture, stft_params["n_fft"], stft_params["hop_length"]
    )

    # Prepare the model input (convert magnitude to dB, add channel dimension)
    model_input, phase = prepare_model_input(
        complex_spec, stft_params["n_fft"], stft_params["hop_length"]
    )
    model_input = model_input.to(device)

    # Set model to evaluation mode and run inference
    model.eval()
    with torch.no_grad():
        outputs = model(model_input)  # Expected shape: [1, out_channels, F, T]

    # Reconstruct separated waveforms by combining model output (converted back to linear)
    reconstructed_sources = reconstruct_waveforms(
        outputs, phase, stft_params, waveform_length
    )

    # Optionally, save the outputs as WAV files
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for source, waveform in reconstructed_sources.items():
            output_path = os.path.join(output_dir, f"{source}.wav")
            torchaudio.save(
                output_path, torch.tensor(waveform).unsqueeze(0), sample_rate=SR
            )
            print(f"Saved reconstructed {source} to {output_path}")
    else:
        # Otherwise, simply print a message and/or play the audio (if desired)
        for source in reconstructed_sources:
            print(f"Reconstructed source: {source}")

    return reconstructed_sources


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference pipeline for audio source separation."
    )
    parser.add_argument(
        "--mixture", type=str, required=True, help="Path to mixture.wav"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save separated sources",
    )
    args = parser.parse_args()

    # Load your trained model (this assumes you have a checkpoint and model definition)
    # For example, assume your model is an instance of SimpleUNet:

    # Adjust the parameters if needed:
    model = SimpleUNet(
        input_channels=1, output_channels=4, base_channels=64, depth=4, dropout_prob=0.3
    )
    # Load model weights (adjust the path accordingly)
    checkpoint_path = "experimets/results/simple_unet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    reconstructed = inference_pipeline(
        model=model,
        mixture_path=args.mixture,
        stft_params=STFT_PARAMS,
        device=device,
        output_dir=args.output_dir,
    )
