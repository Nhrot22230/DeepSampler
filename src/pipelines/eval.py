import os
import torch
import torchaudio
from torch.nn.functional import pad

def load_model(model_path, SCUNetClass, device):
    """
    Loads a trained SCUNet model onto the specified device.
    
    Args:
        model_path (str): Path to the .pth file with trained weights.
        SCUNetClass (nn.Module): Your SCUNet class definition.
        device (torch.device): 'cuda' or 'cpu'.
        
    Returns:
        model (nn.Module): The loaded SCUNet in eval mode.
    """
    model = SCUNetClass()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def separate_sources(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048
):
    """
    Given a raw audio waveform, use SCUNet to separate sources.
    1) Compute STFT => Magnitude + Phase.
    2) Feed Magnitude into model.
    3) Multiply predicted magnitudes by mixture phase.
    4) iSTFT => separated waveforms.
    
    Args:
        waveform (torch.Tensor): Shape [1, num_samples] (mono) or [channels, num_samples].
        model (torch.nn.Module): The trained SCUNet model.
        device (torch.device): GPU or CPU.
        n_fft, hop_length, win_length: STFT params.
    
    Returns:
        separated_waveforms (List[torch.Tensor]): List of separated waveforms (one per output channel).
    """

    # 0) Ensure waveform on device
    waveform = waveform.to(device)
    
    # 1) STFT
    #    shape of 'spec': [batch=1, freq_bins, time_frames, 2]
    #    if you use torchaudio >= 0.10's return_complex=True, it returns complex64. 
    #    We'll keep the older approach that gives real & imag as last dimension = 2.
    spec = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=device),
        return_complex=False,
        center=True,
        pad_mode="reflect"
    )  # shape: [channels, freq, time, 2] for multi-channel input

    # If mono, you might have shape [1, freq, time, 2].
    # We'll sum or average across channels if needed, or assume 1 channel. 
    # Let's assume mono or you want to do per-channel separation. 

    # 2) Separate magnitude & phase
    real_part = spec[..., 0]
    imag_part = spec[..., 1]
    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    phase = torch.atan2(imag_part, real_part)  # shape: [channels, freq, time]

    # 3) Prepare model input: SCUNet might expect shape [batch, 1, freq, time] or [batch, freq, time].
    #    We'll assume [batch=1, in_channels=1, freq, time]. If your model is coded differently, adjust.
    #    Also ensure float32.
    magnitude = magnitude.unsqueeze(1).float()  # => [channels, 1, freq, time]
    
    # If you only have a single channel, channels=1 => shape [1,1,freq,time].
    # We'll treat 'channels' as part of batch dimension if you want to separate each channel independently.
    batch_input = magnitude  # e.g. shape: [1,1,freq,time] if mono
    batch_input = batch_input.to(device)
    
    with torch.no_grad():
        pred = model(batch_input)  # shape [batch_size, out_channels, freq, time]
    
    # Typically 'out_channels' = number of sources (e.g., 4 for [vocals, drum, bass, other]).
    # pred is the predicted *magnitude* for each source.

    # 4) Recombine predicted magnitude with mixture phase
    #    We want: complex = predicted_mag * e^{j*phase} = predicted_mag * (cos(phase) + i sin(phase))
    #    Then do iSTFT for each source.
    
    # pred shape: [1, out_channels, freq, time]
    # But we have 'phase' shape: [channels, freq, time]. If channels=1, shape => [1, freq, time]
    # We can broadcast if needed. We'll just index the first channel if we assume single-channel mixture.
    out_channels = pred.shape[1]
    
    # We'll store separated waveforms in a list
    separated_waveforms = []
    
    for source_idx in range(out_channels):
        pred_mag = pred[:, source_idx, :, :]  # shape [1, freq, time]
        # If 'channels' > 1, adjust indexing or handle each channel separately. 
        # We'll assume single channel for simplicity:
        mixture_phase = phase[0]  # shape [freq, time]

        # Reconstruct real & imag from predicted magnitude + mixture phase
        pred_real = pred_mag * torch.cos(mixture_phase)
        pred_imag = pred_mag * torch.sin(mixture_phase)

        # The stft expects shape [..., 2], so stack real & imag:
        pred_spec = torch.stack([pred_real, pred_imag], dim=-1)  # [freq, time, 2]
        pred_spec = pred_spec.unsqueeze(0)  # add channel dim => [1, freq, time, 2]

        # 5) iSTFT
        #  iSTFT might need the same parameters used above. 
        #  If there's odd length mismatch, you may want to pad/crop carefully.
        est_wav = torch.istft(
            pred_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=device),
            center=True,
            normalized=False,
            onesided=True
        )  # shape [1, num_samples]
        
        separated_waveforms.append(est_wav.cpu().squeeze(0))  # move to CPU, drop channel

    return separated_waveforms

def run_inference_pipeline(
    mixture_path: str,
    model_path: str,
    output_dir: str,
    SCUNetClass,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
):
    """
    1) Load audio from mixture_path.
    2) Load trained SCUNet model from model_path.
    3) Separate mixture into sources.
    4) Save each source as .wav in output_dir.
    """
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on device: {device}")

    # 1) Load mixture waveform
    #    torchaudio.load returns (waveform, sample_rate), shape of waveform: [channels, num_samples]
    waveform, sr = torchaudio.load(mixture_path)
    # If multi-channel, you can choose to mix down to mono or handle each channel separately.
    # For simplicity, let's do mono (sum channels):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 2) Load model
    model = load_model(model_path, SCUNetClass, device)

    # 3) Separate sources
    separated_waveforms = separate_sources(
        waveform, model, device, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )

    # 4) Save each source
    os.makedirs(output_dir, exist_ok=True)
    for i, source_wav in enumerate(separated_waveforms):
        out_path = os.path.join(output_dir, f"source_{i}.wav")
        torchaudio.save(out_path, source_wav.unsqueeze(0), sr)
        print(f"Saved separated source {i} to: {out_path}")
