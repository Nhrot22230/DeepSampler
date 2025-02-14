import argparse
import os

import numpy as np
import torch
import torchaudio
from src.models import SCUNet
from src.utils.audio import chunk_waveform, load_audio
from tqdm import tqdm

# Configuración basada en el paper
SAMPLE_RATE = 44100
CHUNK_SECONDS = 2
N_FFT = 2048
HOP_LENGTH = 512
WINDOW = torch.hann_window(N_FFT)


def load_SCUNET(model_path, device):
    model = SCUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def inference_pipeline(model, mixture_path, output_path, device="cuda"):
    # 1. Cargar la señal y dividirla en chunks (2 segundos cada uno)
    waveform = load_audio(mixture_path)
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    chunks = chunk_waveform(waveform, chunk_len, chunk_len)

    separated_sources = [[] for _ in range(4)]  # out_channels = 4

    # 2. Procesar cada chunk
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Processing chunks"):
            stft = torch.stft(
                chunk.squeeze(0), N_FFT, HOP_LENGTH, window=WINDOW, return_complex=True
            )
            mag = torch.abs(stft).unsqueeze(0).unsqueeze(0).to(device)
            phase = torch.angle(stft).cpu().numpy()

            pred = model(mag)

            for i in range(pred.shape[1]):
                source_mag = pred[0, i].cpu().numpy()  # (freq, time)
                source_stft = source_mag * np.exp(1j * phase)
                source_wav = torch.istft(
                    torch.tensor(source_stft),
                    N_FFT,
                    HOP_LENGTH,
                    window=WINDOW,
                    length=chunk.shape[-1],
                )
                separated_sources[i].append(source_wav.numpy())

    # 3. Concatenar los chunks para cada fuente a lo largo del tiempo
    final_sources = []
    for i in range(len(separated_sources)):
        final_wave = np.concatenate(separated_sources[i], axis=-1)
        final_sources.append(final_wave)

    # 4. Guardar cada fuente en un archivo wav
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
            SAMPLE_RATE,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurar paths
    project_root = os.getcwd()
    while "src" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)

    # Cargar modelo
    model = load_SCUNET(
        os.path.join(project_root, "experiments", "checkpoints", "scunet.pth"), device
    )

    # Ejecutar pipeline
    inference_pipeline(model, args.mixture, args.output_dir, device)
