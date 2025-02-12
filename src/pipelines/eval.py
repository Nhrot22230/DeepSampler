import argparse
import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from src.models.u_net import SimpleUNet
from src.utils.audio import chunk_audio, load_audio
from tqdm import tqdm

SR = 44100  # Frecuencia de muestreo
N_FFT = 2048  # Tamaño de la FFT
HOP_LENGTH = 512  # Hop length para la STFT
CHUNK_DURATION = 5.0  # Duración de cada chunk en segundos


def prepare_model_input(
    complex_spec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A partir del STFT complejo, extrae la magnitud y la fase, convierte la magnitud a dB
    y prepara el tensor de entrada para el modelo (añadiendo una dimensión de canal).

    Args:
        complex_spec (torch.Tensor): STFT complejo de forma [1, F, T].

    Returns:
        model_input (torch.Tensor): Tensor de entrada con forma [1, 1, F, T] (mag en dB).
        phase (torch.Tensor): Fase del espectrograma, de forma [1, F, T].
    """
    magnitude = torch.abs(complex_spec)  # [1, F, T]
    phase = torch.angle(complex_spec)  # [1, F, T]

    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    magnitude_db = amp_to_db(magnitude)  # [1, F, T]

    model_input = magnitude_db.unsqueeze(1)  # [1, 1, F, T]
    return model_input, phase


def reconstruct_chunk(
    outputs: torch.Tensor, phase: torch.Tensor, chunk_length: int
) -> List[np.ndarray]:
    """
    Reconstruye la forma de onda para un chunk dado la salida del modelo y la fase.

    Args:
        outputs (torch.Tensor): Salida del modelo de forma [1, out_channels, F, T] en dB.
        phase (torch.Tensor): Fase del chunk, forma [1, F, T].
        chunk_length (int): Número de muestras en el chunk.

    Returns:
        List[np.ndarray]: Lista con la forma de onda reconstruida para cada fuente.
    """
    # Eliminar la dimensión de batch: [out_channels, F, T]
    outputs_db = outputs.squeeze(0).cpu().numpy()
    # Convertir de dB a amplitud lineal
    outputs_linear = librosa.db_to_amplitude(
        outputs_db, ref=1.0
    )  # [out_channels, F, T]
    phase_np = phase.squeeze(0).cpu().numpy()  # [F, T]

    num_channels = outputs_linear.shape[0]
    reconstructed = []
    for i in range(num_channels):
        # Estimar el espectrograma complejo multiplicando la mag estimada por exp(j*fase)
        complex_spec_est = outputs_linear[i] * np.exp(1j * phase_np)
        # Reconstruir el audio con ISTFT; se fuerza la longitud del chunk
        waveform = librosa.istft(
            complex_spec_est, hop_length=HOP_LENGTH, length=chunk_length
        )
        reconstructed.append(waveform)
    return reconstructed


# -------------------------------
# Pipeline de Inferencia Completo
# -------------------------------


def inference_pipeline(
    model: torch.nn.Module,
    mixture_path: str,
    stft_params: Dict[str, int],
    device: torch.device,
    output_dir: Optional[str] = None,
    chunk_duration: float = CHUNK_DURATION,
) -> Dict[str, np.ndarray]:
    """
    Pipeline de inferencia completo para la separación de fuentes:
      1. Carga el audio (mixture.wav).
      2. Divide el audio en chunks sin solapamiento.
      3. Procesa cada chunk a través del modelo.
      4. Para cada chunk, se obtienen 4 salidas (una por fuente).
      5. Se reconstruye cada chunk mediante ISTFT y se concatenan los chunks por fuente.
      6. Se guardan los resultados en el directorio de salida.

    Args:
        model (torch.nn.Module): Modelo entrenado.
        mixture_path (str): Ruta al archivo mixture.wav.
        stft_params (Dict[str, int]): Parámetros STFT (n_fft, hop_length).
        device (torch.device): Dispositivo para inferencia.
        output_dir (Optional[str]): Directorio para guardar las fuentes reconstruidas.
        chunk_duration (float): Duración de cada chunk en segundos.

    Returns:
        Dict[str, np.ndarray]: Diccionario que mapea nombres de fuente a la forma de onda
        reconstruida.
    """
    # 1. Cargar el audio
    mixture = load_audio(mixture_path, sr=SR)

    # 2. Dividir el audio en chunks sin overlapping
    chunks = chunk_audio(mixture, chunk_duration=chunk_duration, sr=SR)
    print(f"Total chunks: {len(chunks)}")

    # Inicializar contenedores para cada fuente
    source_names = ["vocal", "drum", "bass", "other"]
    reconstructed_chunks = {name: [] for name in source_names}

    model.eval()
    with torch.no_grad():
        # Procesar cada chunk individualmente
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # 3. Calcular la STFT compleja del chunk
            complex_spec = librosa.stft(
                chunk, n_fft=stft_params["n_fft"], hop_length=stft_params["hop_length"]
            )
            complex_spec = torch.tensor(complex_spec, dtype=torch.complex64).unsqueeze(
                0
            )  # [1, F, T]
            # 4. Preparar la entrada para el modelo y obtener la fase
            model_input, phase = prepare_model_input(complex_spec)
            model_input = model_input.to(device)
            # 5. Ejecutar inferencia: se espera salida de forma [1, out_channels, F, T]
            outputs = model(model_input)
            # 6. Reconstruir el chunk para cada fuente
            rec_chunks = reconstruct_chunk(outputs, phase, chunk_length=len(chunk))
            for i, name in enumerate(source_names):
                reconstructed_chunks[name].append(rec_chunks[i])

    # 7. Concatenar todos los chunks para cada fuente a lo largo del tiempo
    full_reconstructed = {
        name: np.concatenate(reconstructed_chunks[name]) for name in source_names
    }

    # 8. Guardar resultados si se especifica output_dir
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for name, waveform in full_reconstructed.items():
            output_path = os.path.join(output_dir, f"{name}.wav")
            # Convertir a tensor con dimensión de batch y guardar usando torchaudio.save
            torchaudio.save(
                output_path, torch.tensor(waveform).unsqueeze(0), sample_rate=SR
            )
            print(f"Saved {name} to {output_path}")
    else:
        for name in full_reconstructed:
            print(f"Reconstructed source: {name}")

    return full_reconstructed


# -------------------------------
# Ejecución del pipeline de inferencia
# -------------------------------
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
    parser.add_argument(
        "--chunk_duration", type=float, default=5.0, help="Chunk duration in seconds"
    )
    args = parser.parse_args()

    # Cargar el modelo entrenado (asegúrate de ajustar los parámetros y el checkpoint)

    # Inicializar el modelo con parámetros que coincidan con el entrenamiento
    model = SimpleUNet(input_channels=1, output_channels=4, depth=1)
    checkpoint_path = "experiments/results/simple_unet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Parámetros STFT
    STFT_PARAMS = {"n_fft": N_FFT, "hop_length": HOP_LENGTH}

    reconstructed = inference_pipeline(
        model=model,
        mixture_path=args.mixture,
        stft_params=STFT_PARAMS,
        device=device,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
    )
