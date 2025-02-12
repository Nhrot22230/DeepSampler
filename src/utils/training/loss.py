import torch
import torch.nn.functional as F

def si_sdr_loss(
    pred_spec: torch.Tensor,
    real_spec: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    window_fn = torch.hann_window,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calcula el SI-SDR entre una señal predicha y una real, partiendo de
    sus espectrogramas complejos. Primero se realiza la iSTFT para
    recuperar las señales en el dominio del tiempo y luego se aplica la
    fórmula de SI-SDR.

    Parámetros
    ----------
    pred_spec : torch.Tensor
        Espectrograma predicho (tensor complejo de forma [..., freq, time]).
    real_spec : torch.Tensor
        Espectrograma real (tensor complejo de forma [..., freq, time]).
    n_fft : int, opcional
        Tamaño de la FFT, por defecto 512.
    hop_length : int, opcional
        Salto entre ventanas sucesivas en la STFT, por defecto 256.
    win_length : int, opcional
        Longitud de la ventana, por defecto 512.
    window_fn : Callable, opcional
        Función que genera la ventana, por defecto torch.hann_window.
    eps : float, opcional
        Pequeña constante para evitar divisiones por cero, por defecto 1e-8.

    Retorna
    -------
    torch.Tensor
        Valor promedio de SI-SDR (en dB) para el batch.
    """

    # ------------------------------------------------
    # 1) Asegurarnos de que los tensores sean complejos
    # ------------------------------------------------
    # PyTorch 1.8+ permite tensores complejos nativos. Si tus espectrogramas
    # están separados en magnitud/fase, tendrás que combinarlos manualmente.
    # Ejemplo:
    #   pred_spec_complex = torch.complex(pred_mag, pred_phase)
    #   real_spec_complex = torch.complex(real_mag, real_phase)
    #
    # Aquí asumimos que pred_spec y real_spec ya son tensores complejos.
    # ------------------------------------------------
    
    if not pred_spec.is_complex() or not real_spec.is_complex():
        raise ValueError("Los espectrogramas deben ser tensores complejos (torch.cfloat o torch.cdouble).")

    # ------------------------------------------------
    # 2) Inversa de STFT para recuperar la señal de audio
    # ------------------------------------------------
    # Suponiendo dimensiones: [batch, freq, time] o similar.
    # Ajusta las dimensiones según tu caso de uso.
    # ------------------------------------------------
    
    device = pred_spec.device
    window = window_fn(win_length, periodic=False, device=device)

    # Señales reconstruidas en el dominio del tiempo
    pred_audio = torch.istft(
        pred_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )
    real_audio = torch.istft(
        real_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    )

    # ------------------------------------------------
    # 3) Calcular SI-SDR
    # ------------------------------------------------
    # Fórmula del SI-SDR (para cada muestra x_hat (pred) y x (real)):
    #
    #  s = <x_hat, x> / ||x||^2 * x
    #  e = x_hat - s
    #  SI-SDR = 10 * log10( ||s||^2 / ||e||^2 )
    #
    # Donde:
    #  <x_hat, x> es la multiplicación punto a punto y luego suma.
    # ------------------------------------------------

    # Asegura que pred_audio y real_audio tengan la misma dimensión
    # si trabajas en batch. Asumimos [batch, samples].
    if pred_audio.ndim == 1:
        # Expand a dimensión [1, tiempo] si es señal única.
        pred_audio = pred_audio.unsqueeze(0)
        real_audio = real_audio.unsqueeze(0)

    # Producto punto y norm
    dot = torch.sum(pred_audio * real_audio, dim=1, keepdim=True)  # <x_hat, x>
    norm_x = torch.sum(real_audio ** 2, dim=1, keepdim=True)       # ||x||^2

    # Evitar división por cero
    alpha = dot / (norm_x + eps)  # Escala
    s_target = alpha * real_audio
    e_noise = pred_audio - s_target

    # Potencias
    target_power = torch.sum(s_target ** 2, dim=1, keepdim=True)   # ||s_target||^2
    noise_power = torch.sum(e_noise ** 2, dim=1, keepdim=True)     # ||e_noise||^2

    si_sdr_linear = (target_power + eps) / (noise_power + eps)
    si_sdr_db = 10 * torch.log10(si_sdr_linear + eps)

    # Retornamos el promedio para todo el batch
    return si_sdr_db.mean()

