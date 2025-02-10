import torch


def compute_snr(
    target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calcula el Signal-to-Noise Ratio (SNR) en decibelios.

    La fórmula utilizada es:
        SNR = 10 * log10 ( ||target||^2 / ||target - estimate||^2 )

    Args:
        target (torch.Tensor): Señal objetivo, tensor de forma (..., T).
        estimate (torch.Tensor): Señal estimada, tensor de forma (..., T).
        eps (float): Pequeño valor para estabilidad numérica.

    Returns:
        torch.Tensor: Tensor con el SNR calculado para cada elemento del batch.
    """
    noise = target - estimate
    snr_val = 10 * torch.log10(
        (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return snr_val


def compute_si_sdr(
    target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calcula el Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) en decibelios.

    Para hacerlo, se centra (se resta la media) tanto la señal objetivo como la estimada,
    se calcula el factor de escalado óptimo y se computa la relación entre la energía del
    proyección de la señal objetivo y la energía del error.

    Args:
        target (torch.Tensor): Señal objetivo, tensor de forma (..., T).
        estimate (torch.Tensor): Señal estimada, tensor de forma (..., T).
        eps (float): Pequeño valor para estabilidad numérica.

    Returns:
        torch.Tensor: Tensor con el SI-SDR calculado para cada elemento del batch.
    """
    # Restar la media para hacer las señales de cero-media
    target_zero_mean = target - torch.mean(target, dim=-1, keepdim=True)
    estimate_zero_mean = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # Calcular el factor de escalado óptimo
    scale = torch.sum(target_zero_mean * estimate_zero_mean, dim=-1, keepdim=True) / (
        torch.sum(target_zero_mean**2, dim=-1, keepdim=True) + eps
    )
    projection = scale * target_zero_mean
    noise = estimate_zero_mean - projection
    si_sdr_val = 10 * torch.log10(
        (torch.sum(projection**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return si_sdr_val


def compute_sdr(
    target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Calcula el Scale-Dependent Signal-to-Distortion Ratio (SDR) en decibelios.

    A diferencia del SI-SDR, en este caso no se centra (no se sustrae la media),
    lo que hace que la métrica dependa de la escala de la señal.

    Args:
        target (torch.Tensor): Señal objetivo, tensor de forma (..., T).
        estimate (torch.Tensor): Señal estimada, tensor de forma (..., T).
        eps (float): Pequeño valor para estabilidad numérica.

    Returns:
        torch.Tensor: Tensor con el SDR calculado para cada elemento del batch.
    """
    scale = torch.sum(target * estimate, dim=-1, keepdim=True) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    projection = scale * target
    noise = estimate - projection
    sdr_val = 10 * torch.log10(
        (torch.sum(projection**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    )
    return sdr_val


def snr_loss(
    estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Función de pérdida basada en el SNR (negativo).

    Se desea maximizar el SNR, por lo que la pérdida se define como el negativo
    del SNR promedio.
    """
    return -compute_snr(target, estimate, eps).mean()


def si_sdr_loss(
    estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Función de pérdida basada en el SI-SDR (negativo).

    Se desea maximizar el SI-SDR, por lo que la pérdida se define como el negativo
    del SI-SDR promedio.
    """
    return -compute_si_sdr(target, estimate, eps).mean()


def sdr_loss(
    estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Función de pérdida basada en el SDR (negativo).

    Se desea maximizar el SDR, por lo que la pérdida se define como el negativo
    del SDR promedio.
    """
    return -compute_sdr(target, estimate, eps).mean()
