import torch

NUM_SOURCES = 4
SR = 44100
CHUNK_DURATION = 2.0
N_FFT = 2048
HOP_LENGTH = 512
N_FREQ = N_FFT // 2 + 1
TIME_STEPS = int(SR * CHUNK_DURATION) // HOP_LENGTH + 1


def generate_unbatched_tensors(
    num_sources=NUM_SOURCES, n_freq=N_FREQ, time_steps=TIME_STEPS
):
    """
    Generates a list of unbatched tensors.

    Each tensor has shape [n_freq, time_steps].

    GIVEN:
        - num_sources: Number of sources to generate.
        - n_freq: Number of frequency bins.
        - time_steps: Number of time steps.
    WHEN:
        - This function is called.
    THEN:
        - It returns a list of torch.Tensor objects with shape [n_freq, time_steps].
    """
    return torch.rand(num_sources, n_freq, time_steps)


def generate_batched_tensors(
    batch_size=2, num_sources=NUM_SOURCES, n_freq=N_FREQ, time_steps=TIME_STEPS
):
    """
    Generates a list of batched tensors.

    Each tensor has shape [batch_size, n_freq, time_steps].

    GIVEN:
        - batch_size: Number of items in each batch.
        - num_sources: Number of sources to generate.
        - n_freq: Number of frequency bins.
        - time_steps: Number of time steps.
    WHEN:
        - This function is called.
    THEN:
        - It returns a list of torch.Tensor objects with shape [batch_size, n_freq, time_steps].
    """
    return torch.rand(batch_size, num_sources, n_freq, time_steps)
