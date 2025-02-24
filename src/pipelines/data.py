import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import chunk_waveform, load_audio
from src.utils.data.dataset import MUSDBDataset

# Constants
VALID_CHANNELS = {"bass", "drums", "other", "vocals", "mixture"}
DEFAULT_SAMPLE_RATE = 44100
CHUNK_SAVE_BATCH_SIZE = 100


def _validate_channels(isolated: Optional[List[str]]) -> None:
    """Validate input channels against allowed values."""
    if isolated and not set(isolated).issubset(VALID_CHANNELS):
        invalid = set(isolated) - VALID_CHANNELS
        raise ValueError(f"Invalid channel(s) specified: {invalid}")


def _load_channel_data(
    audio_folder: Path, channel: str, sample_rate: int, chunk_len: int, hop_len: int
) -> torch.Tensor:
    """Load and chunk audio data for a single channel."""
    audio_path = audio_folder / f"{channel}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Channel file {audio_path} not found")

    waveform = load_audio(
        path=str(audio_path),
        target_sr=sample_rate,
        mono=True,
    )
    return chunk_waveform(waveform, chunk_len=chunk_len, hop_len=hop_len)


def process_audio_folder(
    audio_folder: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = 2.0,
    overlap: float = 0.0,
    isolated: Optional[List[str]] = None,
) -> List[AudioChunk]:
    """Process audio folder.

    Args:
        audio_folder: Path to audio files directory
        sample_rate: Target sample rate
        chunk_duration: Chunk length in seconds
        overlap: Overlap ratio between chunks (0-1)
        isolated: Channels to include in mixture
        device: Target device for tensor operations
        normalize: Apply peak normalization to mixture

    Returns:
        List of processed AudioChunk objects
    """
    audio_folder = Path(audio_folder)
    _validate_channels(isolated)

    chunk_len = int(chunk_duration * sample_rate)
    hop_len = int(chunk_len * (1 - overlap))
    if hop_len <= 0:
        raise ValueError("Invalid overlap ratio resulting in non-positive hop length")

    channels: Dict[str, List[torch.Tensor]] = {
        ch: _load_channel_data(audio_folder, ch, sample_rate, chunk_len, hop_len)
        for ch in VALID_CHANNELS
    }

    num_chunks = len(channels["mixture"])
    if any(len(ch) != num_chunks for ch in channels.values()):
        raise ValueError("Inconsistent chunk counts across channels")

    chunks = [
        AudioChunk(
            mixture=channels["mixture"][idx],
            vocals=channels["vocals"][idx] if "vocals" in channels else None,
            drums=channels["drums"][idx] if "drums" in channels else None,
            bass=channels["bass"][idx] if "bass" in channels else None,
            other=channels["other"][idx] if "other" in channels else None,
        )
        for idx in range(num_chunks)
    ]
    del channels

    if isolated:
        chunks = [AudioChunk.isolate_channels(chunk, isolated) for chunk in chunks]

    return chunks


def musdb_pipeline(
    musdb_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_duration: float = 2.0,
    overlap: float = 0.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_chunks: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    isolated: Optional[List[str]] = None,
) -> MUSDBDataset:
    """Optimized MUSDB processing pipeline with enhanced monitoring.

    Args:
        musdb_path: Path to MUSDB dataset root
        sample_rate: Audio sample rate
        chunk_duration: Chunk duration in seconds
        overlap: Overlap between chunks (0-1)
        n_fft: STFT window size
        hop_length: STFT hop length
        max_chunks: Maximum chunks to process
        save_dir: Directory to save processed chunks
        shuffle: Shuffle chunks before saving/returning
        isolated: Channels to isolate in mixture
        num_workers: Parallel processing workers
        persistent_workers: Maintain worker processes between epochs

    Returns:
        Configured MUSDBDataset instance
    """
    musdb_path = Path(musdb_path)
    if not musdb_path.exists():
        raise FileNotFoundError(f"MUSDB path {musdb_path} not found")

    # Create save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    track_folders = [
        f for f in musdb_path.iterdir() if f.is_dir() and not f.name.startswith(".")
    ]

    np.random.shuffle(track_folders)

    all_chunks = []
    batch_buffer = []
    processed_chunks = 0
    pbar = tqdm(total=max_chunks, desc="Total chunks", unit="chunk")

    for track_folder in tqdm(track_folders, desc="Tracks"):
        chunks = process_audio_folder(
            audio_folder=track_folder,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
            isolated=isolated,
        )

        if not chunks:
            warnings.warn(f"No chunks generated for {track_folder.name}")
            continue

        if save_dir:
            batch_buffer.extend(chunks)
            if len(batch_buffer) >= CHUNK_SAVE_BATCH_SIZE:
                _save_chunk_batch(batch_buffer, save_dir, processed_chunks)
                processed_chunks += len(batch_buffer)
                pbar.update(len(batch_buffer))
                batch_buffer.clear()
        else:
            all_chunks.extend(chunks)
            pbar.update(len(chunks))

        if max_chunks and (
            processed_chunks >= max_chunks or len(all_chunks) >= max_chunks
        ):
            all_chunks = all_chunks[:max_chunks]
            break

    # Save remaining chunks
    if batch_buffer:
        _save_chunk_batch(batch_buffer, save_dir, processed_chunks)
        pbar.update(len(batch_buffer))

    pbar.close()

    if save_dir:
        data_files = sorted(save_dir.glob("*.pt"))
        if max_chunks:
            data_files = data_files[:max_chunks]
        return MUSDBDataset(
            data=[str(f) for f in data_files], n_fft=n_fft, hop_length=hop_length
        )

    np.random.shuffle(all_chunks)
    return MUSDBDataset(data=all_chunks, n_fft=n_fft, hop_length=hop_length)


def _save_chunk_batch(chunks: List[AudioChunk], save_dir: Path, start_idx: int) -> None:
    """Save a batch of chunks with parallel I/O."""
    from concurrent.futures import ThreadPoolExecutor

    def _save_single(idx: int, chunk: AudioChunk):
        torch.save(chunk, save_dir / f"{idx:08d}.pt")

    with ThreadPoolExecutor() as executor:
        executor.map(_save_single, range(start_idx, start_idx + len(chunks)), chunks)
