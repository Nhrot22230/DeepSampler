import os
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import chunk_waveform, load_audio
from src.utils.data.dataset import MUSDBDataset
from src.utils.logging import main_logger as logger


def process_audio_folder(
    audio_folder: str,
    sample_rate: int,
    chunk_duration: float,
    overlap: float,
    isolated: Optional[List[str]] = None,
) -> List[AudioChunk]:
    """
    Processes an audio folder and returns a list of AudioChunk objects.

    If `isolated` is provided, the mixture is computed as the sum of the specified
    isolated channels, and non-specified channels are replaced with zeros.
    Valid channel keys are: "bass", "drums", "other", "vocals". The loaded "mixture"
    file is ignored in this case.

    Args:
        audio_folder (str): Path to the folder containing audio files.
        sample_rate (int): Audio sample rate.
        chunk_duration (float): Duration (in seconds) of each chunk.
        overlap (float): Fraction of overlap between chunks.
        isolated (Optional[List[str]]): List of channels to use for computing the mixture.
                                        If None, the loaded "mixture" file is used.

    Returns:
        List[AudioChunk]: A list of processed AudioChunk objects.
    """
    channels: List[str] = ["mixture", "bass", "drums", "other", "vocals"]
    if isolated is not None:
        isolated = [ch for ch in isolated if ch in channels and ch != "mixture"]

    chunk_len = int(chunk_duration * sample_rate)
    hop_len = int(chunk_len * (1 - overlap))

    channel_chunks = {}
    for ch in channels:
        audio_path = os.path.join(audio_folder, f"{ch}.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
        waveform = load_audio(path=audio_path, target_sr=sample_rate, mono=True)
        chunks = chunk_waveform(waveform, chunk_len=chunk_len, hop_len=hop_len)
        channel_chunks[ch] = chunks

    num_chunks = len(channel_chunks[channels[0]])
    for ch in channels:
        if len(channel_chunks[ch]) != num_chunks:
            raise ValueError(f"Chunk count mismatch for channel '{ch}'.")

    audio_chunks = []
    for i in range(num_chunks):
        if isolated:
            mixture_chunk = sum(channel_chunks[ch][i] for ch in isolated)
            bass_chunk = (
                channel_chunks["bass"][i]
                if "bass" in isolated
                else torch.zeros_like(channel_chunks["bass"][i])
            )
            drums_chunk = (
                channel_chunks["drums"][i]
                if "drums" in isolated
                else torch.zeros_like(channel_chunks["drums"][i])
            )
            other_chunk = (
                channel_chunks["other"][i]
                if "other" in isolated
                else torch.zeros_like(channel_chunks["other"][i])
            )
            vocals_chunk = (
                channel_chunks["vocals"][i]
                if "vocals" in isolated
                else torch.zeros_like(channel_chunks["vocals"][i])
            )
        else:
            mixture_chunk = channel_chunks["mixture"][i]
            bass_chunk = channel_chunks["bass"][i]
            drums_chunk = channel_chunks["drums"][i]
            other_chunk = channel_chunks["other"][i]
            vocals_chunk = channel_chunks["vocals"][i]

        chunk_obj = AudioChunk(
            mixture=mixture_chunk,
            bass=bass_chunk,
            drums=drums_chunk,
            other=other_chunk,
            vocals=vocals_chunk,
        )
        audio_chunks.append(chunk_obj)

    return audio_chunks


def musdb_pipeline(
    musdb_path: str,
    sample_rate: int = 44100,
    chunk_duration: float = 2,
    overlap: float = 0,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_chunks: Optional[int] = None,
    save_dir: Optional[str] = None,
    shuffle: bool = False,
    isolated: Optional[List[str]] = None,
) -> MUSDBDataset:
    """
    Processes the MUSDB18 dataset by iterating over track folders, extracting audio chunks,
    and optionally saving each AudioChunk to disk.

    If 'isolated' is provided, the mixture is computed as the sum of the channels in the list.
    If 'shuffle' is True, the chunks within each track folder are shuffled.

    Args:
        musdb_path: Path to the MUSDB18 dataset.
        sample_rate: Audio sample rate.
        chunk_duration: Duration (in seconds) of each chunk.
        overlap: Fraction of overlap between chunks.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        max_chunks: Maximum number of chunks to process.
        save_dir: If provided, directory to save processed AudioChunks as .pt files.
        shuffle: Whether to shuffle chunks within each track.
        isolated: List of instrument channels to compute the mixture (e.g. ["vocals"]).
                  If provided, the mixture is computed as the sum of these channels.

    Returns:
        MUSDBDataset: A dataset containing all processed AudioChunks or file paths.
    """
    if not os.path.isdir(musdb_path):
        raise FileNotFoundError(f"MUSDB path '{musdb_path}' does not exist.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_audio_chunks = []
    processed_chunks = 0

    track_folders = [
        os.path.join(musdb_path, folder)
        for folder in os.listdir(musdb_path)
        if os.path.isdir(os.path.join(musdb_path, folder))
    ]

    if shuffle:
        track_folders = np.random.permutation(track_folders)

    logger.info(f"Found {len(track_folders)} track folders in '{musdb_path}'.")

    for track_folder in tqdm(track_folders, desc="Processing tracks"):
        audio_chunks = process_audio_folder(
            audio_folder=track_folder,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
            isolated=isolated,
        )
        if not audio_chunks:
            logger.warning(f"No audio chunks found in '{track_folder}'.")
            raise ValueError("No audio chunks found.")

        if max_chunks is not None:
            remaining = max_chunks - processed_chunks
            if remaining <= 0:
                break
            audio_chunks = audio_chunks[:remaining]

        if save_dir:
            for idx, chunk in enumerate(audio_chunks):
                filename = f"{processed_chunks + idx:08d}.pt"
                torch.save(chunk, os.path.join(save_dir, filename))
        else:
            all_audio_chunks.extend(audio_chunks)

        processed_chunks += len(audio_chunks)
        if max_chunks is not None and processed_chunks >= max_chunks:
            break

    if save_dir:
        data_files = sorted(
            [
                os.path.join(save_dir, f)
                for f in os.listdir(save_dir)
                if f.endswith(".pt")
            ]
        )
        if max_chunks is not None:
            data_files = data_files[:max_chunks]
        return MUSDBDataset(data=data_files, n_fft=n_fft, hop_length=hop_length)

    return MUSDBDataset(data=all_audio_chunks, n_fft=n_fft, hop_length=hop_length)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MUSDB18 dataset.")

    parser.add_argument(
        "--musdb_path",
        type=str,
        required=True,
        help="Path to the MUSDB18 dataset.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save processed chunks.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate for audio processing.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=2,
        help="Duration of each audio chunk in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0,
        help="Overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="Number of FFT points.",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Number of samples between successive FFT columns.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=100,
        help="Maximum number of chunks to process.",
    )

    args = parser.parse_args()

    dataset = musdb_pipeline(
        musdb_path=args.musdb_path,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        max_chunks=args.max_chunks,
        save_dir=args.save_dir,
    )

    logger.info(f"Processed {len(dataset)} audio chunks.")
    logger.info(f"Saved processed chunks to '{args.save_dir}'.")
    logger.info(f"Dataset: {dataset}")
