import os
from typing import Dict, List, Optional

import torch
from src.utils.audio.audio_chunk import AudioChunk
from src.utils.audio.processing import chunk_waveform, load_audio
from src.utils.data.dataset import MUSDBDataset
from src.utils.logging import main_logger as logger
from tqdm import tqdm


def process_audio_folder(
    audio_folder: str,
    sample_rate: int,
    chunk_duration: float,
    overlap: float,
) -> List[AudioChunk]:
    """
    Processes an audio folder and returns a list of AudioChunks.

    Args:
        audio_folder: Path to the folder containing audio files.
        sample_rate: The sample rate to use.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Fraction of overlap between chunks.

    Returns:
        List[AudioChunk]: A list of AudioChunk objects.
    """
    if not os.path.isdir(audio_folder):
        raise FileNotFoundError(f"Audio folder '{audio_folder}' does not exist.")

    instruments: Dict[str, List[torch.Tensor]] = {
        "mixture": [],
        "bass": [],
        "drums": [],
        "other": [],
        "vocals": [],
    }

    chunk_len = int(chunk_duration * sample_rate)
    hop_len = int(chunk_len * (1 - overlap))

    for inst in instruments:
        audio_path = os.path.join(audio_folder, f"{inst}.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' not found. Skipping.")

        waveform = load_audio(path=audio_path, target_sr=sample_rate, mono=True)
        chunks = chunk_waveform(waveform=waveform, chunk_len=chunk_len, hop_len=hop_len)
        instruments[inst].extend(chunks)

    audio_chunks = [
        AudioChunk(
            mixture=m,
            bass=b,
            drums=d,
            other=o,
            vocals=v,
        )
        for m, b, d, o, v in zip(
            instruments["mixture"],
            instruments["bass"],
            instruments["drums"],
            instruments["other"],
            instruments["vocals"],
        )
    ]

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
) -> MUSDBDataset:
    """
    Processes the MUSDB18 dataset by iterating over track folders, extracting audio chunks,
    and optionally saving each AudioChunk to disk.

    Args:
        musdb_path: Path to the MUSDB18 dataset.
        sample_rate: Audio sample rate.
        chunk_duration: Duration (in seconds) of each chunk.
        overlap: Fraction of overlap between chunks.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        max_chunks: Maximum number of chunks to process.
        save_dir: If provided, directory to save processed AudioChunks as .pt files.

    Returns:
        MUSDBDataset: A dataset containing all processed AudioChunks or file paths.
    """
    if not os.path.isdir(musdb_path):
        raise FileNotFoundError(f"MUSDB path '{musdb_path}' does not exist.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_audio_chunks = []
    processed_chunks = 0

    # Get track folder paths.
    track_folders = [
        os.path.join(musdb_path, folder)
        for folder in os.listdir(musdb_path)
        if os.path.isdir(os.path.join(musdb_path, folder))
    ]
    logger.info(f"Found {len(track_folders)} track folders in '{musdb_path}'.")

    # Process each track folder.
    for track_folder in tqdm(track_folders, desc="Processing tracks"):
        audio_chunks = process_audio_folder(
            audio_folder=track_folder,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap=overlap,
        )
        if not audio_chunks:
            logger.warning(f"No audio chunks found in '{track_folder}'.")
            raise ValueError("No audio chunks found.")

        # Limit to remaining chunks if max_chunks is set.
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

    # Build dataset from saved files or in-memory chunks.
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
