import os
from typing import Dict, List

import torch
from sklearn.model_selection import train_test_split
from src.utils.audio import AudioChunk, chunk_waveform, load_audio
from tqdm import tqdm


def process_track(
    track_path: str, chunk_seconds: int = 2, sample_rate: int = 44100
) -> List[Dict[str, AudioChunk]]:
    """
    Process a single track by loading the audio files and chunking them.

    Args:
        track_path (str): Path to the track directory.
        chunk_seconds (int, optional): Duration of each chunk in seconds. Defaults to 2.
        sample_rate (int, optional): Sampling rate for loading audio. Defaults to 44100.

    Returns:
        List[Dict[str, AudioChunk]]: A list of dictionaries with chunked audio.
    """
    # Compute chunk length in samples.
    chunk_len = chunk_seconds * sample_rate

    mixture_path = os.path.join(track_path, "mixture.wav")
    bass_path = os.path.join(track_path, "bass.wav")
    drums_path = os.path.join(track_path, "drums.wav")
    other_path = os.path.join(track_path, "other.wav")
    vocals_path = os.path.join(track_path, "vocals.wav")

    mixture_wav = load_audio(mixture_path, target_sr=sample_rate, mono=True)
    bass_wav = load_audio(bass_path, target_sr=sample_rate, mono=True)
    drums_wav = load_audio(drums_path, target_sr=sample_rate, mono=True)
    other_wav = load_audio(other_path, target_sr=sample_rate, mono=True)
    vocals_wav = load_audio(vocals_path, target_sr=sample_rate, mono=True)

    mixture_chunks = chunk_waveform(mixture_wav, chunk_len, chunk_len)
    bass_chunks = chunk_waveform(bass_wav, chunk_len, chunk_len)
    drums_chunks = chunk_waveform(drums_wav, chunk_len, chunk_len)
    other_chunks = chunk_waveform(other_wav, chunk_len, chunk_len)
    vocals_chunks = chunk_waveform(vocals_wav, chunk_len, chunk_len)

    # Free memory
    del mixture_wav, bass_wav, drums_wav, other_wav, vocals_wav

    return [
        {
            "mixture": mixture,
            "bass": bass,
            "drums": drums,
            "other": other,
            "vocals": vocals,
        }
        for mixture, bass, drums, other, vocals in zip(
            mixture_chunks, bass_chunks, drums_chunks, other_chunks, vocals_chunks
        )
    ]


def musdb_pipeline(
    musdb_path: str,
    data_root: str,
    chunk_seconds: int = 2,
    sample_rate: int = 44100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Processes a music source separation dataset by chunking each track
    and splitting the processed data into train and test sets.

    Args:
        musdb_path (str): Path to the directory containing the MUSDB tracks.
        data_root (str): Root directory for the data (to create subdirectories).
        chunk_seconds (int, optional): Duration (in seconds) of each audio chunk.
        sample_rate (int, optional): Audio sample rate.
        test_size (float, optional): Proportion of tracks to use as the test set.
        random_state (int, optional): Random seed for reproducibility.
    """
    # Set up output directories.
    output_dir = os.path.join(data_root, "processed")
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    tracks = os.listdir(musdb_path)
    train_tracks, test_tracks = train_test_split(
        tracks, test_size=test_size, random_state=random_state
    )

    # Process and save train tracks.
    for track in tqdm(train_tracks, desc="Processing train tracks"):
        track_path = os.path.join(musdb_path, track)
        track_chunks = process_track(
            track_path, chunk_seconds=chunk_seconds, sample_rate=sample_rate
        )
        torch.save(track_chunks, os.path.join(train_dir, f"{track}.pt"))

    # Process and save test tracks.
    for track in tqdm(test_tracks, desc="Processing test tracks"):
        track_path = os.path.join(musdb_path, track)
        track_chunks = process_track(
            track_path, chunk_seconds=chunk_seconds, sample_rate=sample_rate
        )
        torch.save(track_chunks, os.path.join(test_dir, f"{track}.pt"))


if __name__ == "__main__":
    project_root = os.getcwd()
    while "src" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)

    data_root = os.path.join(project_root, "data")
    musdb_path = os.path.join(project_root, "data", "musdb18hq", "train")

    musdb_pipeline(
        musdb_path=musdb_path,
        data_root=data_root,
        chunk_seconds=2,
        sample_rate=44100,
        test_size=0.2,
        random_state=42,
    )
