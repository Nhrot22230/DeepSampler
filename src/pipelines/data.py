import os
import logging
from typing import Dict, List, Union

import torch
from sklearn.model_selection import train_test_split
from src.utils.audio import AudioChunk, chunk_waveform, load_audio
from src.utils.data import MUSDB18Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def process_track(
    track_path: str,
    chunk_seconds: int = 2,
    sample_rate: int = 44100,
    instruments: List[str] = None,
) -> List[Dict[str, AudioChunk]]:
    """
    Process a single track by loading the audio files and chunking them.

    Args:
        track_path (str): Path to the track directory.
        chunk_seconds (int, optional): Duration of each chunk in seconds. Defaults to 2.
        sample_rate (int, optional): Sampling rate for loading audio. Defaults to 44100.
        instruments (List[str], optional): List of instrument names to process.

    Returns:
        List[Dict[str, AudioChunk]]: A list of dictionaries,
          each containing chunked audio for the given instruments.
    """
    if instruments is None:
        instruments = ["mixture", "bass", "drums", "other", "vocals"]

    chunk_len = chunk_seconds * sample_rate

    audio_files = {}
    for instrument in instruments:
        file_path = os.path.join(track_path, f"{instrument}.wav")
        if not os.path.exists(file_path):
            logging.warning(f"Missing file: {file_path}")
            continue
        try:
            audio_files[instrument] = load_audio(
                file_path, target_sr=sample_rate, mono=True
            )
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue

    if "mixture" not in audio_files:
        logging.error(f"Track {track_path} has no mixture file. Skipping processing.")
        return []

    chunks_dict = {}
    for instrument, waveform in audio_files.items():
        chunks_dict[instrument] = chunk_waveform(waveform, chunk_len, chunk_len)

    num_chunks = len(chunks_dict["mixture"])
    processed_chunks = []
    for idx in range(num_chunks):
        chunk_data = {}
        for instrument in instruments:
            if instrument in chunks_dict and idx < len(chunks_dict[instrument]):
                chunk_data[instrument] = chunks_dict[instrument][idx]
            else:
                chunk_data[instrument] = None
        processed_chunks.append(chunk_data)

    del audio_files, chunks_dict
    return processed_chunks


def save_track_chunks(chunks: List[Dict[str, AudioChunk]], save_path: str) -> None:
    """
    Save processed track chunks to a .pt file.

    Args:
        chunks (List[Dict[str, AudioChunk]]): List of dictionaries containing chunks.
        save_path (str): File path to save the processed chunks.
    """
    torch.save(chunks, save_path)
    logging.info(f"Saved processed track to {save_path}")


def musdb_pipeline(
    musdb_path: str,
    data_root: str,
    chunk_seconds: int = 2,
    sample_rate: int = 44100,
    test_size: float = 0.2,
    random_state: int = 42,
    return_dataset: bool = False,
    instruments: List[str] = None,
) -> Union[None, torch.utils.data.Dataset]:
    """
    Processes a music source separation dataset by chunking each track and splitting the
    processed data into train and test sets.

    Args:
        musdb_path (str): Path to the directory containing MUSDB tracks.
        data_root (str): Root directory for saving processed data.
        chunk_seconds (int, optional): Duration of each audio chunk. Defaults to 2.
        sample_rate (int, optional): Audio sample rate. Defaults to 44100.
        test_size (float, optional): Proportion of tracks to use as test set.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        return_dataset (bool, optional): If True, returns a torch.utils.data.Dataset.
        instruments (List[str], optional): List of instruments to process.

    Returns:
        Union[None, torch.utils.data.Dataset]: Returns a dataset if requested,
          otherwise None.
    """
    output_dir = os.path.join(data_root, "processed")
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    tracks = sorted(os.listdir(musdb_path))
    train_tracks, test_tracks = train_test_split(
        tracks, test_size=test_size, random_state=random_state
    )
    logging.info(f"Total tracks found: {len(tracks)}")
    logging.info(
        f"Train/Test split: {len(train_tracks)} train / {len(test_tracks)} test"
    )

    for track in tqdm(train_tracks, desc="Processing train tracks"):
        track_path = os.path.join(musdb_path, track)
        track_chunks = process_track(
            track_path,
            chunk_seconds=chunk_seconds,
            sample_rate=sample_rate,
            instruments=instruments,
        )
        if track_chunks:
            save_track_chunks(track_chunks, os.path.join(train_dir, f"{track}.pt"))
        else:
            logging.warning(f"Skipping track {track} due to processing issues.")

    # Process and save test tracks.
    for track in tqdm(test_tracks, desc="Processing test tracks"):
        track_path = os.path.join(musdb_path, track)
        track_chunks = process_track(
            track_path,
            chunk_seconds=chunk_seconds,
            sample_rate=sample_rate,
            instruments=instruments,
        )
        if track_chunks:
            save_track_chunks(track_chunks, os.path.join(test_dir, f"{track}.pt"))
        else:
            logging.warning(f"Skipping track {track} due to processing issues.")

    if return_dataset:
        logging.info("Creating and returning a MusdbDataset from the training data.")
        return MUSDB18Dataset(train_dir)

    return None


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
        return_dataset=False,
    )
