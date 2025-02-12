import os
import sys
from typing import List
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

from src.utils.audio import load_audio, chunk_waveform, AudioChunk

project_root = os.getcwd()
while "src" not in os.listdir(project_root):
    project_root = os.path.dirname(project_root)
sys.path.append(project_root)

data_root = os.path.join(project_root, "data")
musdb_path = os.path.join(project_root, "data", "musdb18hq")
chunk_seconds = 4
sample_rate = 44100
chunk_len = chunk_seconds * sample_rate
overlap = 0.75
hop_len = int(chunk_len * (1 - overlap))


def process_track(track_path: str) -> List[AudioChunk]:
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

    mixture_chunks = chunk_waveform(mixture_wav, chunk_len, hop_len)
    bass_chunks = chunk_waveform(bass_wav, chunk_len, hop_len)
    drums_chunks = chunk_waveform(drums_wav, chunk_len, hop_len)
    other_chunks = chunk_waveform(other_wav, chunk_len, hop_len)
    vocals_chunks = chunk_waveform(vocals_wav, chunk_len, hop_len)

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


if __name__ == "__main__":
    output_dir = os.path.join(data_root, "processed")
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    tracks = os.listdir(musdb_path)
    train_tracks, val_tracks = train_test_split(tracks, test_size=0.1, random_state=42)

    for track in tqdm(train_tracks, desc="Processing train tracks"):
        track_chunks = process_track(os.path.join(musdb_path, track))
        torch.save(track_chunks, os.path.join(train_dir, f"{track}.pt"))

    for track in tqdm(val_tracks, desc="Processing validation tracks"):
        track_chunks = process_track(os.path.join(musdb_path, track))
        torch.save(track_chunks, os.path.join(val_dir, f"{track}.pt"))
