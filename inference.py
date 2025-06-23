import warnings
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from dataset import ChordMatchedDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path
import os

import numpy as np
import librosa

def process_audio(audio_path, sample_rate=11025, hop_length=1024):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)
    y_harm = librosa.effects.hpss(y)[0]  # Harmonic component

    # Compute chroma
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)

    # Compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_2d = [[elem] for elem in onset_env]

    # Concatenate chroma and onset strength
    song_input = np.concatenate((chroma.T, onset_2d), axis=1)

    # Normalize chroma
    onset_data = song_input[:, 12]
    onset_log = np.log1p(onset_data)
    song_input[:, 12] = onset_log / (np.max(onset_log) + 1e-8)

    return song_input

def get_model(model_path):
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = build_transformer(src_seq_len=1976, hop_length=1024, sample_rate=11025, d_model=16, num_classes=25, n_bins=13)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device

"""def sliding_window(src_seq_len, max_seq_len):
    if src_seq_len > """


def run_inference(model, audio_path, device):
    # Process the audio file
    song_input = process_audio(audio_path)
    print(f"Processed input shape: {song_input.shape}")

    song_input = torch.tensor(song_input, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

    # Create a src_key_padding_mask
    src_key_padding_mask = (song_input.abs().sum(dim=-1) == 0)  # (batch_size, seq_len)

    # Run the model
    with torch.no_grad():
        encoder_output = model.encode(song_input, src_key_padding_mask)
        logits = model.project(encoder_output)

        # Retrieve the predicted class with the highest probability
        predicted_classes = torch.argmax(logits, dim=-1)

    return predicted_classes.cpu().numpy()

if __name__ == "__main__":
    # Example usage
    model_path = "music_models/epoch_20.pt"  # Path to your saved model
    audio_path = "perfect.mp3"  # Path to your audio file

    # Load the model
    model, device = get_model(model_path)

    # Run inference
    predicted_classes = run_inference(model, audio_path, device)
    print(f"Predicted classes: {predicted_classes}")
