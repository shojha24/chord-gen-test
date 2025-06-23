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

def sliding_window_and_padding(song_input, max_seq_len, hop_length, sample_rate):
    """
    input dict format:
    {
        1: {
            "timeframe": [0, 1976 * 1024 / 11025]  # timeframes in seconds
            "input": np.array([[...], [...], ...])  # shape (1976, 13)
        },
        2: {
            "timeframe": [0.5 * 1976 * 1024 / 11025, 1.5 * 1976 * 1024 / 11025]
            "input": np.array([[...], [...], ...])  # shape (1976, 13)
        },
        ...
        n: {
            "timeframe": [n * 1976 * 1024 / 11025, end of song]
            "input": np.array([[...], [...], ...])  # shape (1976, 13), including padding
        }
    }
    """

    input_dict = {}
    song_len = song_input.shape[0]
    pad_added = 0

    if song_len > max_seq_len:
        # Use sliding window if it's longer than max_seq_len
        extra_frames = song_len % max_seq_len 
        print(extra_frames)
        if extra_frames > 0:
            pad_added = max_seq_len - extra_frames
            padding = np.zeros((pad_added, song_input.shape[1]))
            song_input = np.vstack((song_input, padding))  # Pad the input to make it divisible by max_seq_len
            print(f"Padding added: {pad_added} frames to make the input divisible by {max_seq_len}. New length: {song_input.shape[0]} frames.")

        num_windows = song_input.shape[0] // max_seq_len

        for i in range(num_windows):
            start = i * max_seq_len
            end = (i + 1) * max_seq_len

            input_dict[i + 1] = {
                "timeframe": [start * hop_length / sample_rate, (end * hop_length / sample_rate if i < num_windows - 1 else song_len * hop_length / sample_rate)],
                "input": song_input[start:end, :]
            }
    else:
        if song_len < max_seq_len:
            # Pad the input if it's shorter than max_seq_len
            padding = np.zeros((max_seq_len - song_input.shape[0], song_input.shape[1]))
            song_input = np.vstack((song_input, padding))
        input_dict[1] = {
            "timeframe": [0, song_len * hop_length / sample_rate],
            "input": song_input
        }
    
    return input_dict


def run_inference(model, audio_path, device):
    # Process the audio file
    song_input = process_audio(audio_path)
    print(f"Processed input shape: {song_input.shape}")

    input_segments = sliding_window_and_padding(song_input, max_seq_len=1976, hop_length=1024, sample_rate=11025)

    print(f"Input segments: {len(input_segments)} segments")

    predicted_classes = torch.tensor([], dtype=torch.long).to(device)

    for segment_id, segment in input_segments.items():
        print(f"Segment {segment_id} timeframe: {segment['timeframe']}, input shape: {segment['input'].shape}")
        seg_input = torch.tensor(segment['input'], dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        src_key_padding_mask = (seg_input.abs().sum(dim=-1) == 0)  # (batch_size, seq_len)

        # Run the model
        with torch.no_grad():
            encoder_output = model.encode(seg_input, src_key_padding_mask)
            logits = model.project(encoder_output)

            # Retrieve the predicted class with the highest probability and append to predicted classes tensor
            segment_predicted_classes = torch.argmax(logits, dim=-1)
            predicted_classes = torch.cat((predicted_classes, segment_predicted_classes), dim=1)

    return predicted_classes.cpu().numpy()[0], song_input.shape[0]  # Return predicted classes and song length in seconds

def decode_chords(predicted_classes, song_len, hop_length=1024, sample_rate=11025):
    """
    Decode the predicted classes into chord names.
    """
    chord_encodings = {0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'Bmaj', 5: 'Bmin', 6: 'C#maj', 7: 'C#min', 
                   8: 'Cmaj', 9: 'Cmin', 10: 'D#maj', 11: 'D#min', 12: 'Dmaj', 13: 'Dmin', 14: 'Emaj', 15: 'Emin', 
                   16: 'F#maj', 17: 'F#min', 18: 'Fmaj', 19: 'Fmin', 20: 'G#maj', 21: 'G#min', 22: 'Gmaj', 
                   23: 'Gmin', 24: 'N.C.'}
    chords = [chord_encodings[chord] for chord in predicted_classes]
    times = [str(item) for item in np.linspace(0, song_len * hop_length / sample_rate, len(chords)).tolist()]
    times_to_chords = dict(zip(times, chords))

    for i in range(1, len(chords)):
        if chords[i] == chords[i - 1]:
            times_to_chords.pop(times[i])
            i -= 1

    print(times_to_chords)
    return times_to_chords

if __name__ == "__main__":
    # Example usage
    model_path = "music_models/epoch_20.pt"  # Path to your saved model
    audio_path = "perfect.mp3"  # Path to your audio file

    # Load the model
    model, device = get_model(model_path)

    # Run inference
    predicted_classes, song_len = run_inference(model, audio_path, device)
    print(f"Predicted classes: {predicted_classes}")

    decoded_chords = decode_chords(predicted_classes, song_len=song_len, hop_length=1024, sample_rate=11025)
