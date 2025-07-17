import warnings
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from hmmlearn.hmm import MultinomialHMM

from dataset import ChordMatchedDataset
from model import build_transformer

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path
import os

import numpy as np
import librosa

import config

def process_audio(audio_path, sample_rate=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH):
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
    model = build_transformer(src_seq_len=config.SEQ_LEN_FRAMES, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE, d_model=config.D_MODEL, num_classes=config.NUM_CLASSES, n_bins=config.N_BINS)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device

# In inference.py, replace sliding_window_and_padding
def create_inference_segments(song_input, seq_len):
    """
    Takes a full song input and splits it into segments of seq_len,
    padding the last segment if necessary.
    Returns a list of segments and their corresponding padding masks.
    """
    song_len = song_input.shape[0]
    segments = []
    masks = []

    for i in range(0, song_len, seq_len):
        segment = song_input[i:i + seq_len]
        current_len = segment.shape[0]
        
        # Create the padding mask (True for padded positions)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        if current_len < seq_len:
            pad_len = seq_len - current_len
            padding = np.zeros((pad_len, song_input.shape[1]))
            segment = np.vstack((segment, padding))
            mask[current_len:] = True

        segments.append(segment)
        masks.append(mask)

    return torch.tensor(np.array(segments), dtype=torch.float32), torch.stack(masks)


# In inference.py, replace run_inference
def run_inference(model, audio_path, device):
    # Process the audio file
    song_input = process_audio(audio_path)
    original_song_len_frames = song_input.shape[0]
    print(f"Processed input shape: {song_input.shape}")

    # Use the new, clean segmentation function
    input_segments, padding_masks = create_inference_segments(song_input, config.SEQ_LEN_FRAMES)
    input_segments = input_segments.to(device)
    padding_masks = padding_masks.to(device)
    
    print(f"Input segments shape: {input_segments.shape}")

    all_logits = []
    with torch.no_grad():
        for i in range(input_segments.size(0)):
            segment = input_segments[i].unsqueeze(0)  # Add batch dim
            mask = padding_masks[i].unsqueeze(0)      # Add batch dim

            # Pass the segment AND the mask to the model
            encoder_output = model.encode(segment, src_key_padding_mask=mask)
            logits = model.project(encoder_output)
            all_logits.append(logits)

    # Concatenate logits from all segments and trim to original song length
    full_logits = torch.cat(all_logits, dim=1).squeeze(0) # (total_frames, num_classes)
    full_logits = full_logits[:original_song_len_frames, :]
    
    return full_logits.cpu(), original_song_len_frames


def decode_chords_with_viterbi(logits, song_len_frames, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE):
    """
    Decodes the most likely sequence of chords using Viterbi decoding.
    """
    # 1. Emission Probabilities (from your model's output)
    # Convert logits to probabilities using softmax
    emission_probs = torch.nn.functional.softmax(logits, dim=1).numpy()

    # 2. Transition Probabilities (can be learned or defined by theory)
    # For now, let's create a simple one: high prob of staying on the same chord,
    # and a small, uniform prob of transitioning to any other chord.
    num_classes = logits.shape[1]
    transition_matrix = np.full((num_classes, num_classes), 0.01)
    np.fill_diagonal(transition_matrix, 1 - (0.01 * (num_classes - 1)))
    
    # Ensure rows sum to 1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    # 3. Initial Probabilities (how likely is a song to start with each chord)
    # Let's assume uniform for simplicity.
    start_probs = np.full(num_classes, 1.0 / num_classes)

    # 4. Set up and run the HMM
    hmm_model = MultinomialHMM(n_components=num_classes, n_iter=10)
    hmm_model.startprob_ = start_probs
    hmm_model.transmat_ = transition_matrix
    hmm_model.emissionprob_ = emission_probs.T # Note: hmmlearn expects (n_components, n_features), so we transpose

    # HMM expects integer observations, not probabilities. A common workaround is to
    # use the argmax as the observations for the Viterbi path finding.
    observations = np.argmax(emission_probs, axis=1).reshape(-1, 1)
    
    log_prob, predicted_states = hmm_model.decode(observations, algorithm="viterbi")
    
    print(f"HMM Log Probability: {log_prob}")

    # 5. Decode the predicted states into chord names
    chord_encodings = {0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'Bmaj', 5: 'Bmin', 6: 'C#maj', 7: 'C#min', 
                        8: 'Cmaj', 9: 'Cmin', 10: 'D#maj', 11: 'D#min', 12: 'Dmaj', 13: 'Dmin', 14: 'Emaj', 15: 'Emin', 
                        16: 'F#maj', 17: 'F#min', 18: 'Fmaj', 19: 'Fmin', 20: 'G#maj', 21: 'G#min', 22: 'Gmaj', 
                        23: 'Gmin'} # (your encodings)
    
    # Group consecutive identical chords
    chords_with_times = []
    if len(predicted_states) > 0:
        current_chord = chord_encodings[predicted_states[0]]
        start_time = 0.0
        for i in range(1, len(predicted_states)):
            frame_time = i * hop_length / sample_rate
            if predicted_states[i] != predicted_states[i-1]:
                end_time = frame_time
                chords_with_times.append({'start': start_time, 'end': end_time, 'chord': current_chord})
                current_chord = chord_encodings[predicted_states[i]]
                start_time = end_time
        
        # Add the final chord
        end_time = song_len_frames * hop_length / sample_rate
        chords_with_times.append({'start': start_time, 'end': end_time, 'chord': current_chord})

    for item in chords_with_times:
        print(f"Time: {item['start']:.2f}s - {item['end']:.2f}s, Chord: {item['chord']}")
        
    return chords_with_times

# Finally, update the main block
if __name__ == "__main__":
    model_path = "music_models/epoch_3.pt"
    audio_path = "perfect.mp3"

    model, device = get_model(model_path)

    # run_inference now returns LOGITS
    all_logits, song_len_frames = run_inference(model, audio_path, device)
    
    # Use the new Viterbi decoder
    decoded_chords = decode_chords_with_viterbi(all_logits, song_len_frames)
