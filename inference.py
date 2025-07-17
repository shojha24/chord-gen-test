import warnings
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from hmmlearn.hmm import CategoricalHMM
import types
from dataset import ChordMatchedDataset
from model import build_transformer
import requests # Add this import to fetch the JSON file
import pretty_midi
from midi2audio import FluidSynth
from pydub import AudioSegment
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
import librosa
import config
import subprocess

CHORD_TO_NOTES = {
    "Cmaj": [60, 64, 67, 72],
    "Cmin": [60, 63, 67, 72],
    
    "C#maj": [61, 65, 68, 73],
    "C#min": [61, 64, 68, 73],
    
    "Dmaj": [62, 66, 69, 74],
    "Dmin": [62, 65, 69, 74],
    
    "D#maj": [63, 67, 70, 75],
    "D#min": [63, 66, 70, 75],
    
    "Emaj": [64, 68, 71, 76],
    "Emin": [64, 67, 71, 76],
    
    "Fmaj": [65, 69, 72, 77],
    "Fmin": [65, 68, 72, 77],
    
    "F#maj": [66, 70, 73, 78],
    "F#min": [66, 69, 73, 78],
    
    "Gmaj": [67, 71, 74, 79],
    "Gmin": [67, 70, 74, 79],
    
    "G#maj": [68, 72, 75, 80],
    "G#min": [68, 71, 75, 80],
    
    "Amaj": [69, 73, 76, 81],
    "Amin": [69, 72, 76, 81],
    
    "A#maj": [70, 74, 77, 82],
    "A#min": [70, 73, 77, 82],
    
    "Bmaj": [71, 75, 78, 83],
    "Bmin": [71, 74, 78, 83]
}

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


def get_hmm_params_from_json(num_classes, chord_encodings, self_transition_prob=0.99):
    """
    Loads chord-to-chord transition data and adapts it for a frame-to-frame HMM
    by injecting a high self-transition probability.
    """
    print(f"Loading HMM parameters...")
    try:
        url = "https://raw.githubusercontent.com/schollz/chords/master/chordIndexInC.json"
        data = requests.get(url).json()

        # Create reverse mapping from chord name to class index
        inverted_encodings = {}
        for idx, name in chord_encodings.items():
            if name == 'N.C.': continue
            json_name = name.replace("maj", "").replace("min", "m").replace("#", "b")
            inverted_encodings[json_name] = idx
        
        # Initialize a chord-to-chord transition matrix with smoothing
        chord_transition_matrix = np.full((num_classes, num_classes), 1e-6)
        start_probs = np.full(num_classes, 1e-6)

        # Populate the matrix using first-order transitions from the JSON
        for from_chord_json, transitions in data.items():
            if " " in from_chord_json or from_chord_json not in inverted_encodings:
                continue
            
            from_idx = inverted_encodings[from_chord_json]
            start_probs[from_idx] += sum(transitions.values())
            
            for to_chord_json, prob in transitions.items():
                if to_chord_json in inverted_encodings:
                    to_idx = inverted_encodings[to_chord_json]
                    chord_transition_matrix[from_idx, to_idx] += prob

        # Normalize the chord-to-chord matrix
        row_sums = chord_transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # Avoid division by zero
        chord_transition_matrix /= row_sums
        
        # Normalize start probabilities
        start_probs /= start_probs.sum()

        # --- Adapt the matrix for frame-to-frame transitions ---
        print(f"Adapting for frame-based HMM with self-transition prob: {self_transition_prob}")
        
        # This is the probability of changing to *any* other chord
        change_prob = 1.0 - self_transition_prob
        
        # Create the final frame-based transition matrix
        frame_transition_matrix = np.zeros((num_classes, num_classes))

        for i in range(num_classes):
            # Get the relative probabilities of transitioning to *other* chords
            off_diagonal_probs = np.copy(chord_transition_matrix[i, :])
            off_diagonal_probs[i] = 0 # Ignore self-transition from the original matrix
            
            # Normalize the off-diagonal probabilities so they sum to 1
            off_diagonal_sum = off_diagonal_probs.sum()
            if off_diagonal_sum > 0:
                normalized_off_diagonal = off_diagonal_probs / off_diagonal_sum
            else:
                # If no transitions are specified, default to uniform change probability
                normalized_off_diagonal = np.full(num_classes, 1.0 / (num_classes - 1))
                normalized_off_diagonal[i] = 0

            # Distribute the small 'change_prob' according to these relative likelihoods
            frame_transition_matrix[i, :] = normalized_off_diagonal * change_prob
            
            # Set the high probability of staying on the same chord
            frame_transition_matrix[i, i] = self_transition_prob

        print("Successfully initialized frame-based HMM transition matrix.")
        return start_probs, frame_transition_matrix

    except Exception as e:
        print(f"Could not load HMM params: {e}. Falling back to uniform probabilities.")
        # Fallback remains the same
        transition_matrix = np.full((num_classes, num_classes), 1.0 / num_classes)
        start_probs = np.full(num_classes, 1.0 / num_classes)
        return start_probs, transition_matrix
    

# Replace the entire decode_chords_with_viterbi function with this version
def decode_chords_with_viterbi(logits, song_len_frames, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE):
    """
    Decodes the most likely sequence of chords using Viterbi decoding with CategoricalHMM.
    """
    num_classes = logits.shape[1]
    
    chord_encodings = {0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'Bmaj', 5: 'Bmin', 6: 'C#maj', 7: 'C#min', 
                       8: 'Cmaj', 9: 'Cmin', 10: 'D#maj', 11: 'D#min', 12: 'Dmaj', 13: 'Dmin', 14: 'Emaj', 15: 'Emin', 
                       16: 'F#maj', 17: 'F#min', 18: 'Fmaj', 19: 'Fmin', 20: 'G#maj', 21: 'G#min', 22: 'Gmaj', 
                       23: 'Gmin', 24: 'N.C.'} # Assuming 25th class is No Chord

    # 1. Get musically-informed HMM parameters
    start_probs, transition_matrix = get_hmm_params_from_json(num_classes, chord_encodings, self_transition_prob=0.99)

    # 2. Set up the HMM
    hmm_model = CategoricalHMM(n_components=num_classes)
    hmm_model.startprob_ = start_probs
    hmm_model.transmat_ = transition_matrix

    # 3. Monkey-patch the HMM to use our neural network's emission probabilities
    hmm_model.emissionprob_ = np.ones((num_classes, 1))
    def custom_log_likelihood_computer(self, log_probs_from_nn):
        return log_probs_from_nn
    hmm_model._compute_log_likelihood = types.MethodType(custom_log_likelihood_computer, hmm_model)

    # 4. Decode using the public API
    log_probs = torch.nn.functional.log_softmax(logits, dim=1).numpy()
    log_prob, predicted_states = hmm_model.decode(log_probs, algorithm="viterbi")
    
    print(f"HMM Log Probability: {log_prob}")

    # 5. Decode the predicted states into chord names
    chords_with_times = [] # ... (rest of the function is identical)
    if len(predicted_states) > 0:
        current_chord = chord_encodings.get(predicted_states[0], "N.C.")
        start_time = 0.0
        for i in range(1, len(predicted_states)):
            frame_time = i * hop_length / sample_rate
            if predicted_states[i] != predicted_states[i-1]:
                end_time = frame_time
                chords_with_times.append({'start': start_time, 'end': end_time, 'chord': current_chord})
                current_chord = chord_encodings.get(predicted_states[i], "N.C.")
                start_time = end_time
        
        end_time = song_len_frames * hop_length / sample_rate
        chords_with_times.append({'start': start_time, 'end': end_time, 'chord': current_chord})

    for item in chords_with_times:
        print(f"Time: {item['start']:.2f}s - {item['end']:.2f}s, Chord: {item['chord']}")
        
    return chords_with_times

def write_chords_to_midi(chords_with_times, output_path):
    """
    Writes the decoded chords to a MIDI file.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for chord_info in chords_with_times:
        start_time = chord_info['start']
        end_time = chord_info['end']
        chord_name = chord_info['chord']

        if chord_name == 'N.C.':
            continue  # Skip No Chord

        # Convert chord name to MIDI pitch numbers
        pitches = CHORD_TO_NOTES.get(chord_name, [])
        notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time) for pitch in pitches]
        
        instrument.notes.extend(notes)

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"MIDI file written to {output_path}")


def impose_midi_on_audio(midi_path, audio_path, sound_font_path, output_path="final_output.wav"):
    """
    Synthesizes a MIDI file to audio using a SoundFont and overlays it onto an existing audio file.
    """
    # Define a temporary path for the synthesized MIDI audio
    temp_midi_audio_path = "temp_midi_audio.wav"

    # 1. Synthesize the MIDI file to a WAV file
    print(f"Synthesizing {midi_path} to {temp_midi_audio_path}...")
    try:
        subprocess.run(["fluidsynth", "-ni", "-F", temp_midi_audio_path, "-r", str(config.SAMPLE_RATE), sound_font_path, midi_path], check=True)
        print(f"Successfully synthesized MIDI audio.")
    except Exception as e:
        print(f"Error during MIDI synthesis: {e}")
        print("Please ensure the SoundFont file exists at the specified path and that FluidSynth is installed correctly.")
        return

    # 2. Load both the original audio and the new MIDI audio
    try:
        original_audio = AudioSegment.from_mp3(audio_path)
        midi_audio = AudioSegment.from_wav(temp_midi_audio_path)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        print("Please ensure your original audio file is at the correct path and that you have ffmpeg installed for pydub to process MP3 files.")
        return

    # 3. Adjust volume and overlay the tracks
    # Increase the volume of the MIDI chords and decrease the original audio
    midi_audio += 8  # Boost MIDI audio volume
    original_audio -= 4 # Slightly reduce original audio volume

    # Ensure both files have the same duration by trimming the longer one
    min_length = min(len(midi_audio), len(original_audio))
    combined_audio = original_audio[:min_length].overlay(midi_audio[:min_length])

    # 4. Export the final mixed audio
    combined_audio.export(output_path, format="wav")
    print(f"Final mixed audio file saved as {output_path}")
    
    # 5. Clean up the temporary file
    if os.path.exists(temp_midi_audio_path):
        os.remove(temp_midi_audio_path)
        print(f"Removed temporary file: {temp_midi_audio_path}")


# Finally, update the main block
if __name__ == "__main__":
    # --- Configuration ---
    sound_font_path = "FluidR3_GM.sf2" 
    model_path = "music_models/epoch_10.pt"
    audio_path = "perfect.mp3"
    final_output_path = "final_output.wav"
    midi_output_path = "output.mid"

    # --- Execution ---
    if not os.path.exists(sound_font_path):
        print(f"ERROR: SoundFont file not found at '{sound_font_path}'. Please update the path.")
    else:
        model, device = get_model(model_path)

        # run_inference now returns LOGITS
        all_logits, song_len_frames = run_inference(model, audio_path, device)
        
        # Use the new Viterbi decoder
        decoded_chords = decode_chords_with_viterbi(all_logits, song_len_frames)

        # Write the decoded chords to a MIDI file
        write_chords_to_midi(decoded_chords, output_path=midi_output_path)

        # Impose the MIDI on the original audio
        impose_midi_on_audio(midi_output_path, audio_path, sound_font_path, output_path=final_output_path)