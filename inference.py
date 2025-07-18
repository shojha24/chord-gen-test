import warnings
import torch
from hmmlearn.hmm import CategoricalHMM
import types
from model import build_transformer
import requests # Add this import to fetch the JSON file
import pretty_midi
from pydub import AudioSegment
import os
import numpy as np
import librosa
import config
import subprocess
from hmm_in_c import init_hmm

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

    # Get the key of the song
    chroma_sum = np.sum(chroma, axis=1)

    # Major/minor templates
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    best_key = None

    # Compare the audio's chroma profile against all 24 key templates
    for i in range(12):
        # Major keys
        key_template = np.roll(major_template, i)
        score = np.corrcoef(chroma_sum, key_template)[0, 1]
        if score > best_score:
            best_score = score
            best_key = (f"{notes[i]}maj", i)

        # Minor keys
        key_template = np.roll(minor_template, i)
        score = np.corrcoef(chroma_sum, key_template)[0, 1]
        if score > best_score:
            best_score = score
            best_key = (f"{notes[i]}min", i)

    interval = best_key[1] # Semitones to shift from C
        
    return song_input, interval


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
    song_input, interval = process_audio(audio_path)
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
    
    return full_logits.cpu(), original_song_len_frames, interval


def get_hmm_params(num_classes, chord_encodings, interval, self_transition_prob):
    """
    This function performs the two steps needed for inference:
    1. Loads the base C-major chord-to-chord HMM parameters.
    2. Transposes them to the detected key.
    3. Adapts the transposed matrix for frame-to-frame decoding.
    """
    # Step 1: Load the raw, C-major, CHORD-TO-CHORD parameters
    start_probs_c, chord_trans_matrix_c = init_hmm(num_classes, chord_encodings)

    # Step 2: Transpose the raw matrix to the song's key
    # (This transposition logic is complex but correct)
    idx_to_pitch_class = {0: 10, 1: 10, 2: 9, 3: 9, 4: 11, 5: 11, 6: 1, 7: 1, 8: 0, 9: 0, 10: 3, 11: 3, 12: 2, 13: 2, 14: 4, 15: 4, 16: 6, 17: 6, 18: 5, 19: 5, 20: 8, 21: 8, 22: 7, 23: 7, 24: -1}
    pitch_quality_to_idx = {(v, k % 2): k for k, v in idx_to_pitch_class.items() if v != -1}
    
    transposition_map = np.arange(num_classes)
    for i in range(num_classes):
        if i in idx_to_pitch_class and idx_to_pitch_class[i] != -1:
            is_minor = i % 2
            transposed_pitch = (idx_to_pitch_class[i] + interval) % 12
            transposition_map[i] = pitch_quality_to_idx.get((transposed_pitch, is_minor), i)
    
    transposed_start_probs = np.zeros_like(start_probs_c)
    transposed_chord_matrix = np.zeros_like(chord_trans_matrix_c)
    for i in range(num_classes):
        transposed_start_probs[transposition_map[i]] = start_probs_c[i]
        for j in range(num_classes):
            transposed_chord_matrix[transposition_map[i], transposition_map[j]] = chord_trans_matrix_c[i, j]

    # Step 3: Adapt the now-transposed matrix for FRAME-TO-FRAME transitions
    change_prob = 1.0 - self_transition_prob
    final_frame_matrix = np.zeros_like(transposed_chord_matrix)
    for i in range(num_classes):
        off_diagonal = np.copy(transposed_chord_matrix[i, :])
        off_diagonal[i] = 0
        if off_diagonal.sum() > 0:
            final_frame_matrix[i, :] = (off_diagonal / off_diagonal.sum()) * change_prob
        final_frame_matrix[i, i] = self_transition_prob

    return transposed_start_probs, final_frame_matrix


def decode_chords_with_viterbi(logits, song_len_frames, interval, self_transition_prob, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE):
    """
    Decodes the most likely sequence of chords using Viterbi decoding with CategoricalHMM.
    """
    num_classes = logits.shape[1]

    # 1. Get musically-informed HMM parameters
    start_probs, transition_matrix = get_hmm_params(num_classes, config.CHORD_ENCODINGS, interval, self_transition_prob)

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
        current_chord = config.CHORD_ENCODINGS.get(predicted_states[0], "N.C.")
        start_time = 0.0
        for i in range(1, len(predicted_states)):
            frame_time = i * hop_length / sample_rate
            if predicted_states[i] != predicted_states[i-1]:
                end_time = frame_time
                chords_with_times.append({'start': start_time, 'end': end_time, 'chord': current_chord})
                current_chord = config.CHORD_ENCODINGS.get(predicted_states[i], "N.C.")
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
        pitches = config.CHORD_TO_NOTES.get(chord_name, [])
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
    original_audio -= 12 # Slightly reduce original audio volume

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
    folder_path = "test_songs"
    song_name = "perfect"
    audio_path = f"{folder_path}/{song_name}.mp3"
    final_output_path = f"{folder_path}/final_{song_name}.wav"
    midi_output_path = f"{folder_path}/{song_name}.mid"
    self_transition_prob = 0.98  # Adjust this as needed

    # --- Execution ---
    if not os.path.exists(sound_font_path):
        print(f"ERROR: SoundFont file not found at '{sound_font_path}'. Please update the path.")
    else:
        model, device = get_model(model_path)

        # run_inference now returns LOGITS
        all_logits, song_len_frames, interval = run_inference(model, audio_path, device)

        # Use the new Viterbi decoder
        decoded_chords = decode_chords_with_viterbi(all_logits, song_len_frames, interval, self_transition_prob)

        # Write the decoded chords to a MIDI file
        write_chords_to_midi(decoded_chords, output_path=midi_output_path)

        # Impose the MIDI on the original audio
        impose_midi_on_audio(midi_output_path, audio_path, sound_font_path, output_path=final_output_path)