import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import librosa
import pandas as pd
import h5py
import os

class ChordMatchedDataset(Dataset):
    def __init__(self, mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft, cache_path="dataset.hdf5"):
        super().__init__()
        self.seq_len = 0
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.mix_path = mix_path
        self.annotation_data = annotation_path
        self.beatinfo_headers = ['Start time in seconds', 'Bar count', 'Quarter count', 'Chord name']
        self.cache_path = cache_path

        self.chord_encodings = {0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'Bmaj', 5: 'Bmin', 6: 'C#maj', 7: 'C#min', 
                        8: 'Cmaj', 9: 'Cmin', 10: 'D#maj', 11: 'D#min', 12: 'Dmaj', 13: 'Dmin', 14: 'Emaj', 15: 'Emin', 
                        16: 'F#maj', 17: 'F#min', 18: 'Fmaj', 19: 'Fmin', 20: 'G#maj', 21: 'G#min', 22: 'Gmaj', 
                        23: 'Gmin'}

        self.inverted_encodings = dict(zip(self.chord_encodings.values(), self.chord_encodings.keys()))

        if os.path.exists(self.cache_path):
            print(f"Loading dataset from {self.cache_path}...")
            self.load_dataset_hdf5(self.cache_path)
            print(f"Dataset loaded with {len(self.raw_X)} samples")
        else:
            # Preprocess all data during initialization
            print("Preprocessing dataset...")
            self.raw_X, self.y = self.preprocess_and_view()
            print(f"Dataset loaded with {len(self.raw_X)} samples")
            self.save_dataset_hdf5(self.cache_path)

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        x = np.array(self.raw_X[idx])
        target = np.array(self.y[idx])
        
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            # Use zeros for padding instead of -1
            x = np.pad(x, ((0, pad_len), (0, 0)), constant_values=0)
            target = np.pad(target, (0, pad_len), constant_values=-1)  # Keep -1 for targets
        
        # Create padding mask
        is_padded = np.zeros(self.seq_len, dtype=bool)
        if len(self.raw_X[idx]) < self.seq_len:
            is_padded[len(self.raw_X[idx]):] = True
        
        return {
            "feature": torch.FloatTensor(x),
            "target": torch.LongTensor(target),
            "padding_mask": torch.BoolTensor(is_padded)
        }

        
    def normalize_audio_features(self, data):
        """
        Normalize audio features: mel bins, chroma bins, and onset strength
        Input shape: (time_steps, 77) - 64 mel + 12 chroma + 1 onset
        """
        normalized_data = np.copy(data)
        
        # Mel bins (first 128 features): -80 to 0 → 0 to 1
        normalized_data[:, :64] = (data[:, :64] + 80) / 80
        
        # Chroma bins (next 12 features): already 0 to 1, keep as is
        # normalized_data[:, 128:140] = data[:, 128:140]
        
        # Onset strength (last feature): log normalization
        onset_data = data[:, 76]
        onset_log = np.log1p(onset_data)
        normalized_data[:, 76] = onset_log / (np.max(onset_log) + 1e-8)  # avoid division by zero
        
        return normalized_data

    def preprocess_and_view(self):
        raw_X = []
        y = []
        
        # Iterate through all files in the directory
        for i in range(500):
            file_num = i + 1
            n_frames = 0

            beatinfo_path = os.path.join(self.annotation_data, f"{file_num :04d}_beatinfo.arff")
            audio_path = os.path.join(self.mix_path, f"{file_num :04d}_mix.flac")

            """This is for targets."""
            beatinfo_df = pd.read_csv(beatinfo_path, comment="@", header=None)
            beatinfo_df.columns = self.beatinfo_headers

            usable = True

            for j in range(beatinfo_df.index.size):
                beatinfo_df.iat[j, 3] = beatinfo_df.iat[j, 3].replace("'", "")
                if beatinfo_df.iat[j, 3] == "BASS_NOTE_EXCEPTION" or beatinfo_df.iat[j, 3] == "N.C.":
                    if j > 0:
                        beatinfo_df.iat[j, 3] = beatinfo_df.iat[j-1, 3]
                    else:
                        usable = False
                        break
                else:
                    beatinfo_df.iat[j, 3] = self.inverted_encodings[beatinfo_df.iat[j, 3]]

            if not usable:
                print(f"Skipping file {file_num} due to unusable annotations.")
                continue

            """This is for features."""
            try:
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # Fixed syntax
                y_harm = librosa.effects.harmonic(y=audio, margin=8)
                mel_spec = librosa.feature.melspectrogram(
                        y=audio, 
                        sr=self.sample_rate,
                        n_mels=self.n_mels,
                        hop_length=self.hop_length,
                        n_fft=self.n_fft
                )
                scaled_mel = librosa.power_to_db(mel_spec, ref=np.max)
                transposed_mel = scaled_mel.T
                chroma = librosa.feature.chroma_cqt(y=y_harm, sr=self.sample_rate)
                transposed_chroma = chroma.T
                onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
                onset_2d = [[elem] for elem in onset_env]
                song_input = np.concatenate((transposed_mel, transposed_chroma, onset_2d), axis=1)
                
                normalized_song_input = self.normalize_audio_features(song_input)
                
                raw_X.append(normalized_song_input.tolist())
                n_frames = len(normalized_song_input)
                self.seq_len = max(self.seq_len, n_frames)
                
            except Exception as e:
                print(e)

            # expand the chord list (n_beats) to match the length of the audio features (n_frames)
            coarse_times = beatinfo_df['Start time in seconds'].astype(float).to_numpy()
            fine_times = librosa.frames_to_time(np.arange(n_frames), sr=self.sample_rate, hop_length=self.hop_length)
            chords = beatinfo_df['Chord name'].astype(int).to_numpy()

            idx = np.searchsorted(coarse_times[1:], fine_times, side='right')
            expanded_chords = chords[idx]

            y.append(expanded_chords.tolist())

            print(f"done with song {file_num}")

        return raw_X, y
    
    def save_dataset_hdf5(self, filename):
        with h5py.File(filename, 'w') as f:
            f.attrs['seq_len'] = self.seq_len
            
            for i, (features, targets) in enumerate(zip(self.raw_X, self.y)):
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('features', data=np.array(features), compression='gzip')
                grp.create_dataset('targets', data=np.array(targets), compression='gzip')

    def load_dataset_hdf5(self, filename):
        self.raw_X = []
        self.y = []
        
        with h5py.File(filename, 'r') as f:
            self.seq_len = f.attrs['seq_len']
            
            for key in sorted(f.keys()):
                grp = f[key]
                self.raw_X.append(grp['features'][:].tolist())
                self.y.append(grp['targets'][:].tolist())

        
if __name__ == "__main__":
    mix_path = "dataset\\mixes"
    annotation_path = "dataset\\annotations"
    sample_rate = 11025
    hop_length = 512
    n_mels = 64
    n_fft = 2048

    dataset = ChordMatchedDataset(mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft)
    
    # Example usage
    for i in range(len(dataset)):
        example = dataset[i]
        print(f"Sample {i}: x shape: {example["feature"].shape}, target shape: {example["target"].shape}")