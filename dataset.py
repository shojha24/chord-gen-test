import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import librosa
import pandas as pd
import h5py
import os
import multiprocessing as mp
from functools import partial
import concurrent.futures

class ChordMatchedDataset(Dataset):
    def __init__(self, mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft, n_files, cache_path="dataset.hdf5"):
        super().__init__()
        self.seq_len = 0
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_files = n_files
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
            self.raw_X, self.y = self.preprocess_and_view_parallel()
            print(f"Dataset loaded with {len(self.raw_X)} samples")
            self.save_dataset_hdf5(self.cache_path)

    def __len__(self):
        return len(self.raw_X)

    def __getitem__(self, idx):
        x = np.array(self.raw_X[idx])
        target = np.array(self.y[idx])

        if len(x) < self.seq_len:
            # dont return samples shorter than seq_len
            return None
        
        return {
            "feature": torch.FloatTensor(x),
            "target": torch.LongTensor(target),
        }


    def process_single_file(self, file_num, mix_path, annotation_data, sample_rate, hop_length, 
                        beatinfo_headers, inverted_encodings):
        """Process a single audio file and return features and targets"""
        
        beatinfo_path = os.path.join(annotation_data, f"{file_num:04d}_beatinfo.arff")
        audio_path = os.path.join(mix_path, f"{file_num:04d}_mix.flac")
        
        # Check if files exist
        if not (os.path.exists(beatinfo_path) and os.path.exists(audio_path)):
            return None, None
        
        try:
            # Process annotations
            beatinfo_df = pd.read_csv(beatinfo_path, comment="@", header=None)
            beatinfo_df.columns = beatinfo_headers
            
            # Clean chord names and encode
            for j in range(beatinfo_df.index.size):
                beatinfo_df.iat[j, 3] = beatinfo_df.iat[j, 3].replace("'", "")
                if beatinfo_df.iat[j, 3] in ["BASS_NOTE_EXCEPTION", "N.C."]:
                    if j > 0:
                        beatinfo_df.iat[j, 3] = 24
                    else:
                        return None, None  # Skip unusable file
                else:
                    beatinfo_df.iat[j, 3] = inverted_encodings[beatinfo_df.iat[j, 3]]
            
            # Process audio
            audio, _ = librosa.load(audio_path, sr=sample_rate)
            y_harm = librosa.effects.harmonic(y=audio, margin=8)
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sample_rate, hop_length=hop_length)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
            
            # Combine features
            song_input = np.concatenate((chroma.T, onset_env.reshape(-1, 1)), axis=1)
            
            # Normalize
            normalized_song_input = self.normalize_audio_features_static(song_input)
            
            # Create segments
            segment_length = int(10 * sample_rate / hop_length)
            segments = [normalized_song_input[i:i + segment_length] 
                    for i in range(0, len(normalized_song_input), segment_length)]
            
            # Expand chords to match audio frames
            coarse_times = beatinfo_df['Start time in seconds'].astype(float).to_numpy()
            fine_times = librosa.frames_to_time(np.arange(len(normalized_song_input)), 
                                            sr=sample_rate, hop_length=hop_length)
            chords = beatinfo_df['Chord name'].astype(int).to_numpy()
            
            idx = np.searchsorted(coarse_times[1:], fine_times, side='right')
            expanded_chords = chords[idx]
            
            chord_segments = [expanded_chords[i:i + segment_length] 
                            for i in range(0, len(expanded_chords), segment_length)]
            
            print(f"Processed file {file_num}: {len(segments)} segments, {len(chord_segments)} chord segments")
            
            return segments, chord_segments
            
        except Exception as e:
            print(f"Error processing file {file_num}: {e}")
            return None, None

    def normalize_audio_features_static(self, data):
        """Static version of normalize function for multiprocessing"""
        normalized_data = np.copy(data)
        onset_data = data[:, 12]
        onset_log = np.log1p(onset_data)
        normalized_data[:, 12] = onset_log / (np.max(onset_log) + 1e-8)
        return normalized_data

    def preprocess_and_view_parallel(self):
        """Parallel version of preprocessing"""
        raw_X = []
        y = []
        
        # Create partial function with fixed arguments
        process_func = partial(
            self.process_single_file,
            mix_path=self.mix_path,
            annotation_data=self.annotation_data,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            beatinfo_headers=self.beatinfo_headers,
            inverted_encodings=self.inverted_encodings
        )
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            file_numbers = range(1, self.n_files + 1)
            results = list(executor.map(process_func, file_numbers))
        
        # Collect results
        for i, (segments, chord_segments) in enumerate(results):
            if segments is not None and chord_segments is not None:
                raw_X.extend(segments)
                y.extend(chord_segments)

        self.seq_len = max(len(segment) for segment in raw_X)
        
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
                if (len(grp['features'][:].tolist()) < self.seq_len):
                    continue
                self.raw_X.append(grp['features'][:].tolist())
                self.y.append(grp['targets'][:].tolist())

        
if __name__ == "__main__":
    mix_path = "dataset\\mixes"
    annotation_path = "dataset\\annotations"
    sample_rate = 22050
    hop_length = 2048
    n_mels = 64
    n_fft = 2048
    n_files = 1000

    dataset = ChordMatchedDataset(mix_path, annotation_path, sample_rate, hop_length, n_mels, n_fft, n_files)
    
    # Example usage
    for i in range(len(dataset)):
        example = dataset[i]
        print(f"Sample {i}: x shape: {example["feature"].shape}, target shape: {example["target"].shape}")