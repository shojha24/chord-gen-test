import torch
import librosa
import numpy as np

def preprocess_audio(audio_path, sr=22050, hop_length=512):
    """
    This function implements the first step of the ChordFormer pipeline:
    converting an audio file into a CQT spectrogram representation.
    
    The parameters are based on Section IV-A of the paper.
    """
    
    # 1. Load the audio file. The paper specifies a sample rate of 22,050 Hz.
    y, sr = librosa.load(audio_path, sr=sr)

    # 2. Compute the Constant-Q Transform (CQT).
    #    - hop_length=512: Determines the time resolution of the spectrogram.
    #    - fmin: The lowest frequency to analyze, set to the note C1.
    #    - n_bins=252: The total number of frequency bins.
    #    - bins_per_octave=36: Gives a resolution of 3 bins per semitone (12 semitones * 3).
    cqt_spectrogram = librosa.cqt(y=y, sr=sr,
                                  hop_length=hop_length,
                                  fmin=librosa.note_to_hz('C1'),
                                  n_bins=252,
                                  bins_per_octave=36)
    
    # 3. Convert the CQT's amplitude to the decibel (dB) scale.
    #    This helps balance the loud and soft parts of the audio and often
    #    improves model performance.
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spectrogram), ref=np.max)
    
    # 4. Convert the result to a PyTorch tensor and transpose it.
    #    The output shape becomes (time_frames, frequency_bins)
    #    to be fed into the model.
    return torch.tensor(cqt_db).T

# --- Example Usage ---
# First, you'll need an audio file. Let's create a dummy one for demonstration.
# In a real scenario, you'd use a .wav or .mp3 file.
import soundfile as sf

sr = 22050
hop_length = 512
duration = 5 # seconds
frequency = 440 # A4 note
t = np.linspace(0., duration, int(sr * duration))
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)
sf.write('dummy_audio.wav', data.astype(np.int16), sr)


# Now, let's process it
cqt_tensor = preprocess_audio('dummy_audio.wav', sr=sr, hop_length=hop_length)

print(f"Shape of the CQT tensor: {cqt_tensor.shape}")
print(f"Data type: {cqt_tensor.dtype}")
