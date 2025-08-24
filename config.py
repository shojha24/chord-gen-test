# config.py
SAMPLE_RATE = 22050
HOP_LENGTH = 2048
N_FFT = 2048 # Should be >= HOP_LENGTH
N_MELS = 64
N_BINS = 13 # 12 chroma + 1 onset
N_FILES = 1000
N_MEL_FILES = 3000
D_MODEL = 128
NUM_CLASSES = 25
SEQ_LEN_SECONDS = 10
# Calculated, not hardcoded
SEQ_LEN_FRAMES = int(SEQ_LEN_SECONDS * SAMPLE_RATE / HOP_LENGTH)
LEARNING_RATE = 5e-4
LEARNING_RATE_2 = 1e-5
NUM_EPOCHS = 20
BATCH_SIZE = 32
CHORD_ENCODINGS = {
    0: 'A#maj', 1: 'A#min', 2: 'Amaj', 3: 'Amin', 4: 'Bmaj', 5: 'Bmin', 6: 'C#maj', 7: 'C#min', 8: 'Cmaj', 
    9: 'Cmin', 10: 'D#maj', 11: 'D#min', 12: 'Dmaj', 13: 'Dmin', 14: 'Emaj', 15: 'Emin', 16: 'F#maj', 
    17: 'F#min', 18: 'Fmaj', 19: 'Fmin', 20: 'G#maj', 21: 'G#min', 22: 'Gmaj', 23: 'Gmin', 24: 'N.C.'
}
CHORD_TO_NOTES = {
    "Cmaj": [60, 64, 67, 72], "Cmin": [60, 63, 67, 72], "C#maj": [61, 65, 68, 73], "C#min": [61, 64, 68, 73],
    "Dmaj": [62, 66, 69, 74], "Dmin": [62, 65, 69, 74], "D#maj": [63, 67, 70, 75], "D#min": [63, 66, 70, 75],
    "Emaj": [64, 68, 71, 76], "Emin": [64, 67, 71, 76], "Fmaj": [65, 69, 72, 77], "Fmin": [65, 68, 72, 77],
    "F#maj": [66, 70, 73, 78], "F#min": [66, 69, 73, 78], "Gmaj": [67, 71, 74, 79], "Gmin": [67, 70, 74, 79],
    "G#maj": [68, 72, 75, 80], "G#min": [68, 71, 75, 80], "Amaj": [69, 73, 76, 81], "Amin": [69, 72, 76, 81],
    "A#maj": [70, 74, 77, 82], "A#min": [70, 73, 77, 82], "Bmaj": [71, 75, 78, 83], "Bmin": [71, 74, 78, 83]
}

