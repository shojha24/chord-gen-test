tried an encoder model with (n_frames, d_model) chroma input -> (n_frames) chords output, attention seems to not work so well especially on such a long sequence, and chords still fluctuate a lot. the options from here are:

- try doing a full encoder decoder structure with (n_frames, d_model) input and an output with the onset timestamps and the chords to play, instead of mapping a chord for every single frame
- try a model pipeline where an lstm/transformer estimates the onset of a chord, then a cnn that predicts the chord itself using the frames corresponding to the start of the chord -> the end of the next chord
- learn how to use mamba models? apparently they are better for longer sequences
- keep training this model for a lot more epochs to see if it just reaches the global minimum very late