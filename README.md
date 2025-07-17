tried an encoder model with (n_frames, d_model) chroma input -> (n_frames) chords output, attention seemed to not work so well especially on such a long sequence, and chords still fluctuated a lot. my solution was to split songs into 10 second chunks to increase effectiveness of attention and then further utilize viterbi decoding to smooth out the predictions. it works pretty well! next steps:

- test on more songs to see if it generalizes well
- create data visualizations for validation loss/accuracy, etc.
- set up a separate hmm.py file that requests the chains from https://raw.githubusercontent.com/schollz/chords/master/chordIndexInC.json once and saves them for future reference
- build a website/app for other people to use this!!!!


other cool things i thought about doing with the model architecture:

- try doing a full encoder decoder structure with (n_frames, d_model) input and an output with the onset timestamps and the chords to play, instead of mapping a chord for every single frame
- try a model pipeline where an lstm/transformer estimates the onset of a chord, then a cnn that predicts the chord itself using the frames corresponding to the start of the chord -> the end of the next chord
- learn how to use mamba models? apparently they are better for longer sequences
- keep training this model for a lot more epochs to see if it just reaches the global minimum very late