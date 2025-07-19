tried an encoder-only model with (n_frames, d_model) chroma/onset strength input -> (n_frames) chords output, attention seemed to not work so well especially on such a long sequence, and chords still fluctuated a lot. my solution was to split songs into 10 second chunks to increase effectiveness of attention and then further utilize viterbi decoding to smooth out the predictions. it works pretty well! next steps:

- see future.md... :O
- test on more songs to see if it generalizes well
 - there is definitely work that needs to be done here. thinking of using some tracks from the same datasets the btc researchers used so that the model can generalize to real songs w/ varying levels of loudness from each instrument as opposed to the artificial tracks where all of the instruments are played at equal strength.
 - [Isophonics](http://isophonics.net/datasets), [UsPop2002](https://github.com/tmc323/Chord-Annotations)
- create data visualizations for validation loss/accuracy, etc.
- build a website/app for other people to use this!!!!
- find/generate datasets that match monophonic melodies to chord charts and/or use the method schollz used to calculate chord progression probabilities to build distributions across various genres


other cool things i thought about doing with the model architecture:

- right now, i'm using a first order markov model to determine what the most plausible next chord would be based on the previous one; the predictions end up being decent because the semantic data retrieved from the transformer via attention supplements the markov model in describing chord changes and relationships. it could be a good idea to try implementing a higher order markov model as well for smoothing; could utilize all of the data in [schollz's json](https://raw.githubusercontent.com/schollz/chords/refs/heads/master/chordIndexInC.json)
- try doing a full encoder decoder structure with (n_frames, d_model) input and an output with the onset timestamps and the chords to play, instead of mapping a chord for every single frame
- try a model pipeline where an lstm/transformer estimates the onset of a chord, then a cnn that predicts the chord itself using the frames corresponding to the start of the chord -> the end of the next chord
- learn how to use mamba models? apparently they are better for longer sequences, could potentially train on sequences longer than 10 seconds

cool stuff to read through: 
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
- [Bidirectional Transformer for Chord Recognition](https://arxiv.org/pdf/1907.02698), was able to use this paper as an architectural framework, was able to achieve similar accuracy levels just by using 12 chroma bins and Viterbi decoding instead of 128 spectrogram bins.
- [Chord Progressions from Melodies](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-023-00314-6), something I plan on implementing myself
- [The Probability of Every Chord Progression](https://schollz.com/tinker/chords/) is a cool read, and created [the chord probabilities i got here](https://github.com/schollz/common-chords)
