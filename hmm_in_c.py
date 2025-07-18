# FILE: hmm_in_c.py

import numpy as np
import requests
import os
import config
import json

def get_hmm_params(num_classes, chord_encodings):
    """
    Loads and processes the raw CHORD-TO-CHORD transition data from the source.
    This function does NOT perform frame adaptation.
    """
    print("Loading raw chord-to-chord HMM parameters...")
    try:
        url = "https://raw.githubusercontent.com/schollz/chords/master/chordIndexInC.json"
        data = requests.get(url).json()

        inverted_encodings = {name.replace("maj", "").replace("min", "m"): idx for idx, name in chord_encodings.items() if name != 'N.C.'}
        
        chord_transition_matrix = np.full((num_classes, num_classes), 1e-6) # Smoothing
        start_probs = np.full((num_classes,), 1e-6)

        for from_chord, transitions in data.items():
            if from_chord in inverted_encodings:
                from_idx = inverted_encodings[from_chord]
                start_probs[from_idx] += sum(transitions.values())
                for to_chord, prob in transitions.items():
                    if to_chord in inverted_encodings:
                        chord_transition_matrix[from_idx, inverted_encodings[to_chord]] += prob

        # Normalize probabilities
        start_probs /= start_probs.sum()
        chord_transition_matrix /= (chord_transition_matrix.sum(axis=1, keepdims=True) + 1e-9)

        print("Successfully created raw chord-to-chord HMM parameters.")
        # --- FIX: Return the matrix you actually calculated ---
        return start_probs, chord_transition_matrix

    except Exception as e:
        print(f"Could not load HMM params: {e}. Falling back to uniform probabilities.")
        return np.full(num_classes, 1.0/num_classes), np.full((num_classes, num_classes), 1.0/num_classes)

def save_hmm_params_to_json(start_probs, transition_matrix, filename='hmm_params.json'):
    # This function is correct.
    params = {'start_probs': start_probs.tolist(), 'transition_matrix': transition_matrix.tolist()}
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"HMM parameters saved to {filename}.")

def load_hmm_params_from_json(filename='hmm_params.json'):
    # This function is correct.
    with open(filename, 'r') as f:
        params = json.load(f)
    return np.array(params['start_probs']), np.array(params['transition_matrix'])

def init_hmm(num_classes, chord_encodings, filename='hmm_params.json'):
    # This function is correct.
    if os.path.exists(filename):
        print(f"Loading cached HMM parameters from {filename}...")
        start_probs, transition_matrix = load_hmm_params_from_json(filename)
    else:
       start_probs, transition_matrix = get_hmm_params(num_classes, chord_encodings)
       save_hmm_params_to_json(start_probs, transition_matrix)
    return start_probs, transition_matrix

if __name__ == "__main__":
    # When run, this script will create 'hmm_params.json' with the raw data.
    init_hmm(config.NUM_CLASSES, config.CHORD_ENCODINGS)
