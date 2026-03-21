"""
chord_utils.py
==============
Shared utilities for encoding/decoding chord labels between the factorized
6-head representation and standard Harte chord strings consumable by mir_eval.

Also owns the CRF Viterbi post-processing functions so that evaluation.py
can import them without creating a circular dependency with train_unified.py.

Vocabulary layout (mirrors preprocessing.py in both branches):
  Head 0 – Root/Triad : 0 = N, 1..85 = root_pc * 7 + triad_idx  (7 triads)
  Head 1 – Bass       : 0 = N, 1..12 = bass_pc + 1
  Head 2 – 7th        : {N, 7, b7, bb7}
  Head 3 – 9th        : {N, 9, #9, b9}
  Head 4 – 11th       : {N, 11, #11}
  Head 5 – 13th       : {N, 13, b13}
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Vocabulary tables (must exactly match preprocessing.py)
# ---------------------------------------------------------------------------

ROOT_TRIAD_VOCAB: List[str] = ["5", "maj", "min", "sus4", "sus2", "dim", "aug"]
SEVENTH_VOCAB:    List[str] = ["N", "7", "b7", "bb7"]
NINTH_VOCAB:      List[str] = ["N", "9", "#9", "b9"]
ELEVENTH_VOCAB:   List[str] = ["N", "11", "#11"]
THIRTEENTH_VOCAB: List[str] = ["N", "13", "b13"]

PC_TO_NOTE: List[str] = ["C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B"]

N_TRIADS = len(ROOT_TRIAD_VOCAB)  # 7


# ---------------------------------------------------------------------------
# Single-frame decoder: 6 integer indices → Harte chord string
# ---------------------------------------------------------------------------

def decode_frame(
    root_triad_idx: int,
    bass_idx:       int,
    seventh_idx:    int,
    ninth_idx:      int,
    eleventh_idx:   int,
    thirteenth_idx: int,
) -> str:
    """
    Convert one frame's worth of 6-head integer labels into a Harte-style
    chord string that mir_eval can parse.

    Returns "N" when root_triad_idx == 0 (No-Chord).
    """
    if root_triad_idx == 0:
        return "N"

    # ---- root & triad -------------------------------------------------------
    val       = root_triad_idx - 1
    root_pc   = val // N_TRIADS
    triad_idx = val %  N_TRIADS
    root_str  = PC_TO_NOTE[root_pc % 12]
    triad_str = ROOT_TRIAD_VOCAB[triad_idx]

    # ---- bass ---------------------------------------------------------------
    bass_pc  = bass_idx - 1 if bass_idx > 0 else root_pc
    bass_str = PC_TO_NOTE[bass_pc % 12]

    # ---- extensions ---------------------------------------------------------
    desc = _triad_to_harte_base(triad_str)

    seventh    = SEVENTH_VOCAB[seventh_idx]
    ninth      = NINTH_VOCAB[ninth_idx]
    eleventh   = ELEVENTH_VOCAB[eleventh_idx]
    thirteenth = THIRTEENTH_VOCAB[thirteenth_idx]

    added: List[str] = []

    if seventh == "7":
        desc = _upgrade_to_maj7(desc)
    elif seventh == "b7":
        desc = _upgrade_to_dom7(desc)
    elif seventh == "bb7":
        desc = _upgrade_to_dim7(desc)

    if ninth      != "N": added.append(ninth)
    if eleventh   != "N": added.append(eleventh)
    if thirteenth != "N": added.append(thirteenth)

    chord = f"{root_str}:{desc}"
    if added:
        chord += f"(*{','.join(added)})" if desc.startswith("(") else f"({','.join(added)})"

    # ---- slash chord --------------------------------------------------------
    if bass_pc != root_pc:
        chord += f"/{bass_str}"

    return chord


# ---------------------------------------------------------------------------
# Batch decoder: arrays of integer predictions → list[str]
# ---------------------------------------------------------------------------

def decode_sequence(
    root_triad: np.ndarray,
    bass:       np.ndarray,
    seventh:    np.ndarray,
    ninth:      np.ndarray,
    eleventh:   np.ndarray,
    thirteenth: np.ndarray,
) -> List[str]:
    """Decode an entire sequence of frame predictions to chord strings."""
    T = len(root_triad)
    return [
        decode_frame(
            int(root_triad[t]),
            int(bass[t]),
            int(seventh[t]),
            int(ninth[t]),
            int(eleventh[t]),
            int(thirteenth[t]),
        )
        for t in range(T)
    ]


# ---------------------------------------------------------------------------
# Segment builder: frame labels → (intervals, chord_labels) for mir_eval
# ---------------------------------------------------------------------------

def frames_to_mir_eval_format(
    chord_strings: List[str],
    hop_length:    int = 512,
    sample_rate:   int = 22050,
) -> Tuple[np.ndarray, List[str]]:
    """
    Collapse a per-frame list of chord strings into contiguous segments
    and return (intervals, labels) in the format expected by mir_eval.

    intervals : np.ndarray of shape (N, 2) with [start_time, end_time] in seconds
    labels    : list of N chord strings
    """
    frame_duration = hop_length / sample_rate

    if not chord_strings:
        return np.zeros((0, 2), dtype=float), []

    intervals: List[List[float]] = []
    labels:    List[str]         = []

    current_label = chord_strings[0]
    seg_start     = 0.0

    for i in range(1, len(chord_strings)):
        if chord_strings[i] != current_label:
            seg_end = i * frame_duration
            intervals.append([seg_start, seg_end])
            labels.append(current_label)
            current_label = chord_strings[i]
            seg_start     = seg_end

    seg_end = len(chord_strings) * frame_duration
    intervals.append([seg_start, seg_end])
    labels.append(current_label)

    return np.array(intervals, dtype=float), labels


# ---------------------------------------------------------------------------
# CRF Viterbi post-processing
#
# Defined here — not in train_unified.py — so that evaluation.py can import
# them without creating a circular dependency.
# ---------------------------------------------------------------------------

def build_crf_transition_matrix(num_classes: int, penalty: float) -> torch.Tensor:
    """
    Fixed transition log-probability matrix.
    log_trans[i, j] = 0 if i == j (stay in same chord) else -penalty (change).
    """
    trans = torch.full((num_classes, num_classes), -penalty, dtype=torch.float32)
    trans.fill_diagonal_(0.0)
    return trans


def viterbi_decode_crf(logits: torch.Tensor, penalty: float = 2.0) -> torch.Tensor:
    """
    Vectorised Viterbi decoding over a batch of logit sequences.

    Parameters
    ----------
    logits  : (batch, seq_len, num_classes)
    penalty : transition cost for changing chord label between frames

    Returns
    -------
    decoded : (batch, seq_len)  integer class indices
    """
    batch, seq_len, num_classes = logits.shape
    device    = logits.device
    log_probs = torch.log_softmax(logits, dim=-1)
    trans_log = build_crf_transition_matrix(num_classes, penalty).to(device)

    score        = log_probs[:, 0, :]
    backpointers = []

    for t in range(1, seq_len):
        next_score          = score.unsqueeze(2) + trans_log.unsqueeze(0)
        best_score, best_bp = torch.max(next_score, dim=1)
        score               = best_score + log_probs[:, t, :]
        backpointers.append(best_bp)

    decoded    = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    last_state = torch.argmax(score, dim=1)
    decoded[:, seq_len - 1] = last_state

    for t in range(seq_len - 2, -1, -1):
        last_state    = backpointers[t].gather(1, last_state.unsqueeze(1)).squeeze(1)
        decoded[:, t] = last_state

    return decoded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _triad_to_harte_base(triad: str) -> str:
    mapping = {
        "maj":  "maj",
        "min":  "min",
        "dim":  "dim",
        "aug":  "aug",
        "sus4": "sus4",
        "sus2": "sus2",
        "5":    "5",
    }
    return mapping.get(triad, "maj")


def _upgrade_to_maj7(desc: str) -> str:
    _map = {
        "maj":  "maj7",
        "min":  "minmaj7",
        "dim":  "dimmaj7",
        "aug":  "augmaj7",
        "sus4": "sus4(7)",
        "sus2": "sus2(7)",
        "5":    "5",
    }
    return _map.get(desc, desc + "maj7")


def _upgrade_to_dom7(desc: str) -> str:
    _map = {
        "maj":  "7",
        "min":  "min7",
        "dim":  "hdim7",
        "aug":  "aug7",
        "sus4": "sus4(b7)",
        "sus2": "sus2(b7)",
        "5":    "5",
    }
    return _map.get(desc, desc + "7")


def _upgrade_to_dim7(desc: str) -> str:
    _map = {"dim": "dim7"}
    return _map.get(desc, desc + "(bb7)")


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        (0,  0, 0, 0, 0, 0,  "N"),
        (2,  0, 0, 0, 0, 0,  "C:maj"),
        (2,  0, 2, 0, 0, 0,  "C:7"),
        (3,  0, 2, 0, 0, 0,  "C:min7"),
        (51, 0, 0, 0, 0, 0,  "G:maj"),
    ]

    print("chord_utils sanity check:")
    for rt, b, s, n, e, t, expected in examples:
        result = decode_frame(rt, b, s, n, e, t)
        status = "✓" if expected in result or result == expected else "?"
        print(f"  {status}  expected≈{expected:15s}  got={result}")

    print("\nviterbi_decode_crf smoke test:")
    dummy = torch.randn(2, 10, 5)
    out   = viterbi_decode_crf(dummy, penalty=2.0)
    print(f"  input shape: {list(dummy.shape)}  output shape: {list(out.shape)}  ✓")
