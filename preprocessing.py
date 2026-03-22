"""
preprocessing.py
================
Handles all data preprocessing for the Bello Chord Dataset:
  - Parsing Harte chord labels into structured multi-head targets
  - Computing CQT features from raw audio
  - Aligning chord labels to CQT frames
  - Segmenting songs into fixed-length chunks with appropriate padding
  - Two-stage caching so each song's CQT is computed exactly once across
    all folds (see precompute_song_cache)
  - On-the-fly pitch-shift augmentation during training

Two-stage cache design
----------------------
Stage 1 — precompute_song_cache(pairs, cfg)
    Called once before any fold's DataLoaders are built.  Writes one .pt
    file per song under:
        <cfg.cache_dir>/songs/<cfg_hash>/<stem>.pt
    Each file contains the pre-chunked features and targets for that song.
    Skips songs whose cache file already exists (idempotent).

Stage 2 — BelloChordFormerDataset(file_pairs, cfg, ...)
    Loads the pre-computed per-song .pt files for the songs in this split
    and concatenates their chunks into self.features / self.targets.
    No CQT computation happens here at all.

This eliminates the old split-level cache (bello_{digest}_{split_name}.pt)
which caused each song's CQT to be recomputed up to 3× across 5 folds
(once per split role: train, val, test).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import hashlib
import random
import re

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

ROOT_TRIAD_VOCAB = ["5", "maj", "min", "sus4", "sus2", "dim", "aug"]
SEVENTH_VOCAB    = ["N", "7", "b7", "bb7"]
NINTH_VOCAB      = ["N", "9", "#9", "b9"]
ELEVENTH_VOCAB   = ["N", "11", "#11"]
THIRTEENTH_VOCAB = ["N", "13", "b13"]

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

DEGREE_TO_INTERVAL = {
    "1": 0, "#1": 1, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4,
    "4": 5, "#4": 6, "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9,
    "bb7": 9, "b7": 10, "7": 11,
    "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    dataset_root:   str            = "bello_dataset"
    sample_rate:    int            = 22050
    hop_length:     int            = 512
    fmin_note:      str            = "C1"
    n_bins:         int            = 252
    bins_per_octave: int           = 36
    segment_seconds: float         = 10.0
    max_songs:      Optional[int]  = None
    use_cache:      bool           = True
    refresh_cache:  bool           = False
    cache_dir:      str            = ".cache/chordformer"


# ---------------------------------------------------------------------------
# Config hash (covers every field that affects CQT shape/content)
# ---------------------------------------------------------------------------

def _cfg_hash(cfg: PreprocessingConfig) -> str:
    """
    Short MD5 hash of the preprocessing parameters that affect CQT output.
    Used as a subdirectory name so that changing any parameter automatically
    invalidates the entire per-song cache.

    Deliberately excludes: dataset_root, max_songs, use_cache,
    refresh_cache, cache_dir — these are control flags, not data parameters.
    """
    key = {
        "sample_rate":    cfg.sample_rate,
        "hop_length":     cfg.hop_length,
        "fmin_note":      cfg.fmin_note,
        "n_bins":         cfg.n_bins,
        "bins_per_octave": cfg.bins_per_octave,
        "segment_seconds": cfg.segment_seconds,
    }
    return hashlib.md5(str(sorted(key.items())).encode("utf-8")).hexdigest()[:12]


def _song_cache_dir(cfg: PreprocessingConfig) -> Path:
    d = Path(cfg.cache_dir) / "songs" / _cfg_hash(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _song_cache_path(audio_path: str, cfg: PreprocessingConfig) -> Path:
    stem = Path(audio_path).stem
    return _song_cache_dir(cfg) / f"{stem}.pt"


# ---------------------------------------------------------------------------
# Harte label parsing
# ---------------------------------------------------------------------------

def _note_to_pitch_class(note: str) -> Optional[int]:
    token = note.strip().replace("♭", "b").replace("♯", "#")
    match = re.match(r"^([A-Ga-g])([#b]*)$", token)
    if not match:
        return None
    base           = NOTE_TO_PC[match.group(1).upper()]
    accidentals    = match.group(2)
    semitone_shift = accidentals.count("#") - accidentals.count("b")
    return (base + semitone_shift) % 12


def _degree_to_pitch_class(degree: str, root_pc: Optional[int]) -> Optional[int]:
    if root_pc is None:
        return None
    token    = degree.strip().replace("*", "")
    interval = DEGREE_TO_INTERVAL.get(token)
    if interval is None:
        return None
    return (root_pc + interval) % 12


def _encode_root_triad(root_pc: Optional[int], triad: str) -> int:
    if root_pc is None or triad == "N":
        return 0
    triad_idx = ROOT_TRIAD_VOCAB.index(triad)
    return 1 + root_pc * len(ROOT_TRIAD_VOCAB) + triad_idx


def _encode_bass(bass_pc: Optional[int]) -> int:
    if bass_pc is None:
        return 0
    return bass_pc + 1


def _infer_triad(descriptor: str) -> str:
    d = descriptor.lower()
    if d == "" or d == "1":                              return "maj"
    if d.startswith("sus2"):                             return "sus2"
    if d.startswith("sus") or d.startswith("sus4"):      return "sus4"
    if "dim" in d or "hdim" in d:                        return "dim"
    if "aug" in d or d.startswith("+") or "#5" in d:    return "aug"
    if d.startswith("min") or ":min" in d:               return "min"
    if d.startswith("maj") or re.match(r"^(7|9|11|13)", d): return "maj"
    if d.startswith("5"):                                return "5"
    if d.startswith("("):
        has_b3 = "b3" in d
        has_b5 = "b5" in d
        has_3  = re.search(r"(^|[^b#])3", d) is not None
        if has_b3 and has_b5: return "dim"
        if has_b3:            return "min"
        if has_3:             return "maj"
        return "5"
    return "maj"


def _extract_extension_states(descriptor: str, triad: str) -> Tuple[str, str, str, str]:
    d          = descriptor.lower()
    seventh    = ninth = eleventh = thirteenth = "N"

    if "13" in d or "6" in d:
        if "maj13" in d: seventh, ninth, thirteenth = "7",  "9", "13"
        else:            seventh, ninth, thirteenth = "b7", "9", "13"
    elif "11" in d:
        if "maj11" in d: seventh, ninth, eleventh = "7",  "9", "11"
        else:            seventh, ninth, eleventh = "b7", "9", "11"
    elif "9" in d:
        if "maj9" in d: seventh, ninth = "7",  "9"
        else:           seventh, ninth = "b7", "9"
    elif "7" in d or triad == "dim":
        if   "maj7" in d:                  seventh = "7"
        elif "dim7" in d and triad == "dim": seventh = "bb7"
        elif "7" in d:                     seventh = "b7"

    matches = re.findall(r"(bb|b|#)?(7|9|11|13|6)", d)
    for accidental, degree in matches:
        if degree == "7":
            if   accidental == "bb":           seventh = "bb7"
            elif accidental == "b":            seventh = "b7"
            elif accidental == "" and "maj" in d: seventh = "7"
        elif degree == "9":
            if   accidental == "#": ninth = "#9"
            elif accidental == "b": ninth = "b9"
            else:                   ninth = "9"
        elif degree == "11":
            if accidental == "#": eleventh = "#11"
            else:                 eleventh = "11"
        elif degree in {"13", "6"}:
            if accidental == "b": thirteenth = "b13"
            else:                 thirteenth = "13"

    return seventh, ninth, eleventh, thirteenth


def parse_harte_label(label: str) -> Tuple[int, int, int, int, int, int]:
    symbol = label.strip()
    if symbol in {"N", "X", ""}:
        return 0, 0, 0, 0, 0, 0

    body, bass_token = (symbol.split("/", 1) + [None])[:2]
    if ":" in body:
        root_token, descriptor = body.split(":", 1)
    else:
        root_token, descriptor = body, "maj"

    root_pc  = _note_to_pitch_class(root_token)
    triad    = _infer_triad(descriptor)
    seventh, ninth, eleventh, thirteenth = _extract_extension_states(descriptor, triad)

    bass_pc = None
    if bass_token:
        bass_pc = _note_to_pitch_class(bass_token)
        if bass_pc is None:
            bass_pc = _degree_to_pitch_class(bass_token, root_pc)
    else:
        bass_pc = root_pc

    return (
        _encode_root_triad(root_pc, triad),
        _encode_bass(bass_pc),
        SEVENTH_VOCAB.index(seventh),
        NINTH_VOCAB.index(ninth),
        ELEVENTH_VOCAB.index(eleventh),
        THIRTEENTH_VOCAB.index(thirteenth),
    )


# ---------------------------------------------------------------------------
# Audio / label loading
# ---------------------------------------------------------------------------

def load_chord_intervals(
    lab_path: str,
) -> List[Tuple[float, float, Tuple[int, int, int, int, int, int]]]:
    intervals = []
    with open(lab_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            intervals.append((
                float(parts[0]),
                float(parts[1]),
                parse_harte_label(parts[2]),
            ))
    return intervals


def compute_cqt(audio_path: str, cfg: PreprocessingConfig) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=cfg.sample_rate)
    cqt  = librosa.cqt(
        y               = y,
        sr              = cfg.sample_rate,
        hop_length      = cfg.hop_length,
        fmin            = librosa.note_to_hz(cfg.fmin_note),
        n_bins          = cfg.n_bins,
        bins_per_octave = cfg.bins_per_octave,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db.T.astype(np.float32)


def align_intervals_to_frames(
    intervals: Sequence[Tuple[float, float, Tuple[int, int, int, int, int, int]]],
    n_frames:  int,
    cfg:       PreprocessingConfig,
) -> List[np.ndarray]:
    starts  = np.array([x[0] for x in intervals], dtype=np.float64)
    ends    = np.array([x[1] for x in intervals], dtype=np.float64)
    encoded = np.array([x[2] for x in intervals], dtype=np.int64)

    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=cfg.sample_rate, hop_length=cfg.hop_length
    )

    labels = [np.zeros(n_frames, dtype=np.int64) for _ in range(6)]
    if len(intervals) == 0:
        return labels

    interval_idx = np.searchsorted(ends, frame_times, side="right")
    valid  = interval_idx < len(intervals)
    valid &= frame_times >= starts[np.clip(interval_idx, 0, len(intervals) - 1)]

    for head in range(6):
        labels[head][valid] = encoded[interval_idx[valid], head]

    return labels


def find_audio_lab_pairs(dataset_root: str) -> List[Tuple[str, str]]:
    root      = Path(dataset_root)
    audio_dir = root / "audio"
    lab_dir   = root / "chordlab"
    filelist  = audio_dir / "filelist.txt"
    pairs: List[Tuple[str, str]] = []
    allowed   = [".mp3", ".wav", ".flac", ".ogg"]

    if filelist.exists():
        for line in filelist.read_text(encoding="utf-8").splitlines():
            stem     = Path(line.strip()).stem
            lab_path = lab_dir / f"{stem}.lab"
            if not lab_path.exists():
                continue
            for ext in allowed:
                candidate = audio_dir / f"{stem}{ext}"
                if candidate.exists():
                    pairs.append((str(candidate), str(lab_path)))
                    break
    else:
        for lab_path in sorted(lab_dir.glob("*.lab")):
            stem = lab_path.stem
            for ext in allowed:
                audio_path = audio_dir / f"{stem}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), str(lab_path)))
                    break

    return pairs


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _chunk_song(
    cqt:            np.ndarray,
    labels:         Sequence[np.ndarray],
    segment_frames: int,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    feature_chunks: List[np.ndarray]       = []
    label_chunks:   List[List[np.ndarray]] = []
    n_frames  = cqt.shape[0]
    pad_value = cqt.min() if n_frames > 0 else -80.0

    for start in range(0, n_frames, segment_frames):
        end  = min(start + segment_frames, n_frames)
        feat = cqt[start:end]
        labs = [head[start:end] for head in labels]

        if end - start < segment_frames:
            pad  = segment_frames - (end - start)
            feat = np.pad(feat, ((0, pad), (0, 0)),
                          mode="constant", constant_values=pad_value)
            labs = [np.pad(h, (0, pad), mode="constant", constant_values=-100)
                    for h in labs]

        feature_chunks.append(feat)
        label_chunks.append(labs)

    return feature_chunks, label_chunks


# ---------------------------------------------------------------------------
# Stage 1: per-song cache
# ---------------------------------------------------------------------------

def precompute_song_cache(
    pairs: List[Tuple[str, str]],
    cfg:   PreprocessingConfig,
) -> None:
    """
    Compute and cache CQT + labels for every song in *pairs*.

    Each song is written to:
        <cfg.cache_dir>/songs/<cfg_hash>/<stem>.pt

    Songs whose cache file already exists are skipped unless
    cfg.refresh_cache is True.  Safe to call multiple times and across
    folds — by fold 2 every file is already present and the function
    returns in milliseconds.
    """
    if not cfg.use_cache:
        return

    segment_frames = int(round(
        cfg.segment_seconds * cfg.sample_rate / cfg.hop_length
    ))

    total   = len(pairs)
    missing = [
        (a, l) for a, l in pairs
        if cfg.refresh_cache or not _song_cache_path(a, cfg).exists()
    ]

    if not missing:
        print(f"  Song cache: all {total} songs already cached — skipping CQT computation.")
        return

    print(f"  Song cache: computing CQTs for {len(missing)}/{total} songs …")
    for i, (audio_path, lab_path) in enumerate(missing, 1):
        cache_file = _song_cache_path(audio_path, cfg)
        try:
            cqt       = compute_cqt(audio_path, cfg)
            intervals = load_chord_intervals(lab_path)
            labels    = align_intervals_to_frames(intervals, cqt.shape[0], cfg)
            feat_chunks, label_chunks = _chunk_song(cqt, labels, segment_frames)

            torch.save(
                {
                    "features": [
                        torch.tensor(f, dtype=torch.float32)
                        for f in feat_chunks
                    ],
                    "targets": [
                        [torch.tensor(l, dtype=torch.long) for l in labs]
                        for labs in label_chunks
                    ],
                },
                cache_file,
            )
        except Exception as exc:
            print(f"  WARNING: skipping {Path(audio_path).name} — {exc}")
            continue

        if i % 50 == 0 or i == len(missing):
            print(f"    {i}/{len(missing)} done")

    print(f"  Song cache complete → {_song_cache_dir(cfg)}")


# ---------------------------------------------------------------------------
# Stage 2: split-view dataset
# ---------------------------------------------------------------------------

class BelloChordFormerDataset(Dataset):
    """
    Assembles a split (train / val / test) by loading pre-computed per-song
    cache files produced by precompute_song_cache().

    The constructor does NO audio loading or CQT computation — it only reads
    .pt files from disk and concatenates their chunk lists.

    Public interface is unchanged from the original:
        BelloChordFormerDataset(file_pairs, cfg, augment, split_name)
        __len__()
        __getitem__(idx) -> (cqt_tensor, [label_tensor × 6])

    split_name is kept as a parameter for API compatibility but is no longer
    used for caching (there are no split-level cache files).
    """

    def __init__(
        self,
        file_pairs: List[Tuple[str, str]],
        cfg:        Optional[PreprocessingConfig] = None,
        augment:    bool = False,
        split_name: str  = "train",
    ):
        super().__init__()
        self.cfg        = cfg or PreprocessingConfig()
        self.augment    = augment
        self.split_name = split_name

        self.features: List[torch.Tensor]       = []
        self.targets:  List[List[torch.Tensor]] = []

        missing = []
        for audio_path, _ in file_pairs:
            cache_file = _song_cache_path(audio_path, self.cfg)
            if not cache_file.exists():
                missing.append(audio_path)
                continue
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            self.features.extend(payload["features"])
            self.targets.extend(payload["targets"])

        if missing:
            raise RuntimeError(
                f"BelloChordFormerDataset ({split_name}): {len(missing)} song(s) have no "
                f"cache file.  Call precompute_song_cache() before building DataLoaders.\n"
                f"First missing: {missing[0]}"
            )

        print(f"  Loaded {split_name}: {len(file_pairs)} songs → {len(self.features)} segments")

    def __len__(self) -> int:
        return len(self.features)

    # ------------------------------------------------------------------
    # Augmentation helpers (unchanged)
    # ------------------------------------------------------------------

    def _shift_cqt(self, cqt: torch.Tensor, shift_bins: int) -> torch.Tensor:
        if shift_bins == 0:
            return cqt
        shifted = torch.full_like(cqt, fill_value=cqt.min())
        if shift_bins > 0:
            shifted[:, shift_bins:] = cqt[:, :-shift_bins]
        else:
            shift_bins = abs(shift_bins)
            shifted[:, :-shift_bins] = cqt[:, shift_bins:]
        return shifted

    def _transpose_labels(
        self, targets: List[torch.Tensor], semitone_shift: int
    ) -> List[torch.Tensor]:
        if semitone_shift == 0:
            return targets
        transposed = [t.clone() for t in targets]

        # Head 0: Root/Triad  (encoding: 1 + root_pc * 7 + triad_idx)
        rt     = transposed[0]
        mask_0 = rt > 0
        if mask_0.any():
            val    = rt[mask_0] - 1
            pc     = val // 7
            ti     = val %  7
            new_pc = (pc + semitone_shift) % 12
            transposed[0][mask_0] = 1 + (new_pc * 7) + ti

        # Head 1: Bass  (encoding: bass_pc + 1)
        bass   = transposed[1]
        mask_1 = bass > 0
        if mask_1.any():
            pc     = bass[mask_1] - 1
            new_pc = (pc + semitone_shift) % 12
            transposed[1][mask_1] = new_pc + 1

        # Heads 2-5: extensions — pitch-class independent, no change
        return transposed

    def __getitem__(self, idx: int):
        cqt     = self.features[idx]
        targets = self.targets[idx]

        if self.augment:
            semitone_shift = random.randint(-5, 6)
            if semitone_shift != 0:
                shift_bins = semitone_shift * self.cfg.bins_per_octave // 12
                cqt        = self._shift_cqt(cqt, shift_bins)
                targets    = self._transpose_labels(targets, semitone_shift)

        return cqt, targets


# ---------------------------------------------------------------------------
# Standalone helper (not used by train_unified, kept for direct use)
# ---------------------------------------------------------------------------

def create_dataloaders(
    cfg:         PreprocessingConfig,
    batch_size:  int = 24,
    num_workers: int = 4,
):
    """Simple single-split dataloader builder for standalone / debug use."""
    all_pairs = find_audio_lab_pairs(cfg.dataset_root)
    if cfg.max_songs is not None:
        all_pairs = all_pairs[:cfg.max_songs]

    random.seed(42)
    random.shuffle(all_pairs)
    total     = len(all_pairs)
    train_end = int(0.6 * total)
    val_end   = int(0.8 * total)

    train_pairs = all_pairs[:train_end]
    val_pairs   = all_pairs[train_end:val_end]
    test_pairs  = all_pairs[val_end:]

    print(f"Song split: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    precompute_song_cache(all_pairs, cfg)

    train_ds = BelloChordFormerDataset(train_pairs, cfg, augment=True,  split_name="train")
    val_ds   = BelloChordFormerDataset(val_pairs,   cfg, augment=False, split_name="val")
    test_ds  = BelloChordFormerDataset(test_pairs,  cfg, augment=False, split_name="test")

    kwargs   = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kwargs)

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PreprocessingConfig(dataset_root="bello_dataset", segment_seconds=10.0)
    train_dl, val_dl, test_dl = create_dataloaders(cfg)

    sample_x, sample_y = train_dl.dataset[0]
    print(f"Segments : {len(train_dl.dataset)}")
    print(f"Input    : {sample_x.shape}")
    print(f"Heads    : {[y.shape for y in sample_y]}")
    print(f"Dims     : {[85, 13, 4, 4, 3, 3]}")