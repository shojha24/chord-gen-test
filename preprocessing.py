from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import re
import hashlib
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Vocabulary ────────────────────────────────────────────────────────────────
# Root/triad head: 0 = N/X, 1–84 = 12 roots × 7 triads
# Triads match the ChordFormer paper exactly (no separate hdim class).
# hdim7 is folded into dim + bb7 in the seventh head (paper-correct).
ROOT_TRIAD_VOCAB = ["5", "maj", "min", "sus4", "sus2", "dim", "aug"]  # 7 triads, index 0-6
SEVENTH_VOCAB    = ["N", "7", "b7", "bb7"]   # index 0-3
NINTH_VOCAB      = ["N", "9", "#9", "b9"]    # index 0-3
ELEVENTH_VOCAB   = ["N", "11", "#11"]         # index 0-2
THIRTEENTH_VOCAB = ["N", "13", "b13"]         # index 0-2

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

DEGREE_TO_INTERVAL = {
    "1": 0, "#1": 1, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4,
    "4": 5, "#4": 6, "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9,
    "bb7": 9, "b7": 10, "7": 11,
    "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9,
}


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    dataset_root: str = "bello_dataset"
    sample_rate: int = 22050
    hop_length: int = 512
    fmin_note: str = "C1"
    n_bins: int = 252
    bins_per_octave: int = 36
    segment_seconds: float = 23.2
    max_songs: Optional[int] = None
    use_cache: bool = True
    refresh_cache: bool = False
    cache_dir: str = ".cache/chordformer"


# ── Note / degree helpers ─────────────────────────────────────────────────────
def _note_to_pitch_class(note: str) -> Optional[int]:
    token = note.strip().replace("♭", "b").replace("♯", "#")
    match = re.match(r"^([A-Ga-g])([#b]*)$", token)
    if not match:
        return None
    base = NOTE_TO_PC[match.group(1).upper()]
    accidentals = match.group(2)
    return (base + accidentals.count("#") - accidentals.count("b")) % 12


def _degree_to_pitch_class(degree: str, root_pc: Optional[int]) -> Optional[int]:
    if root_pc is None:
        return None
    token = degree.strip().replace("*", "")
    interval = DEGREE_TO_INTERVAL.get(token)
    if interval is None:
        return None
    return (root_pc + interval) % 12


# ── Encoding helpers ──────────────────────────────────────────────────────────
def _encode_root_triad(root_pc: Optional[int], triad: str) -> int:
    """Returns 0 for silence/unknown, else 1 + root_pc * 7 + triad_idx."""
    if root_pc is None or triad == "N":
        return 0
    triad_idx = ROOT_TRIAD_VOCAB.index(triad)
    return 1 + root_pc * len(ROOT_TRIAD_VOCAB) + triad_idx


def _encode_bass(bass_pc: Optional[int]) -> int:
    return 0 if bass_pc is None else bass_pc + 1


# ── Triad inference ───────────────────────────────────────────────────────────
def _infer_triad(descriptor: str) -> str:
    """
    Maps a Harte descriptor to one of ROOT_TRIAD_VOCAB.
    Rules (highest priority first):
      • sus2 / sus4
      • aug / #5
      • hdim  → dim  (paper-correct: bb7 is captured in seventh head)
      • dim
      • min
      • "5"   → power chord class
      • maj (default for bare numbers like 7/9/11/13 and empty)
    """
    d = descriptor.lower()

    # Bare root forms ("1", "1/1", or empty)
    if d in ("", "1"):
        return "maj"

    if d.startswith("sus2"):
        return "sus2"
    if d.startswith("sus4") or d.startswith("sus"):
        return "sus4"
    if "aug" in d or d.startswith("+") or "#5" in d:
        return "aug"
    if "hdim" in d or "dim" in d:
        return "dim"
    if d.startswith("min") or ":min" in d:
        return "min"
    if d.startswith("maj") or re.match(r"^(7|9|11|13)", d):
        return "maj"
    if d.startswith("5"):
        return "5"

    # Interval-list form e.g. "(1,5)", "(1)", "(b3,b5)"
    if d.startswith("("):
        has_b3 = "b3" in d
        has_b5 = "b5" in d
        has_3  = re.search(r"(^|[^b#])3", d) is not None
        if has_b3 and has_b5:
            return "dim"
        if has_b3:
            return "min"
        if has_3:
            return "maj"
        return "5"   # root+5th only → power chord

    return "maj"


# ── Extension extraction ──────────────────────────────────────────────────────
def _extract_extension_states(descriptor: str, triad: str) -> Tuple[str, str, str, str]:
    """
    Returns (seventh, ninth, eleventh, thirteenth) strings drawn from their
    respective vocabs.

    Fixed bugs vs. original:
      1. maj6 / min6 / 6 / 6-9 chords: 7th = N  (the "6" is an *add* tone, not
         a dominant cascade).  The 6th itself is still captured as thirteenth=13.
      2. minmaj7: seventh = "7"  (major seventh over minor triad).
      3. dim7 cascade: seventh = "bb7", ninth stays "N"  (was wrongly set to "9").
      4. 5(b7) and similar: seventh = "b7" via phase-2 regex even when triad="5"
         (phase-1 cascade guard must not swallow these).
      5. add chords: seventh always "N".
    """
    d = descriptor.lower()
    seventh = ninth = eleventh = thirteenth = "N"

    # ── Classifier flags ──────────────────────────────────────────────────────
    is_sixth_chord = bool(re.match(r"^(maj6|min6|6)(?![/9])", d)) \
                  or bool(re.match(r"^6/9", d))
    is_add         = d.startswith("add")
    is_minmaj7     = "minmaj7" in d

    # ── Phase 1: cascade from the highest extension present ──────────────────
    if is_sixth_chord or is_add:
        # No seventh cascade; phase-2 will pick up accidentals only.
        pass

    elif is_minmaj7:
        # Minor-major seventh: major 7th, no higher cascade unless explicit.
        seventh = "7"

    elif "13" in d or ("6" in d and not is_sixth_chord):
        if "maj13" in d:
            seventh, ninth, thirteenth = "7",  "9", "13"
        else:
            seventh, ninth, thirteenth = "b7", "9", "13"

    elif "11" in d:
        if "maj11" in d:
            seventh, ninth, eleventh = "7",  "9", "11"
        else:
            seventh, ninth, eleventh = "b7", "9", "11"

    elif "9" in d:
        if "maj9" in d:
            seventh, ninth = "7",  "9"
        else:
            seventh, ninth = "b7", "9"

    elif "7" in d or triad == "dim":
        if "maj7" in d:
            seventh = "7"
        elif triad == "dim" and "dim7" in d:
            # Fully-diminished seventh — bb7, do NOT cascade a ninth.
            seventh = "bb7"
            ninth   = "N"
        elif triad == "dim" and "hdim" in d:
            # Half-diminished: minor seventh (b7) over dim triad.
            seventh = "b7"
        elif "7" in d:
            seventh = "b7"

    # ── Phase 2: explicit accidentals override / fill in ─────────────────────
    matches = re.findall(r"(bb|b|#)?(7|9|11|13|6)", d)
    for acc, deg in matches:
        if deg == "7":
            if acc == "bb":
                seventh = "bb7"
            elif acc == "b":
                seventh = "b7"
            elif acc == "" and "maj" in d and not is_sixth_chord and not is_minmaj7:
                seventh = "7"
        elif deg == "9":
            if acc == "#":
                ninth = "#9"
            elif acc == "b":
                ninth = "b9"
            else:
                ninth = "9"
        elif deg == "11":
            eleventh = "#11" if acc == "#" else "11"
        elif deg in {"13", "6"}:
            thirteenth = "b13" if acc == "b" else "13"

    # add chords never cascade a seventh
    if is_add:
        seventh = "N"

    return seventh, ninth, eleventh, thirteenth


# ── Harte label parser ────────────────────────────────────────────────────────
def parse_harte_label(label: str) -> Tuple[int, int, int, int, int, int]:
    """
    Returns a 6-tuple of integer indices:
        (root_triad_idx, bass_idx, seventh_idx, ninth_idx, eleventh_idx, thirteenth_idx)

    Head sizes: 85, 13, 4, 4, 3, 3
      root_triad: 0 = N, 1-84 = 12 roots × 7 triads
      bass:       0 = N, 1-12 = pitch class + 1
    """
    symbol = label.strip()
    if symbol in {"N", "X", ""}:
        return 0, 0, 0, 0, 0, 0

    # Split off bass note
    body, bass_token = (symbol.split("/", 1) + [None])[:2]

    # Split root from descriptor
    if ":" in body:
        root_token, descriptor = body.split(":", 1)
    else:
        # Bare note with no colon → treat as major (e.g. "G", "C#")
        root_token, descriptor = body, "maj"

    # Handle "1/1" bare-root form: descriptor becomes "1", bass_token becomes "1"
    # We want triad=maj and bass=root (not a degree).
    if descriptor == "1":
        descriptor = "maj"
        # bass_token from "1/1" split is "1" — treat as root degree below.

    root_pc = _note_to_pitch_class(root_token)
    triad   = _infer_triad(descriptor)
    seventh, ninth, eleventh, thirteenth = _extract_extension_states(descriptor, triad)

    # Resolve bass
    if bass_token:
        bass_pc = _note_to_pitch_class(bass_token)
        if bass_pc is None:
            # Degree-based bass (e.g. "/5", "/b3")
            bass_pc = _degree_to_pitch_class(bass_token, root_pc)
    else:
        bass_pc = root_pc  # no explicit bass → root in bass

    root_triad_idx  = _encode_root_triad(root_pc, triad)
    bass_idx        = _encode_bass(bass_pc)
    seventh_idx     = SEVENTH_VOCAB.index(seventh)
    ninth_idx       = NINTH_VOCAB.index(ninth)
    eleventh_idx    = ELEVENTH_VOCAB.index(eleventh)
    thirteenth_idx  = THIRTEENTH_VOCAB.index(thirteenth)

    return root_triad_idx, bass_idx, seventh_idx, ninth_idx, eleventh_idx, thirteenth_idx


# ── File I/O ──────────────────────────────────────────────────────────────────
def load_chord_intervals(
    lab_path: str,
) -> List[Tuple[float, float, Tuple[int, int, int, int, int, int]]]:
    intervals = []
    with open(lab_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            intervals.append((float(parts[0]), float(parts[1]), parse_harte_label(parts[2])))
    return intervals


def compute_cqt(audio_path: str, cfg: PreprocessingConfig) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=cfg.sample_rate)
    cqt = librosa.cqt(
        y=y,
        sr=cfg.sample_rate,
        hop_length=cfg.hop_length,
        fmin=librosa.note_to_hz(cfg.fmin_note),
        n_bins=cfg.n_bins,
        bins_per_octave=cfg.bins_per_octave,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db.T.astype(np.float32)  # (frames, bins)


def align_intervals_to_frames(
    intervals: Sequence[Tuple[float, float, Tuple[int, int, int, int, int, int]]],
    n_frames: int,
    cfg: PreprocessingConfig,
) -> List[np.ndarray]:
    starts  = np.array([x[0] for x in intervals], dtype=np.float64)
    ends    = np.array([x[1] for x in intervals], dtype=np.float64)
    encoded = np.array([x[2] for x in intervals], dtype=np.int64)

    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=cfg.sample_rate, hop_length=cfg.hop_length,
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
    exts      = [".mp3", ".wav", ".flac", ".ogg"]

    pairs: List[Tuple[str, str]] = []

    if filelist.exists():
        for line in filelist.read_text(encoding="utf-8").splitlines():
            stem     = Path(line.strip()).stem
            lab_path = lab_dir / f"{stem}.lab"
            if not lab_path.exists():
                continue
            for ext in exts:
                candidate = audio_dir / f"{stem}{ext}"
                if candidate.exists():
                    pairs.append((str(candidate), str(lab_path)))
                    break
    else:
        for lab_path in sorted(lab_dir.glob("*.lab")):
            stem = lab_path.stem
            for ext in exts:
                audio_path = audio_dir / f"{stem}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), str(lab_path)))
                    break

    return pairs


# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk_song(
    cqt: np.ndarray,
    labels: Sequence[np.ndarray],
    segment_frames: int,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    feature_chunks: List[np.ndarray] = []
    label_chunks:   List[List[np.ndarray]] = []

    n_frames  = cqt.shape[0]
    pad_value = cqt.min() if n_frames > 0 else -80.0  # silence in dB

    for start in range(0, n_frames, segment_frames):
        end  = min(start + segment_frames, n_frames)
        feat = cqt[start:end]
        labs = [head[start:end] for head in labels]

        if end - start < segment_frames:
            pad  = segment_frames - (end - start)
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant", constant_values=pad_value)
            # -100 is ignored by CrossEntropyLoss
            labs = [np.pad(h, (0, pad), mode="constant", constant_values=-100) for h in labs]

        feature_chunks.append(feat)
        label_chunks.append(labs)

    return feature_chunks, label_chunks


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_file_path(cfg: PreprocessingConfig) -> Path:
    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = {
        "dataset_root":    str(Path(cfg.dataset_root).resolve()),
        "sample_rate":     cfg.sample_rate,
        "hop_length":      cfg.hop_length,
        "fmin_note":       cfg.fmin_note,
        "n_bins":          cfg.n_bins,
        "bins_per_octave": cfg.bins_per_octave,
        "segment_seconds": cfg.segment_seconds,
        "max_songs":       cfg.max_songs,
    }
    digest = hashlib.md5(str(sorted(key.items())).encode()).hexdigest()[:12]
    return cache_root / f"bello_{digest}.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────
class BelloChordFormerDataset(Dataset):
    def __init__(
        self,
        file_pairs:  List[Tuple[str, str]],
        cfg:         Optional[PreprocessingConfig] = None,
        augment:     bool = False,
        split_name:  str  = "train",
    ):
        super().__init__()
        self.cfg        = cfg or PreprocessingConfig()
        self.augment    = augment
        self.split_name = split_name

        segment_frames = int(round(
            self.cfg.segment_seconds * self.cfg.sample_rate / self.cfg.hop_length
        ))
        if segment_frames <= 0:
            raise ValueError("segment_seconds must produce at least 1 frame")

        base_cache = _cache_file_path(self.cfg)
        cache_path = base_cache.parent / f"{base_cache.stem}_{split_name}.pt"

        if self.cfg.use_cache and cache_path.exists() and not self.cfg.refresh_cache:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.features: List[torch.Tensor]       = payload["features"]
            self.targets:  List[List[torch.Tensor]] = payload["targets"]
            print(f"Loaded cached {split_name} dataset: {len(self.features)} segments")
        else:
            self.features = []
            self.targets  = []

            print(f"Building {split_name} dataset…")
            for audio_path, lab_path in file_pairs:
                cqt       = compute_cqt(audio_path, self.cfg)
                intervals = load_chord_intervals(lab_path)
                labels    = align_intervals_to_frames(intervals, cqt.shape[0], self.cfg)
                feat_chunks, label_chunks = _chunk_song(cqt, labels, segment_frames)

                for feat, labs in zip(feat_chunks, label_chunks):
                    self.features.append(torch.tensor(feat, dtype=torch.float32))
                    self.targets.append([torch.tensor(x, dtype=torch.long) for x in labs])

            if self.cfg.use_cache:
                torch.save({"features": self.features, "targets": self.targets}, cache_path)
                print(f"Saved {split_name} cache: {len(self.features)} segments")

    def __len__(self) -> int:
        return len(self.features)

    # ── Pitch shifting ────────────────────────────────────────────────────────
    def _shift_cqt(self, cqt: torch.Tensor, shift_bins: int) -> torch.Tensor:
        """
        Shift the CQT along the frequency axis by `shift_bins`.
        Positive = pitch UP (rows move toward higher-frequency bins).
        Empty bins are filled with the minimum dB value (silence).
        """
        if shift_bins == 0:
            return cqt
        out = torch.full_like(cqt, fill_value=cqt.min())
        if shift_bins > 0:
            # Shift UP: content moves from low bins to higher bins.
            out[:, shift_bins:] = cqt[:, :-shift_bins]
        else:
            # Shift DOWN: content moves from high bins to lower bins.
            s = abs(shift_bins)
            out[:, :-s] = cqt[:, s:]
        return out

    def _transpose_labels(
        self, targets: List[torch.Tensor], semitone_shift: int
    ) -> List[torch.Tensor]:
        """
        Transpose root_triad (head 0) and bass (head 1) by `semitone_shift`.
        Extension heads (2-5) are pitch-class-independent and are left unchanged.

        Encoding review:
          root_triad : 0 = N,  else  1 + root_pc * 7 + triad_idx
          bass       : 0 = N,  else  bass_pc + 1
        """
        if semitone_shift == 0:
            return targets

        out = [t.clone() for t in targets]

        # Head 0 — root/triad
        rt   = out[0]
        mask = rt > 0
        if mask.any():
            val      = rt[mask] - 1
            pc       = val // 7
            tidx     = val % 7
            new_pc   = (pc + semitone_shift) % 12
            rt[mask] = 1 + new_pc * 7 + tidx

        # Head 1 — bass
        bass = out[1]
        mask = bass > 0
        if mask.any():
            pc         = bass[mask] - 1
            new_pc     = (pc + semitone_shift) % 12
            bass[mask] = new_pc + 1

        return out

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


# ── Dataloaders ───────────────────────────────────────────────────────────────
def create_dataloaders(
    cfg:         PreprocessingConfig,
    batch_size:  int = 24,
    num_workers: int = 4,
):
    all_pairs = find_audio_lab_pairs(cfg.dataset_root)
    if cfg.max_songs is not None:
        all_pairs = all_pairs[:cfg.max_songs]

    random.seed(42)
    random.shuffle(all_pairs)

    total      = len(all_pairs)
    train_end  = int(0.6 * total)
    val_end    = int(0.8 * total)

    train_pairs = all_pairs[:train_end]
    val_pairs   = all_pairs[train_end:val_end]
    test_pairs  = all_pairs[val_end:]

    print(f"Song split: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    train_ds = BelloChordFormerDataset(train_pairs, cfg, augment=True,  split_name="train")
    val_ds   = BelloChordFormerDataset(val_pairs,   cfg, augment=False, split_name="val")
    test_ds  = BelloChordFormerDataset(test_pairs,  cfg, augment=False, split_name="test")

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_dl  = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_dl    = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_dl   = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    return train_dl, val_dl, test_dl


# ── Output head sizes (for model construction) ────────────────────────────────
# root_triad : 1 + 12 * 7 = 85
# bass       : 1 + 12     = 13
# seventh    : 4
# ninth      : 4
# eleventh   : 3
# thirteenth : 3
HEAD_SIZES = [85, 13, 4, 4, 3, 3]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick parser unit tests — no audio needed.
    def _check(label, expected):
        got = parse_harte_label(label)
        status = "✓" if got == expected else "✗"
        print(f"  {status}  {label:<30s}  expected={expected}  got={got}")

    print("=== Parser smoke tests ===")
    # Silence
    _check("N",            (0, 0, 0, 0, 0, 0))
    _check("X",            (0, 0, 0, 0, 0, 0))
    # maj6: 7th must be N
    _check("D:maj6",       parse_harte_label("D:maj"))   # triad same, 7th=N, 13th=13
    # Re-check manually: D root=2, maj triad idx=1 → 1+2*7+1=16
    # 7th=N=0, 9th=N=0, 11th=N=0, 13th=13=1
    _check("D:maj6",       (16, 3, 0, 0, 0, 1))   # bass=D=3
    # minmaj7: 7th=7 (major seventh), triad=min
    # C root=0, min triad idx=2 → 1+0*7+2=3
    _check("C:minmaj7",    (3, 1, 1, 0, 0, 0))
    # hdim7: triad=dim, 7th=b7 (half-diminished = minor seventh)
    # D root=2, dim idx=5 → 1+2*7+5=20; 7th=b7=2
    _check("D:hdim7",      (20, 3, 2, 0, 0, 0))
    # dim7: triad=dim, 7th=bb7, 9th=N (no cascade!)
    # B root=11, dim idx=5 → 1+11*7+5=83; 7th=bb7=3
    _check("B:dim7",       (83, 12, 3, 0, 0, 0))
    # 5(b7): triad=5, 7th=b7
    # E root=4, "5" idx=0 → 1+4*7+0=29; 7th=b7=2
    _check("E:5(b7)",      (29, 5, 2, 0, 0, 0))
    # 1/1 bare root
    _check("E:1",          (29, 5, 0, 0, 0, 0))   # triad=maj for E, bass=E
    # power chord
    _check("G:5",          (1+7*7+0, 8, 0, 0, 0, 0))   # G root=7, "5" idx=0 → 50

    print("\n=== Head sizes ===")
    print(f"  {HEAD_SIZES}")

    # Full pipeline test (requires bello_dataset)
    try:
        cfg = PreprocessingConfig(dataset_root="bello_dataset", segment_seconds=10.0)
        train_dl, val_dl, test_dl = create_dataloaders(cfg)
        x, y = train_dl.dataset[0]
        print(f"\n=== Dataset ===")
        print(f"  Train segments : {len(train_dl.dataset)}")
        print(f"  Input shape    : {x.shape}")
        print(f"  Head shapes    : {[t.shape for t in y]}")
    except Exception as e:
        print(f"\n(Skipping full pipeline test: {e})")