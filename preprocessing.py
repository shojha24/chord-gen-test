from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import hashlib
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROOT_TRIAD_VOCAB = ["5", "maj", "min", "sus4", "sus2", "dim", "aug"]
SEVENTH_VOCAB = ["N", "7", "b7", "bb7"]
NINTH_VOCAB = ["N", "9", "#9", "b9"]
ELEVENTH_VOCAB = ["N", "11", "#11"]
THIRTEENTH_VOCAB = ["N", "13", "b13"]

NOTE_TO_PC = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

DEGREE_TO_INTERVAL = {
    "1": 0,
    "#1": 1,
    "b2": 1,
    "2": 2,
    "#2": 3,
    "b3": 3,
    "3": 4,
    "4": 5,
    "#4": 6,
    "b5": 6,
    "5": 7,
    "#5": 8,
    "b6": 8,
    "6": 9,
    "bb7": 9,
    "b7": 10,
    "7": 11,
    "b9": 1,
    "9": 2,
    "#9": 3,
    "11": 5,
    "#11": 6,
    "b13": 8,
    "13": 9,
}


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


def _note_to_pitch_class(note: str) -> Optional[int]:
    token = note.strip().replace("♭", "b").replace("♯", "#")
    match = re.match(r"^([A-Ga-g])([#b]*)$", token)
    if not match:
        return None

    base = NOTE_TO_PC[match.group(1).upper()]
    accidentals = match.group(2)
    semitone_shift = accidentals.count("#") - accidentals.count("b")
    return (base + semitone_shift) % 12


def _degree_to_pitch_class(degree: str, root_pc: Optional[int]) -> Optional[int]:
    if root_pc is None:
        return None
    token = degree.strip().replace("*", "")
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
    if d == "" or d == "1":
        return "maj"
    if d.startswith("sus2"):
        return "sus2"
    if d.startswith("sus") or d.startswith("sus4"):
        return "sus4"
    if "dim" in d or "hdim" in d:
        return "dim"
    if "aug" in d or d.startswith("+") or "#5" in d:
        return "aug"
    if d.startswith("min") or ":min" in d:
        return "min"
    if d.startswith("maj") or re.match(r"^(7|9|11|13)", d):
        return "maj"
    if d.startswith("5"):
        return "5"  # <--- Map to the new power chord class, NOT silence
    if d.startswith("("):
        has_b3 = "b3" in d
        has_b5 = "b5" in d
        has_3 = re.search(r"(^|[^b#])3", d) is not None
        if has_b3 and has_b5: return "dim"
        if has_b3: return "min"
        if has_3: return "maj"
        return "5"  # <--- If no 3rd is found, fallback to power chord instead of silence
    return "maj"


def _extract_extension_states(descriptor: str, triad: str) -> Tuple[str, str, str, str]:
    d = descriptor.lower()

    seventh = "N"
    ninth = "N"
    eleventh = "N"
    thirteenth = "N"

    # PHASE 1: Broadly catch base extension levels (cascading defaults)
    # Checking highest intervals first ensures we cascade down correctly
    if "13" in d or "6" in d:
        if "maj13" in d:
            seventh, ninth, thirteenth = "7", "9", "13"
        else:
            seventh, ninth, thirteenth = "b7", "9", "13"
    elif "11" in d:
        if "maj11" in d:
            seventh, ninth, eleventh = "7", "9", "11"
        else:
            seventh, ninth, eleventh = "b7", "9", "11"
    elif "9" in d:
        if "maj9" in d:
            seventh, ninth = "7", "9"
        else:
            seventh, ninth = "b7", "9"
    elif "7" in d or triad == "dim": 
        if "maj7" in d:
            seventh = "7"
        elif "dim7" in d and triad == "dim":
            seventh = "bb7"
        elif "7" in d:
            seventh = "b7"

    # PHASE 2: Extract explicit modifiers and apply specific accidentals
    matches = re.findall(r"(bb|b|#)?(7|9|11|13|6)", d)
    for accidental, degree in matches:
        if degree == "7":
            if accidental == "bb": 
                seventh = "bb7"
            elif accidental == "b": 
                seventh = "b7"
            elif accidental == "" and "maj" in d: 
                seventh = "7" # Only force a major 7th if "maj" is explicitly in the descriptor
        elif degree == "9":
            if accidental == "#": ninth = "#9"
            elif accidental == "b": ninth = "b9"
            else: ninth = "9"
        elif degree == "11":
            if accidental == "#": eleventh = "#11"
            else: eleventh = "11"
        elif degree in {"13", "6"}:
            if accidental == "b": thirteenth = "b13"
            else: thirteenth = "13"

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

    root_pc = _note_to_pitch_class(root_token)
    triad = _infer_triad(descriptor)
    seventh, ninth, eleventh, thirteenth = _extract_extension_states(descriptor, triad)

    bass_pc = None
    if bass_token:
        bass_pc = _note_to_pitch_class(bass_token)
        if bass_pc is None:
            bass_pc = _degree_to_pitch_class(bass_token, root_pc)
    else:
        bass_pc = root_pc

    root_triad_idx = _encode_root_triad(root_pc, triad)
    bass_idx = _encode_bass(bass_pc)
    seventh_idx = SEVENTH_VOCAB.index(seventh)
    ninth_idx = NINTH_VOCAB.index(ninth)
    eleventh_idx = ELEVENTH_VOCAB.index(eleventh)
    thirteenth_idx = THIRTEENTH_VOCAB.index(thirteenth)

    return root_triad_idx, bass_idx, seventh_idx, ninth_idx, eleventh_idx, thirteenth_idx


def load_chord_intervals(lab_path: str) -> List[Tuple[float, float, Tuple[int, int, int, int, int, int]]]:
    intervals = []
    with open(lab_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            start_time = float(parts[0])
            end_time = float(parts[1])
            label = parts[2]
            intervals.append((start_time, end_time, parse_harte_label(label)))
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
    return cqt_db.T.astype(np.float32)


def align_intervals_to_frames(
    intervals: Sequence[Tuple[float, float, Tuple[int, int, int, int, int, int]]],
    n_frames: int,
    cfg: PreprocessingConfig,
) -> List[np.ndarray]:
    starts = np.array([x[0] for x in intervals], dtype=np.float64)
    ends = np.array([x[1] for x in intervals], dtype=np.float64)
    encoded = np.array([x[2] for x in intervals], dtype=np.int64)

    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=cfg.sample_rate,
        hop_length=cfg.hop_length,
    )

    labels = [np.zeros(n_frames, dtype=np.int64) for _ in range(6)]
    if len(intervals) == 0:
        return labels

    interval_idx = np.searchsorted(ends, frame_times, side="right")
    valid = interval_idx < len(intervals)
    valid &= frame_times >= starts[np.clip(interval_idx, 0, len(intervals) - 1)]

    for head in range(6):
        labels[head][valid] = encoded[interval_idx[valid], head]

    return labels


def find_audio_lab_pairs(dataset_root: str) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    audio_dir = root / "audio"
    lab_dir = root / "chordlab"
    filelist = audio_dir / "filelist.txt"

    pairs: List[Tuple[str, str]] = []
    allowed_audio_exts = [".mp3", ".wav", ".flac", ".ogg"]

    if filelist.exists():
        for line in filelist.read_text(encoding="utf-8").splitlines():
            stem = Path(line.strip()).stem
            lab_path = lab_dir / f"{stem}.lab"
            if not lab_path.exists():
                continue
            audio_path = None
            for ext in allowed_audio_exts:
                candidate = audio_dir / f"{stem}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is not None:
                pairs.append((str(audio_path), str(lab_path)))
    else:
        for lab_path in sorted(lab_dir.glob("*.lab")):
            stem = lab_path.stem
            for ext in allowed_audio_exts:
                audio_path = audio_dir / f"{stem}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), str(lab_path)))
                    break

    return pairs


def _chunk_song(
    cqt: np.ndarray,
    labels: Sequence[np.ndarray],
    segment_frames: int,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[List[np.ndarray]] = []

    n_frames = cqt.shape[0]
    
    # NEW: Find the actual "silence" value of this specific CQT to use as padding
    pad_value = cqt.min() if n_frames > 0 else -80.0 

    for start in range(0, n_frames, segment_frames):
        end = min(start + segment_frames, n_frames)
        feat = cqt[start:end]
        labs = [head[start:end] for head in labels]

        if end - start < segment_frames:
            pad = segment_frames - (end - start)
            
            # FIXED: Pad with silence (negative dB), not max volume (0)
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant", constant_values=pad_value)
            
            # FIXED: Pad with -100 so CrossEntropyLoss ignores these frames
            labs = [np.pad(head, (0, pad), mode="constant", constant_values=-100) for head in labs]

        feature_chunks.append(feat)
        label_chunks.append(labs)

    return feature_chunks, label_chunks


def _cache_file_path(cfg: PreprocessingConfig) -> Path:
    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = {
        "dataset_root": str(Path(cfg.dataset_root).resolve()),
        "sample_rate": cfg.sample_rate,
        "hop_length": cfg.hop_length,
        "fmin_note": cfg.fmin_note,
        "n_bins": cfg.n_bins,
        "bins_per_octave": cfg.bins_per_octave,
        "segment_seconds": cfg.segment_seconds,
        "max_songs": cfg.max_songs,
    }
    digest = hashlib.md5(str(sorted(key.items())).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"bello_{digest}.pt"


class BelloChordFormerDataset(Dataset):
    def __init__(
        self, 
        file_pairs: List[Tuple[str, str]], 
        cfg: Optional[PreprocessingConfig] = None,
        augment: bool = False,
        split_name: str = "train"
    ):
        super().__init__()
        self.cfg = cfg or PreprocessingConfig()
        self.augment = augment
        self.split_name = split_name

        segment_frames = int(round(self.cfg.segment_seconds * self.cfg.sample_rate / self.cfg.hop_length))
        if segment_frames <= 0:
            raise ValueError("segment_seconds must produce at least 1 frame")

        # We append the split name to the cache file so train/val/test don't overwrite each other
        base_cache_path = _cache_file_path(self.cfg)
        cache_path = base_cache_path.parent / f"{base_cache_path.stem}_{split_name}.pt"

        if self.cfg.use_cache and cache_path.exists() and not self.cfg.refresh_cache:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.features = payload["features"]
            self.targets = payload["targets"]
            print(f"Loaded cached {split_name} dataset: {len(self.features)} segments")
            return

        self.features: List[torch.Tensor] = []
        self.targets: List[List[torch.Tensor]] = []

        print(f"Building {split_name} dataset...")
        for audio_path, lab_path in file_pairs:
            cqt = compute_cqt(audio_path, self.cfg)
            intervals = load_chord_intervals(lab_path)
            labels = align_intervals_to_frames(intervals, cqt.shape[0], self.cfg)
            feat_chunks, label_chunks = _chunk_song(cqt, labels, segment_frames)

            for feat, labs in zip(feat_chunks, label_chunks):
                self.features.append(torch.tensor(feat, dtype=torch.float32))
                self.targets.append([torch.tensor(x, dtype=torch.long) for x in labs])

        if self.cfg.use_cache:
            payload = {
                "features": self.features,
                "targets": self.targets,
            }
            torch.save(payload, cache_path)
            print(f"Saved {split_name} dataset cache: {len(self.features)} segments")

    def __len__(self) -> int:
        return len(self.features)

    def _shift_cqt(self, cqt: torch.Tensor, shift_bins: int) -> torch.Tensor:
        """Shifts the CQT tensor along the frequency axis and pads with the minimum value."""
        if shift_bins == 0:
            return cqt
            
        shifted = torch.full_like(cqt, fill_value=cqt.min())
        if shift_bins > 0:
            # Shift UP (Drop top bins, pad bottom)
            shifted[:, shift_bins:] = cqt[:, :-shift_bins]
        else:
            # Shift DOWN (Drop bottom bins, pad top)
            shift_bins = abs(shift_bins)
            shifted[:, :-shift_bins] = cqt[:, shift_bins:]
            
        return shifted

    def _transpose_labels(self, targets: List[torch.Tensor], semitone_shift: int) -> List[torch.Tensor]:
        """Mathematically transposes the Root/Triad (Head 0) and Bass (Head 1) labels."""
        if semitone_shift == 0:
            return targets

        # Copy the targets so we don't accidentally modify the cached tensors in memory
        transposed_targets = [t.clone() for t in targets]
        
        # --- 1. Transpose Root/Triad (Head 0) ---
        # Encoding: 1 + root_pc * 7 + triad_idx
        root_triad = transposed_targets[0]
        mask_0 = root_triad > 0  # Ignore "N" class (0)
        if mask_0.any():
            val = root_triad[mask_0] - 1
            pc = val // 7
            triad_idx = val % 7
            new_pc = (pc + semitone_shift) % 12
            transposed_targets[0][mask_0] = 1 + (new_pc * 7) + triad_idx

        # --- 2. Transpose Bass (Head 1) ---
        # Encoding: bass_pc + 1
        bass = transposed_targets[1]
        mask_1 = bass > 0 # Ignore "N" class (0)
        if mask_1.any():
            pc = bass[mask_1] - 1
            new_pc = (pc + semitone_shift) % 12
            transposed_targets[1][mask_1] = new_pc + 1

        # Heads 2, 3, 4, 5 (Extensions) do not change pitch class!
        return transposed_targets

    def __getitem__(self, idx: int):
        cqt = self.features[idx]
        targets = self.targets[idx]

        # Apply data augmentation exclusively if the flag is True
        if self.augment:
            semitone_shift = random.randint(-5, 6)
            if semitone_shift != 0:
                shift_bins = semitone_shift * self.cfg.bins_per_octave // 12
                cqt = self._shift_cqt(cqt, shift_bins)
                targets = self._transpose_labels(targets, semitone_shift)

        return cqt, targets

# --- Dataset Instantiation & Splitting Logic ---
def create_dataloaders(cfg: PreprocessingConfig, batch_size: int = 24, num_workers: int = 4):
    all_pairs = find_audio_lab_pairs(cfg.dataset_root)
    if cfg.max_songs is not None:
        all_pairs = all_pairs[:cfg.max_songs]
        
    # Song-level splitting (60% Train, 20% Val, 20% Test)
    random.seed(42)
    random.shuffle(all_pairs)
    
    total_songs = len(all_pairs)
    train_end = int(0.6 * total_songs)
    val_end = int(0.8 * total_songs)
    
    train_pairs = all_pairs[:train_end]
    val_pairs = all_pairs[train_end:val_end]
    test_pairs = all_pairs[val_end:]
    
    print(f"Song Split: {len(train_pairs)} Train | {len(val_pairs)} Val | {len(test_pairs)} Test")

    # Instantiate datasets (Only the training set gets augment=True)
    train_ds = BelloChordFormerDataset(train_pairs, cfg, augment=True, split_name="train")
    val_ds = BelloChordFormerDataset(val_pairs, cfg, augment=False, split_name="val")
    test_ds = BelloChordFormerDataset(test_pairs, cfg, augment=False, split_name="test")
    
    # FIXED: Added multiprocessing and pinned memory for WSL speed!
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dl, val_dl, test_dl

if __name__ == "__main__":
    cfg = PreprocessingConfig(dataset_root="bello_dataset", segment_seconds=10.0)
    train_dl, val_dl, test_dl = create_dataloaders(cfg)

    sample_x, sample_y = train_dl.dataset[0]
    print(f"Samples: {len(train_dl.dataset)}")
    print(f"Input shape: {sample_x.shape}")
    print(f"Head shapes: {[y.shape for y in sample_y]}")
    print(f"Output head dims: {[85, 13, 4, 4, 3, 3]}")
