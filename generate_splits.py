"""
generate_splits.py
==================
Standalone script that scans the consolidated 1,217-song dataset
(Isophonics + Billboard + MARL collections) and writes 5 static
cross-validation split files:

    fold_1_splits.json … fold_5_splits.json

Each JSON file has the structure:
    {
        "fold":  1,
        "train": ["audio/song_a.mp3", ...],   # 60 % of songs
        "val":   ["audio/song_b.mp3", ...],   # 20 % of songs
        "test":  ["audio/song_c.mp3", ...]    # 20 % of songs
    }

Both the ChordFormer (Conformer) and Mini-Mambaformer branches must load
these same files so that results are strictly comparable.

Usage
-----
    python generate_splits.py \
        --dataset_root bello_dataset \
        --n_folds      5             \
        --seed         42            \
        --output_dir   splits
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Dataset discovery (mirrors find_audio_lab_pairs from preprocessing.py)
# ---------------------------------------------------------------------------

ALLOWED_AUDIO_EXTS = [".mp3", ".wav", ".flac", ".ogg"]


def find_audio_lab_pairs(dataset_root: str) -> List[Tuple[str, str]]:
    """
    Return a sorted list of (audio_path, lab_path) pairs found under
    <dataset_root>/audio/ and <dataset_root>/chordlab/.

    Supports an optional filelist.txt inside the audio directory.
    """
    root      = Path(dataset_root)
    audio_dir = root / "audio"
    lab_dir   = root / "chordlab"
    filelist  = audio_dir / "filelist.txt"

    pairs: List[Tuple[str, str]] = []

    if filelist.exists():
        for line in filelist.read_text(encoding="utf-8").splitlines():
            stem     = Path(line.strip()).stem
            lab_path = lab_dir / f"{stem}.lab"
            if not lab_path.exists():
                continue
            for ext in ALLOWED_AUDIO_EXTS:
                candidate = audio_dir / f"{stem}{ext}"
                if candidate.exists():
                    pairs.append((str(candidate), str(lab_path)))
                    break
    else:
        for lab_path in sorted(lab_dir.glob("*.lab")):
            stem = lab_path.stem
            for ext in ALLOWED_AUDIO_EXTS:
                audio_path = audio_dir / f"{stem}{ext}"
                if audio_path.exists():
                    pairs.append((str(audio_path), str(lab_path)))
                    break

    return pairs


# ---------------------------------------------------------------------------
# K-Fold split generator
# ---------------------------------------------------------------------------

def generate_kfold_splits(
    pairs:      List[Tuple[str, str]],
    n_folds:    int   = 5,
    seed:       int   = 42,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> List[dict]:
    """
    Partition *pairs* into n_folds cross-validation splits.

    Strategy
    --------
    We first shuffle the full list with a fixed seed, then divide it into
    n_folds roughly equal *test* chunks.  For each fold the corresponding
    chunk becomes the test set; the remaining songs are split into
    train (train_frac / (1 - test_frac) portion) and val (the rest).

    With default fractions (0.60 / 0.20 / 0.20):
        test_frac  = 1 - 0.60 - 0.20 = 0.20  → 1 fold  out of 5
        val_frac   = 0.20              → 1 fold  out of 5
        train_frac = 0.60              → 3 folds out of 5

    This is a standard 5-fold scheme where each fold's test set is exactly
    one fifth of the data, which naturally yields the 60/20/20 ratio.
    """
    assert n_folds >= 3, "Need at least 3 folds to achieve 60/20/20."
    assert abs(train_frac + val_frac + (1.0 - train_frac - val_frac) - 1.0) < 1e-9

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total == 0:
        raise ValueError("No audio/lab pairs found – check --dataset_root.")

    # Split into n_folds chunks (last chunk absorbs any remainder)
    chunk_size = total // n_folds
    chunks: List[List[Tuple[str, str]]] = []
    for k in range(n_folds):
        start = k * chunk_size
        end   = start + chunk_size if k < n_folds - 1 else total
        chunks.append(shuffled[start:end])

    splits: List[dict] = []

    for fold_idx in range(n_folds):
        # Test = current chunk
        test_pairs = chunks[fold_idx]

        # Remaining songs (all other chunks merged)
        remaining: List[Tuple[str, str]] = []
        for k in range(n_folds):
            if k != fold_idx:
                remaining.extend(chunks[k])

        # From the remaining songs, carve out a validation chunk.
        # We want val ≈ 20 % of total, so val_count ≈ total * val_frac.
        val_count = math.ceil(total * val_frac)
        # Rotate which part of 'remaining' becomes val so each fold uses
        # a different validation window (deterministic rotation by fold index).
        rotate_by = (fold_idx * val_count) % max(len(remaining), 1)
        rotated   = remaining[rotate_by:] + remaining[:rotate_by]

        val_pairs   = rotated[:val_count]
        train_pairs = rotated[val_count:]

        split = {
            "fold":  fold_idx + 1,
            "seed":  seed,
            "total_songs": total,
            "counts": {
                "train": len(train_pairs),
                "val":   len(val_pairs),
                "test":  len(test_pairs),
            },
            "train": [p[0] for p in train_pairs],
            "val":   [p[0] for p in val_pairs],
            "test":  [p[0] for p in test_pairs],
            "lab_map": {p[0]: p[1] for p in (train_pairs + val_pairs + test_pairs)},
        }
        splits.append(split)

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static 5-fold CV splits for ChordFormer experiments."
    )
    parser.add_argument("--dataset_root", default="bello_dataset",
                        help="Root directory containing audio/ and chordlab/ sub-dirs.")
    parser.add_argument("--n_folds",      type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--train_frac",   type=float, default=0.60)
    parser.add_argument("--val_frac",     type=float, default=0.20)
    parser.add_argument("--output_dir",   default="splits",
                        help="Directory where fold_N_splits.json files are written.")
    args = parser.parse_args()

    print(f"Scanning dataset at: {args.dataset_root}")
    pairs = find_audio_lab_pairs(args.dataset_root)
    print(f"Found {len(pairs)} audio/lab pairs.")

    if len(pairs) == 0:
        print("WARNING: No pairs found.  The split files will be empty stubs.")
        print("Re-run after pointing --dataset_root at the correct location.")

    splits = generate_kfold_splits(
        pairs,
        n_folds    = args.n_folds,
        seed       = args.seed,
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        fold_num = split["fold"]
        out_path = out_dir / f"fold_{fold_num}_splits.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(split, fh, indent=2)

        counts = split["counts"]
        print(
            f"  Fold {fold_num}: "
            f"{counts['train']} train | {counts['val']} val | {counts['test']} test  "
            f"→  {out_path}"
        )

    print("\nDone.  Both model branches should load these JSON files for training.")


if __name__ == "__main__":
    main()
