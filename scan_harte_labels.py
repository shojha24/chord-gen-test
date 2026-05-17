"""
scan_harte_labels.py

Scans all .lab files in your dataset and prints:
  1. Every unique Harte label, sorted by frequency
  2. Every unique descriptor (the part after the colon)
  3. How each descriptor maps through _infer_triad and _extract_extension_states
  4. A comparison table against the ChordFormer paper's vocab

Run:
    python scan_harte_labels.py --dataset_root bello_dataset
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

# ── Paste or import your current vocab/parsers here ──────────────────────────
# (copied inline so this script is self-contained)

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


def _infer_triad_OLD(descriptor: str) -> str:
    d = descriptor.lower()
    if d in {"", "1"}:         return "maj"
    if d.startswith("sus2"):   return "sus2"
    if d.startswith("sus"):    return "sus4"
    if "dim" in d or "hdim" in d: return "dim"
    if "aug" in d or d.startswith("+") or "#5" in d: return "aug"
    if d.startswith("min") or ":min" in d: return "min"
    if d.startswith("maj") or re.match(r"^(7|9|11|13)", d): return "maj"
    if d.startswith("5"):      return "5"
    if d.startswith("("):
        has_b3 = "b3" in d
        has_b5 = "b5" in d
        has_3  = re.search(r"(^|[^b#])3", d) is not None
        if has_b3 and has_b5: return "dim"
        if has_b3:            return "min"
        if has_3:             return "maj"
        return "5"
    return "maj"


def _extract_extension_states_OLD(descriptor: str, triad: str) -> Tuple[str,str,str,str]:
    d = descriptor.lower()
    seventh = ninth = eleventh = thirteenth = "N"
    if "13" in d or "6" in d:
        if "maj13" in d:   seventh, ninth, thirteenth = "7",  "9", "13"
        else:              seventh, ninth, thirteenth = "b7", "9", "13"
    elif "11" in d:
        if "maj11" in d:   seventh, ninth, eleventh = "7",  "9", "11"
        else:              seventh, ninth, eleventh = "b7", "9", "11"
    elif "9" in d:
        if "maj9" in d:    seventh, ninth = "7",  "9"
        else:              seventh, ninth = "b7", "9"
    elif "7" in d or triad == "dim":
        if "maj7" in d:    seventh = "7"
        elif "dim7" in d and triad == "dim": seventh = "bb7"
        elif "7" in d:     seventh = "b7"
    matches = re.findall(r"(bb|b|#)?(7|9|11|13|6)", d)
    for acc, deg in matches:
        if deg == "7":
            if acc == "bb":  seventh = "bb7"
            elif acc == "b": seventh = "b7"
            elif acc == "" and "maj" in d: seventh = "7"
        elif deg == "9":
            if acc == "#":   ninth = "#9"
            elif acc == "b": ninth = "b9"
            else:            ninth = "9"
        elif deg == "11":
            eleventh = "#11" if acc == "#" else "11"
        elif deg in {"13","6"}:
            thirteenth = "b13" if acc == "b" else "13"
    return seventh, ninth, eleventh, thirteenth


# ── Scanner ───────────────────────────────────────────────────────────────────

def find_lab_files(dataset_root: str) -> list[Path]:
    root = Path(dataset_root)
    lab_dir = root / "chordlab"
    if not lab_dir.exists():
        raise FileNotFoundError(f"No chordlab/ directory found under {dataset_root}")
    return sorted(lab_dir.glob("*.lab"))


def collect_labels(lab_files: list[Path]) -> Counter:
    counts: Counter = Counter()
    for path in lab_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    counts[parts[2]] += 1
    return counts


def extract_descriptor(label: str) -> str:
    """Return just the quality descriptor from a Harte label."""
    if label in {"N", "X", ""}:
        return label
    body = label.split("/")[0]          # drop bass
    if ":" in body:
        return body.split(":", 1)[1]    # drop root
    return "maj"                        # bare root = major


def parse_descriptor(descriptor: str):
    """Run through current parser and return (triad, 7th, 9th, 11th, 13th)."""
    triad = _infer_triad_OLD(descriptor)
    s7, s9, s11, s13 = _extract_extension_states_OLD(descriptor, triad)
    return triad, s7, s9, s11, s13


# ── Paper vocab ───────────────────────────────────────────────────────────────
# Taken verbatim from Section III-A of the ChordFormer paper.
PAPER_TRIADS     = {"N", "major", "minor", "sus4", "sus2", "diminished", "augmented"}
PAPER_SEVENTHS   = {"N", "7", "b7", "bb7"}
PAPER_NINTHS     = {"N", "9", "#9", "b9"}
PAPER_ELEVENTHS  = {"N", "11", "#11"}
PAPER_THIRTEENTHS = {"N", "13", "b13"}

# Map paper triad names to our internal tokens
PAPER_TO_INTERNAL = {
    "major": "maj", "minor": "min",
    "diminished": "dim", "augmented": "aug",
    "sus4": "sus4", "sus2": "sus2", "N": "N",
}
INTERNAL_TO_PAPER = {v: k for k, v in PAPER_TO_INTERNAL.items()}
# hdim is NOT in the paper vocab — flag it
PAPER_MISSING_TRIADS = {"hdim", "5"}   # power chords also absent from paper


def compare_with_paper(triad_counts: Counter, extension_counts: dict) -> None:
    print("\n" + "═"*70)
    print("COMPARISON: Your implementation vs. ChordFormer paper vocab")
    print("═"*70)

    print("\n── Triad vocab ─────────────────────────────────────────────────────")
    print(f"  Paper defines:  {sorted(PAPER_TRIADS)}")
    print(f"  Your code has:  {sorted(ROOT_TRIAD_VOCAB + ['N'])}")
    our_internal = set(ROOT_TRIAD_VOCAB) | {"N"}
    paper_internal = set(PAPER_TO_INTERNAL.values())
    only_in_paper = paper_internal - our_internal
    only_in_ours  = our_internal - paper_internal
    if only_in_paper:
        print(f"  [!] In paper but NOT your code: {only_in_paper}")
    if only_in_ours:
        print(f"  [!] In your code but NOT paper:  {only_in_ours}")
        for t in sorted(only_in_ours):
            print(f"      '{t}' appears {triad_counts.get(t, 0)} times in dataset")

    for head_name, paper_set, our_list in [
        ("7th",  PAPER_SEVENTHS,    SEVENTH_VOCAB),
        ("9th",  PAPER_NINTHS,      NINTH_VOCAB),
        ("11th", PAPER_ELEVENTHS,   ELEVENTH_VOCAB),
        ("13th", PAPER_THIRTEENTHS, THIRTEENTH_VOCAB),
    ]:
        print(f"\n── {head_name} vocab ────────────────────────────────────────────────")
        our_set = set(our_list)
        only_paper = paper_set - our_set
        only_ours  = our_set - paper_set
        if not only_paper and not only_ours:
            print(f"  ✓ Match: {sorted(our_set)}")
        if only_paper:
            print(f"  [!] In paper not code: {only_paper}")
        if only_ours:
            print(f"  [!] In code not paper: {only_ours}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="bello_dataset")
    parser.add_argument("--top_n", type=int, default=200,
                        help="Print only the top N labels by frequency (0=all)")
    args = parser.parse_args(args)

    print(f"Scanning {args.dataset_root}/chordlab/ ...")
    lab_files = find_lab_files(args.dataset_root)
    print(f"Found {len(lab_files)} .lab files\n")

    label_counts = collect_labels(lab_files)
    total_frames = sum(label_counts.values())

    # ── 1. All unique labels ──────────────────────────────────────────────────
    print("═"*70)
    print(f"UNIQUE HARTE LABELS  (total unique: {len(label_counts)}, total frames: {total_frames:,})")
    print("═"*70)
    print(f"{'Label':<30} {'Count':>8}  {'%':>6}  {'Triad':<8} {'7th':<6} {'9th':<6} {'11th':<5} {'13th'}")
    print("─"*80)

    triad_counts: Counter = Counter()
    extension_cols = {"7th": Counter(), "9th": Counter(), "11th": Counter(), "13th": Counter()}

    top_labels = label_counts.most_common(args.top_n if args.top_n > 0 else None)
    for label, count in top_labels:
        pct = 100 * count / total_frames
        desc = extract_descriptor(label)
        if label in {"N", "X"}:
            triad, s7, s9, s11, s13 = "N", "N", "N", "N", "N"
        else:
            triad, s7, s9, s11, s13 = parse_descriptor(desc)
        triad_counts[triad] += count
        extension_cols["7th"][s7]   += count
        extension_cols["9th"][s9]   += count
        extension_cols["11th"][s11] += count
        extension_cols["13th"][s13] += count
        print(f"{label:<30} {count:>8,}  {pct:>5.2f}%  {triad:<8} {s7:<6} {s9:<6} {s11:<5} {s13}")

    if args.top_n and len(label_counts) > args.top_n:
        print(f"  ... ({len(label_counts) - args.top_n} more labels not shown)")

    # ── 2. Unique descriptors ─────────────────────────────────────────────────
    desc_counts: Counter = Counter()
    for label, count in label_counts.items():
        desc_counts[extract_descriptor(label)] += count

    print("\n" + "═"*70)
    print(f"UNIQUE DESCRIPTORS  (total: {len(desc_counts)})")
    print("═"*70)
    print(f"{'Descriptor':<25} {'Frames':>8}  {'Triad':<8} {'7th':<6} {'9th':<6} {'11th':<5} {'13th'}")
    print("─"*70)
    for desc, count in desc_counts.most_common():
        if desc in {"N", "X"}:
            row = "N", "N", "N", "N", "N"
        else:
            row = parse_descriptor(desc)
        print(f"{desc:<25} {count:>8,}  {row[0]:<8} {row[1]:<6} {row[2]:<6} {row[3]:<5} {row[4]}")

    # ── 3. Triad distribution ─────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("TRIAD CLASS DISTRIBUTION (after _infer_triad)")
    print("═"*70)
    for triad, count in triad_counts.most_common():
        pct = 100 * count / total_frames
        flag = " ← NOT IN PAPER VOCAB" if triad in PAPER_MISSING_TRIADS else ""
        print(f"  {triad:<8} {count:>8,}  {pct:>5.2f}%{flag}")

    # ── 4. Extension distribution ─────────────────────────────────────────────
    print("\n" + "═"*70)
    print("EXTENSION HEAD DISTRIBUTIONS")
    print("═"*70)
    for head, counter in extension_cols.items():
        print(f"\n  {head}:")
        for val, count in counter.most_common():
            pct = 100 * count / total_frames
            print(f"    {val:<6} {count:>8,}  {pct:>5.2f}%")

    # ── 5. Paper comparison ───────────────────────────────────────────────────
    compare_with_paper(triad_counts, extension_cols)

    # ── 6. Labels that hit unexpected/fallthrough paths ───────────────────────
    print("\n" + "═"*70)
    print("LABELS WITH POTENTIALLY AMBIGUOUS PARSING")
    print("═"*70)
    flagged = []
    for label, count in label_counts.items():
        desc = extract_descriptor(label)
        if desc in {"N", "X"}:
            continue
        d = desc.lower()
        # Heuristics for suspicious cases
        is_hdim = "hdim" in d
        is_add  = "add" in d
        is_69   = bool(re.match(r"^6/?9", d))
        is_maj6 = bool(re.match(r"^(maj6|6)(?![/9])", d))
        is_power = d.startswith("5")
        triad, s7, s9, s11, s13 = parse_descriptor(desc)
        # Flag: ninth present but seventh absent (impossible in standard harmony)
        impossible_ext = (s9 != "N" or s11 != "N" or s13 != "N") and s7 == "N"
        if is_hdim or is_add or is_69 or is_maj6 or impossible_ext:
            reason = []
            if is_hdim:       reason.append("hdim→dim collision")
            if is_add:        reason.append("add chord: 7th incorrectly cascaded?")
            if is_69 or is_maj6: reason.append("6th chord: 7th should be N")
            if impossible_ext: reason.append(f"extension without 7th: 7={s7} 9={s9}")
            flagged.append((label, count, triad, s7, s9, s11, s13, "; ".join(reason)))

    if not flagged:
        print("  None found.")
    else:
        print(f"{'Label':<28} {'Count':>7}  {'Triad':<8} {'7':>4} {'9':>5} {'11':>4} {'13':>4}  Issue")
        print("─"*90)
        for label, count, triad, s7, s9, s11, s13, reason in sorted(flagged, key=lambda x: -x[1]):
            print(f"{label:<28} {count:>7,}  {triad:<8} {s7:>4} {s9:>5} {s11:>4} {s13:>4}  {reason}")


if __name__ == "__main__":
    import sys
    out_path = Path("harte_label_scan.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        main(sys.argv[1:])
    sys.stdout = sys.__stdout__
    print(f"Written to {out_path.resolve()}")