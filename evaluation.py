"""
evaluation.py
=============
Industry-standard evaluation using the mir_eval library.

Implements:
  • WCSR (Weighted Chord Symbol Recall) across all standard metric tiers
  • Accepts the 6-head factorized model outputs (raw logits or argmax indices)
  • Compatible with both ChordFormer (Conformer) and Mini-Mambaformer branches

Standard metric tiers (mir_eval.chord vocabularies)
----------------------------------------------------
  root       – Root note only
  thirds     – Root + major/minor quality
  majmin     – Major/minor/no-chord
  triads     – Full triad (root + quality, no extensions)
  sevenths   – Triad + 7th extension
  tetrads    – Full four-note chords
  mirex      – MIREX 2013 vocabulary (major/minor only, enharmonic equivalence)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import mir_eval
    _MIR_EVAL_AVAILABLE = True
except ImportError:
    _MIR_EVAL_AVAILABLE = False
    warnings.warn(
        "mir_eval not installed.  Run `pip install mir_eval` to enable WCSR metrics.",
        ImportWarning,
        stacklevel=2,
    )

# CRF helpers live in chord_utils to avoid a circular import with train_unified.
from chord_utils import (
    decode_sequence,
    frames_to_mir_eval_format,
    viterbi_decode_crf,
)

METRIC_TIERS = ["root", "thirds", "majmin", "triads", "sevenths", "tetrads", "mirex"]


# ---------------------------------------------------------------------------
# Core WCSR computation for one sequence
# ---------------------------------------------------------------------------

def compute_wcsr_sequence(
    ref_strings: List[str],
    est_strings: List[str],
    hop_length:  int = 512,
    sample_rate: int = 22050,
) -> Dict[str, float]:
    """
    Compute WCSR for every metric tier for a single sequence.

    Parameters
    ----------
    ref_strings : ground-truth chord strings, one per frame
    est_strings : predicted chord strings, one per frame
    hop_length  : CQT hop length in samples (for converting frames → seconds)
    sample_rate : audio sample rate

    Returns
    -------
    dict mapping tier name → WCSR score in [0, 1]
    """
    if not _MIR_EVAL_AVAILABLE:
        return {t: float("nan") for t in METRIC_TIERS}

    # An explicitly empty sequence scores 0.0 on every tier rather than being
    # silently dropped — ensures the denominator in downstream averaging is correct.
    if not ref_strings or not est_strings:
        return {t: 0.0 for t in METRIC_TIERS}

    ref_intervals, ref_labels = frames_to_mir_eval_format(
        ref_strings, hop_length, sample_rate
    )
    est_intervals, est_labels = frames_to_mir_eval_format(
        est_strings, hop_length, sample_rate
    )

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return {t: 0.0 for t in METRIC_TIERS}

    scores: Dict[str, float] = {}
    for tier in METRIC_TIERS:
        try:
            comparison_fn = getattr(mir_eval.chord, tier)
            score = mir_eval.chord.weighted_accuracy(
                comparison_fn,
                ref_intervals,
                ref_labels,
                est_intervals,
                est_labels,
            )
        except Exception:
            score = 0.0
        scores[tier] = float(score)

    return scores


# ---------------------------------------------------------------------------
# Batch WCSR — processes a whole dataset split
# ---------------------------------------------------------------------------

def compute_wcsr_dataset(
    model:        torch.nn.Module,
    dataloader:   torch.utils.data.DataLoader,
    device:       torch.device,
    loss_fn:      torch.nn.Module,
    crf_penalty:  Optional[float] = None,
    hop_length:   int             = 512,
    sample_rate:  int             = 22050,
    ignore_index: int             = -100,
) -> Tuple[float, Dict[str, float]]:
    """
    Run inference over *dataloader* and return:
      (avg_loss,  {tier: mean_wcsr_across_sequences})

    Each segment in a batch is treated as an independent sequence for WCSR.
    Padding frames (label == ignore_index) are masked out before conversion.
    Sequences that are entirely padding score 0.0 on every tier so they are
    counted in the denominator rather than silently dropped.
    """
    model.eval()
    total_loss = 0.0

    all_tier_scores: Dict[str, List[float]] = {t: [] for t in METRIC_TIERS}

    with torch.no_grad():
        for batch in dataloader:
            cqt_segments, target_labels = batch
            cqt_segments  = cqt_segments.to(device)
            target_labels = [lbl.to(device) for lbl in target_labels]

            predictions = model(cqt_segments)
            loss        = loss_fn(predictions, target_labels)
            total_loss += loss.item()

            batch_size = cqt_segments.shape[0]

            # Optionally apply CRF Viterbi smoothing per head
            if crf_penalty is not None:
                decoded_preds = [
                    viterbi_decode_crf(predictions[h], penalty=crf_penalty)
                    for h in range(6)
                ]
            else:
                decoded_preds = [
                    torch.argmax(predictions[h], dim=-1)
                    for h in range(6)
                ]

            for b in range(batch_size):
                seq_len = cqt_segments.shape[1]

                # Build a validity mask: frames where ALL targets are not padding
                valid_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
                for h in range(6):
                    valid_mask &= (target_labels[h][b] != ignore_index)

                valid_mask_np = valid_mask.cpu().numpy()

                # Entirely-padding segment: score 0.0 on all tiers and count it.
                if not valid_mask_np.any():
                    for tier in METRIC_TIERS:
                        all_tier_scores[tier].append(0.0)
                    continue

                ref_heads = [
                    target_labels[h][b].cpu().numpy()[valid_mask_np]
                    for h in range(6)
                ]
                ref_strings = decode_sequence(*ref_heads)

                est_heads = [
                    decoded_preds[h][b].cpu().numpy()[valid_mask_np]
                    for h in range(6)
                ]
                est_strings = decode_sequence(*est_heads)

                seq_scores = compute_wcsr_sequence(
                    ref_strings, est_strings, hop_length, sample_rate
                )
                for tier, score in seq_scores.items():
                    all_tier_scores[tier].append(score)

    avg_loss  = total_loss / max(len(dataloader), 1)
    mean_wcsr = {
        tier: float(np.mean(scores)) if scores else float("nan")
        for tier, scores in all_tier_scores.items()
    }

    return avg_loss, mean_wcsr


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_wcsr_report(
    mean_wcsr:  Dict[str, float],
    model_name: str   = "Model",
    fold:       int   = 0,
    avg_loss:   float = float("nan"),
) -> None:
    """Print a formatted WCSR report to stdout."""
    fold_str = f"Fold {fold}" if fold > 0 else "All Folds"
    print(f"\n{'='*55}")
    print(f"  WCSR Report — {model_name}  [{fold_str}]")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"{'='*55}")
    print(f"  {'Tier':<12}  {'WCSR':>8}")
    print(f"  {'-'*22}")
    for tier in METRIC_TIERS:
        score     = mean_wcsr.get(tier, float("nan"))
        score_str = f"{score:.4f}" if not np.isnan(score) else "  n/a "
        print(f"  {tier:<12}  {score_str:>8}")
    print(f"{'='*55}\n")
