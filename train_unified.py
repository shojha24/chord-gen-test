"""
train_unified.py
================
Unified, modular training and evaluation pipeline for the architectural
comparison between ChordFormer (Conformer) and Mini-Mambaformer.

Features
--------
  • Selects model via --model {chordformer | mambaformer}
  • Loads pre-generated fold JSON files (fold_1_splits.json … fold_5_splits.json)
  • Identical optimizer, loss function, LR scheduler, and CRF post-processing
    for both architectures
  • Proper weight/optimizer/scheduler re-initialization between folds
  • Best-checkpoint tracking: only the single lowest-val-loss epoch is kept on
    disk per fold; final WCSR evaluation reloads that checkpoint
  • WCSR evaluation (mir_eval) across 7 standard metric tiers
  • Final CSV + Markdown report with mean ± std and full per-fold tables

Usage
-----
    # Generate splits first (only needs to run once)
    python generate_splits.py --dataset_root bello_dataset --output_dir splits

    # Train and evaluate ChordFormer across all 5 folds
    python train_unified.py --model chordformer --splits_dir splits

    # Train and evaluate Mini-Mambaformer across all 5 folds
    python train_unified.py --model mambaformer --splits_dir splits
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from evaluation import (
    METRIC_TIERS,
    compute_wcsr_dataset,
    print_wcsr_report,
)
from preprocessing import (
    BelloChordFormerDataset,
    PreprocessingConfig,
)
# CRF helpers live in chord_utils (not here) to avoid a circular import
# with evaluation.py.  Re-export for callers that expect them here.
from chord_utils import build_crf_transition_matrix, viterbi_decode_crf  # noqa: F401


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    """
    Instantiate and return the requested model with Xavier-uniform
    parameter initialisation.  Called once per fold — always fresh weights.
    """
    if model_name == "chordformer":
        from chordformer_model import build_chordformer  # type: ignore
        return build_chordformer()
    elif model_name == "mambaformer":
        from mambaformer_model import build_chordformer as build_mambaformer  # type: ignore
        return build_mambaformer()
    else:
        raise ValueError(
            f"Unknown model '{model_name}'.  Choose 'chordformer' or 'mambaformer'."
        )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_fold_pairs(
    fold_path:    str,
    dataset_root: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load a fold JSON and return (train_pairs, val_pairs, test_pairs)."""
    with open(fold_path, "r", encoding="utf-8") as fh:
        fold = json.load(fh)

    lab_map: Dict[str, str] = fold["lab_map"]

    def to_pairs(audio_list: List[str]) -> List[Tuple[str, str]]:
        pairs = []
        for audio_path in audio_list:
            lab_path = lab_map.get(audio_path)
            if lab_path is None:
                stem     = Path(audio_path).stem
                lab_path = str(Path(dataset_root) / "chordlab" / f"{stem}.lab")
            pairs.append((audio_path, lab_path))
        return pairs

    return to_pairs(fold["train"]), to_pairs(fold["val"]), to_pairs(fold["test"])


def build_dataloaders(
    train_pairs: List[Tuple[str, str]],
    val_pairs:   List[Tuple[str, str]],
    test_pairs:  List[Tuple[str, str]],
    cfg:         PreprocessingConfig,
    batch_size:  int = 48,
    num_workers: int = 4,
    fold_num:    int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for one fold using fold-aware cache names."""
    train_ds = BelloChordFormerDataset(
        train_pairs, cfg, augment=True,  split_name=f"fold{fold_num}_train"
    )
    val_ds = BelloChordFormerDataset(
        val_pairs,   cfg, augment=False, split_name=f"fold{fold_num}_val"
    )
    test_ds = BelloChordFormerDataset(
        test_pairs,  cfg, augment=False, split_name=f"fold{fold_num}_test"
    )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

OUTPUT_DIMS = [85, 13, 4, 4, 3, 3]


class ChordFormerLoss(nn.Module):
    """Weighted multi-head cross-entropy loss, identical for both branches."""

    def __init__(
        self,
        class_weights: Optional[List[torch.Tensor]] = None,
        ignore_index:  int = -100,
    ):
        super().__init__()
        self.loss_functions = nn.ModuleList()
        if class_weights:
            for w in class_weights:
                self.loss_functions.append(
                    nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
                )
        else:
            for _ in range(6):
                self.loss_functions.append(
                    nn.CrossEntropyLoss(ignore_index=ignore_index)
                )

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets:     List[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            pred_flat   = pred.view(-1, pred.size(-1))
            target_flat = target.view(-1)
            total_loss  = total_loss + self.loss_functions[i](pred_flat, target_flat)
        return total_loss


def compute_class_weights(
    dataset,
    output_dims: List[int] = OUTPUT_DIMS,
    gamma:       float     = 0.5,
    w_max:       float     = 10.0,
    eps:         float     = 1e-6,
) -> List[torch.Tensor]:
    """
    Compute per-class weights using the ChordFormer paper formula:
        w = min( (n_m / max_n)^(-gamma), w_max )
    """
    print(f"  Computing class weights (gamma={gamma}, w_max={w_max}) …")
    counts = [torch.zeros(dim, dtype=torch.float64) for dim in output_dims]

    for _, labels in dataset:
        for i, head_labels in enumerate(labels):
            flat  = head_labels.view(-1)
            valid = flat[flat >= 0]
            counts[i] += torch.bincount(valid, minlength=output_dims[i]).to(torch.float64)

    weights: List[torch.Tensor] = []
    for count in counts:
        max_c = count.max()
        if max_c == 0:
            weights.append(torch.ones(len(count), dtype=torch.float32))
            continue
        ratio = torch.clamp(count / max_c, min=eps)
        w     = torch.clamp(ratio ** (-gamma), max=w_max)
        weights.append(w.to(torch.float32))

    return weights


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def run_validation(
    model:     nn.Module,
    val_dl:    DataLoader,
    device:    torch.device,
    loss_fn:   nn.Module,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="  Validation", leave=False):
            cqt, targets = batch
            cqt     = cqt.to(device)
            targets = [t.to(device) for t in targets]
            total  += loss_fn(model(cqt), targets).item()
    return total / max(len(val_dl), 1)


# ---------------------------------------------------------------------------
# Single-fold training loop
# ---------------------------------------------------------------------------

def train_one_fold(
    model_name:  str,
    fold_num:    int,
    train_dl:    DataLoader,
    val_dl:      DataLoader,
    test_dl:     DataLoader,
    device:      torch.device,
    cfg:         PreprocessingConfig,
    max_epochs:  int   = 200,
    lr:          float = 1e-3,
    use_weights: bool  = True,
    crf_penalty: float = 2.0,
    exp_dir:     str   = "runs",
) -> Dict[str, float]:
    """
    Train one fold from scratch and return the WCSR scores on the test set.

    Model weights, optimizer state, and scheduler state are all freshly
    created inside this function — no state leaks between folds.

    Only the single best-validation-loss checkpoint is kept on disk;
    all other epoch checkpoints are deleted as training progresses.
    Final WCSR evaluation reloads that best checkpoint, not the last epoch.
    """
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_num}  |  Model: {model_name}")
    print(f"{'─'*60}")

    # ---- Fresh model --------------------------------------------------------
    model = build_model(model_name).to(device)

    # ---- Class weights (computed from training split) -----------------------
    class_weights = None
    if use_weights:
        class_weights = compute_class_weights(train_dl.dataset)
        class_weights = [w.to(device) for w in class_weights]

    # ---- Loss, optimiser, scheduler (identical for both architectures) ------
    loss_fn   = ChordFormerLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # ---- TensorBoard --------------------------------------------------------
    writer = SummaryWriter(
        log_dir=str(Path(exp_dir) / model_name / f"fold_{fold_num}")
    )

    global_step    = 0
    best_val_loss  = float("inf")
    best_ckpt_path = None
    ckpt_dir       = Path("models") / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_dl, desc=f"  Epoch {epoch+1:>3}/{max_epochs}", leave=False)
        for batch in pbar:
            cqt, targets = batch
            cqt     = cqt.to(device)
            targets = [t.to(device) for t in targets]

            loss = loss_fn(model(cqt), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            epoch_loss  += loss.item()
            n_batches   += 1

        # ---- Validation -----------------------------------------------------
        val_loss = run_validation(model, val_dl, device, loss_fn)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.flush()

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        is_best    = val_loss < best_val_loss

        print(
            f"  Epoch {epoch+1:>3}  "
            f"train_loss={epoch_loss/max(n_batches,1):.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}"
            + ("  ✓ best" if is_best else "")
        )

        # ---- Checkpoint: keep only the best, delete the previous one --------
        if is_best:
            if best_ckpt_path is not None:
                prev = Path(best_ckpt_path)
                if prev.exists():
                    prev.unlink()

            best_val_loss  = val_loss
            best_ckpt_path = str(
                ckpt_dir / f"{model_name}_fold{fold_num}_best.pt"
            )
            torch.save(model.state_dict(), best_ckpt_path)

        writer.add_scalar("Val/best_loss", best_val_loss, epoch)

        # ---- Early stopping -------------------------------------------------
        if current_lr < 1e-6:
            print(f"  Early stopping: LR < 1e-6  (epoch {epoch+1})")
            break

    writer.close()

    # ---- Reload best checkpoint before WCSR evaluation ---------------------
    print(
        f"\n  Reloading best checkpoint (val_loss={best_val_loss:.4f}): "
        f"{best_ckpt_path}"
    )
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # ---- Final WCSR evaluation on test set ----------------------------------
    print("  Running final WCSR evaluation on test set …")
    test_loss, wcsr_scores = compute_wcsr_dataset(
        model       = model,
        dataloader  = test_dl,
        device      = device,
        loss_fn     = loss_fn,
        crf_penalty = crf_penalty,
        hop_length  = cfg.hop_length,
        sample_rate = cfg.sample_rate,
    )
    print_wcsr_report(wcsr_scores, model_name=model_name, fold=fold_num, avg_loss=test_loss)

    return wcsr_scores


# ---------------------------------------------------------------------------
# Cross-validation orchestrator
# ---------------------------------------------------------------------------

def run_cross_validation(
    model_name:   str,
    splits_dir:   str,
    dataset_root: str,
    cfg:          PreprocessingConfig,
    device:       torch.device,
    n_folds:      int   = 5,
    batch_size:   int   = 48,
    num_workers:  int   = 4,
    max_epochs:   int   = 200,
    lr:           float = 1e-3,
    use_weights:  bool  = True,
    crf_penalty:  float = 2.0,
    exp_dir:      str   = "runs",
) -> List[Dict[str, float]]:
    """Iterate through all fold JSON files, train + evaluate, return per-fold scores."""
    all_fold_scores: List[Dict[str, float]] = []

    for fold_num in range(1, n_folds + 1):
        fold_path = Path(splits_dir) / f"fold_{fold_num}_splits.json"
        if not fold_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {fold_path}\n"
                f"Run generate_splits.py first."
            )

        train_pairs, val_pairs, test_pairs = load_fold_pairs(
            str(fold_path), dataset_root
        )
        print(
            f"\nFold {fold_num}: "
            f"{len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test"
        )

        train_dl, val_dl, test_dl = build_dataloaders(
            train_pairs, val_pairs, test_pairs,
            cfg        = cfg,
            batch_size = batch_size,
            num_workers= num_workers,
            fold_num   = fold_num,
        )

        fold_scores = train_one_fold(
            model_name  = model_name,
            fold_num    = fold_num,
            train_dl    = train_dl,
            val_dl      = val_dl,
            test_dl     = test_dl,
            device      = device,
            cfg         = cfg,
            max_epochs  = max_epochs,
            lr          = lr,
            use_weights = use_weights,
            crf_penalty = crf_penalty,
            exp_dir     = exp_dir,
        )
        all_fold_scores.append(fold_scores)

        torch.cuda.empty_cache()

    return all_fold_scores


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(
    all_fold_scores: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Return {tier: {mean, std, min, max}} computed over all folds."""
    aggregated: Dict[str, Dict[str, float]] = {}
    for tier in METRIC_TIERS:
        vals = [
            s[tier] for s in all_fold_scores
            if not np.isnan(s.get(tier, float("nan")))
        ]
        if vals:
            aggregated[tier] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=0)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
            }
        else:
            aggregated[tier] = {
                "mean": float("nan"), "std": float("nan"),
                "min":  float("nan"), "max": float("nan"),
            }
    return aggregated


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_comparison_report(
    results:          Dict[str, Dict[str, Dict[str, float]]],
    all_fold_scores:  Dict[str, List[Dict[str, float]]],
    output_dir:       str = ".",
) -> Tuple[str, str]:
    """
    Save a side-by-side CSV and Markdown comparison report.

    Parameters
    ----------
    results         : {model_name: {tier: {mean, std, min, max}}}
    all_fold_scores : {model_name: [fold_1_scores, fold_2_scores, ...]}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    n_folds     = max(len(v) for v in all_fold_scores.values())

    # ---- CSV ----------------------------------------------------------------
    csv_path = out / "wcsr_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        header = ["Tier"]
        for name in model_names:
            header += [f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max"]
            for k in range(1, n_folds + 1):
                header.append(f"{name}_fold{k}")
        writer.writerow(header)

        for tier in METRIC_TIERS:
            row = [tier]
            for name in model_names:
                stats = results[name].get(tier, {})
                row += [
                    f"{stats.get('mean', float('nan')):.4f}",
                    f"{stats.get('std',  float('nan')):.4f}",
                    f"{stats.get('min',  float('nan')):.4f}",
                    f"{stats.get('max',  float('nan')):.4f}",
                ]
                for fold_scores in all_fold_scores.get(name, []):
                    row.append(f"{fold_scores.get(tier, float('nan')):.4f}")
            writer.writerow(row)

    # ---- Markdown -----------------------------------------------------------
    md_path = out / "wcsr_comparison.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# ChordFormer vs Mini-Mambaformer — WCSR Comparison\n\n")
        fh.write(
            "Scores are **Weighted Chord Symbol Recall (WCSR)** computed with "
            "`mir_eval` across 5-fold cross-validation.  "
            "Values reported as **mean ± std** (min / max).\n\n"
        )

        # ---- Summary table --------------------------------------------------
        cols = ["Tier"] + [f"{n} (mean ± std)" for n in model_names]
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "|".join(["---"] * len(cols)) + "|\n")

        for tier in METRIC_TIERS:
            row_parts = [f"**{tier}**"]
            for name in model_names:
                stats = results[name].get(tier, {})
                mean  = stats.get("mean", float("nan"))
                std   = stats.get("std",  float("nan"))
                lo    = stats.get("min",  float("nan"))
                hi    = stats.get("max",  float("nan"))
                if np.isnan(mean):
                    row_parts.append("n/a")
                else:
                    row_parts.append(
                        f"{mean:.4f} ± {std:.4f} ({lo:.4f} / {hi:.4f})"
                    )
            fh.write("| " + " | ".join(row_parts) + " |\n")

        # ---- Per-fold tables ------------------------------------------------
        fh.write("\n## Per-fold scores\n\n")
        for name in model_names:
            fh.write(f"### {name}\n\n")
            fold_headers = ["Tier"] + [f"Fold {k+1}" for k in range(n_folds)] + ["Mean", "Std"]
            fh.write("| " + " | ".join(fold_headers) + " |\n")
            fh.write("|" + "|".join(["---"] * len(fold_headers)) + "|\n")

            for tier in METRIC_TIERS:
                fold_vals = [
                    fs.get(tier, float("nan"))
                    for fs in all_fold_scores.get(name, [])
                ]
                valid = [v for v in fold_vals if not np.isnan(v)]
                mean_str = f"{np.mean(valid):.4f}" if valid else "n/a"
                std_str  = f"{np.std(valid, ddof=0):.4f}" if valid else "n/a"
                cells    = [
                    f"{v:.4f}" if not np.isnan(v) else "n/a"
                    for v in fold_vals
                ]
                fh.write(
                    "| " + " | ".join([f"**{tier}**"] + cells + [mean_str, std_str]) + " |\n"
                )
            fh.write("\n")

    print(f"\nReports saved:\n  CSV      → {csv_path}\n  Markdown → {md_path}")
    return str(csv_path), str(md_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified 5-fold CV training pipeline for ChordFormer comparison."
    )
    parser.add_argument(
        "--model",
        choices=["chordformer", "mambaformer"],
        required=True,
        help="Which model architecture to train.",
    )
    parser.add_argument("--splits_dir",      default="splits")
    parser.add_argument("--dataset_root",    default="bello_dataset")
    parser.add_argument("--n_folds",         type=int,   default=5)
    parser.add_argument("--max_epochs",      type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=48)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--crf_penalty",     type=float, default=2.0)
    parser.add_argument("--no_weights",      action="store_true",
                        help="Disable class re-weighting.")
    parser.add_argument("--exp_dir",         default="runs")
    parser.add_argument("--output_dir",      default="results")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = PreprocessingConfig(
        dataset_root    = args.dataset_root,
        segment_seconds = args.segment_seconds,
        use_cache       = True,
        refresh_cache   = False,
        cache_dir       = ".cache/chordformer",
    )

    # ---- Run 5-fold CV ------------------------------------------------------
    all_fold_scores = run_cross_validation(
        model_name   = args.model,
        splits_dir   = args.splits_dir,
        dataset_root = args.dataset_root,
        cfg          = cfg,
        device       = device,
        n_folds      = args.n_folds,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        max_epochs   = args.max_epochs,
        lr           = args.lr,
        use_weights  = not args.no_weights,
        crf_penalty  = args.crf_penalty,
        exp_dir      = args.exp_dir,
    )

    # ---- Aggregate ----------------------------------------------------------
    aggregated = aggregate_scores(all_fold_scores)

    print(f"\n{'='*60}")
    print(f"  FINAL AGGREGATE  ({args.model}, {args.n_folds} folds)")
    print(f"{'='*60}")
    for tier in METRIC_TIERS:
        stats = aggregated[tier]
        print(
            f"  {tier:<12}  "
            f"mean={stats['mean']:.4f}  "
            f"std={stats['std']:.4f}  "
            f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        )

    # ---- Save per-model results to JSON -------------------------------------
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.model}_cv_results.json"

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model":       args.model,
                "n_folds":     args.n_folds,
                "fold_scores": all_fold_scores,
                "aggregated":  aggregated,
            },
            fh,
            indent=2,
        )
    print(f"\nPer-model results saved → {results_path}")

    # ---- Generate combined report if both models have results ---------------
    other_model = "mambaformer" if args.model == "chordformer" else "chordformer"
    other_path  = out_dir / f"{other_model}_cv_results.json"

    if other_path.exists():
        print(f"\nFound results for both models — generating comparison report …")
        with open(other_path, "r", encoding="utf-8") as fh:
            other_payload = json.load(fh)

        save_comparison_report(
            results={
                args.model: aggregated,
                other_model: other_payload["aggregated"],
            },
            all_fold_scores={
                args.model: all_fold_scores,
                other_model: other_payload["fold_scores"],
            },
            output_dir=str(out_dir),
        )
    else:
        print(
            f"\nRun with --model {other_model} next to get the other model's results, "
            f"after which the combined comparison report will be generated automatically."
        )


if __name__ == "__main__":
    main()
