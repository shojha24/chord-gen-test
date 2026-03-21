# Unified ChordFormer Evaluation Pipeline
 
A modular, reproducible training and evaluation framework for comparing
**ChordFormer** (Conformer-based) and **Mini-Mambaformer** (Mamba SSM-based)
automatic chord recognition architectures.
 
---
 
## What This Pipeline Does
 
| Capability | Detail |
|---|---|
| **Standard metric** | Weighted Chord Symbol Recall (WCSR) via `mir_eval` |
| **Metric tiers** | root, thirds, majmin, triads, sevenths, tetrads, mirex |
| **Validation scheme** | 5-fold cross-validation with **static, seed-locked splits** |
| **Data ratio (per fold)** | 60 % train / 20 % val / 20 % test |
| **Both models** | Identical optimizer, loss, LR scheduler, CRF post-processing |
| **Output** | Side-by-side CSV + Markdown report (mean ± std across 5 folds) |
 
---
 
## File Overview
 
```
chord_utils.py       – Shared vocabulary, 6-head → Harte chord string decoder,
                       and CRF Viterbi post-processing (viterbi_decode_crf)
evaluation.py        – mir_eval WCSR computation across all 7 metric tiers
generate_splits.py   – One-time script: writes fold_1_splits.json … fold_5_splits.json
train_unified.py     – Main pipeline: model selection, 5-fold CV, metric aggregation
```
 
Both model branches continue to use their own **`chordformer_model.py`** /
**`mambaformer_model.py`** and the **shared `preprocessing.py`** unchanged.
 
---
 
## Quick-Start
 
### 1. Install dependencies
 
```bash
pip install torch librosa mir_eval scikit-learn tensorboard tqdm mamba-ssm
```
 
### 2. Generate the static fold splits (run once)
 
```bash
python generate_splits.py \
    --dataset_root bello_dataset \
    --n_folds      5             \
    --seed         42            \
    --output_dir   splits
```
 
This writes `splits/fold_1_splits.json` … `splits/fold_5_splits.json`.
Both model branches **must** use these exact files to prevent data leakage.
 
Output looks like:
```
Found 1217 audio/lab pairs.
  Fold 1: 731 train | 244 val | 242 test  →  splits/fold_1_splits.json
  Fold 2: 731 train | 244 val | 242 test  →  splits/fold_2_splits.json
  ...
```
 
### 3. Train and evaluate ChordFormer
 
```bash
python train_unified.py \
    --model          chordformer \
    --splits_dir     splits      \
    --dataset_root   bello_dataset
```
 
### 4. Train and evaluate Mini-Mambaformer
 
```bash
python train_unified.py \
    --model          mambaformer \
    --splits_dir     splits      \
    --dataset_root   bello_dataset
```
 
After both have run, the pipeline automatically generates:
- `results/wcsr_comparison.csv`
- `results/wcsr_comparison.md`
 
---
 
## All CLI Flags (`train_unified.py`)
 
| Flag | Default | Description |
|---|---|---|
| `--model` | *required* | `chordformer` or `mambaformer` |
| `--splits_dir` | `splits` | Directory with fold JSON files |
| `--dataset_root` | `bello_dataset` | Root of consolidated corpus |
| `--n_folds` | `5` | Number of CV folds |
| `--max_epochs` | `200` | Per-fold epoch limit (early stopping fires first) |
| `--lr` | `1e-3` | Initial learning rate (AdamW) |
| `--batch_size` | `48` | Batch size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--segment_seconds` | `10.0` | CQT segment length |
| `--crf_penalty` | `2.0` | Viterbi CRF transition penalty |
| `--no_weights` | off | Disable class re-weighting |
| `--exp_dir` | `runs` | TensorBoard log root |
| `--output_dir` | `results` | Where CSV/Markdown reports are saved |
 
---
 
## Architecture of the Pipeline
 
```
generate_splits.py
    └─ Scans bello_dataset/
    └─ Shuffles with seed=42
    └─ Writes fold_N_splits.json (train/val/test audio paths + lab_map)
 
train_unified.py
    ├─ Loads fold_N_splits.json
    ├─ Builds BelloChordFormerDataset (fold-aware cache names)
    │
    ├─ FOR EACH FOLD (1 … 5):
    │   ├─ build_model()          ← fresh weights every fold
    │   ├─ compute_class_weights()
    │   ├─ ChordFormerLoss (weighted cross-entropy, 6 heads)
    │   ├─ AdamW + ReduceLROnPlateau (patience=5, factor=0.1)
    │   ├─ Training loop with gradient clipping (max_norm=1.0)
    │   ├─ Best-checkpoint tracking (lowest val loss kept, others deleted)
    │   ├─ Early stopping: lr < 1e-6
    │   └─ compute_wcsr_dataset() → WCSR per tier (best checkpoint reloaded first)
    │
    └─ aggregate_scores() → mean ± std across folds
    └─ save_comparison_report() → wcsr_comparison.{csv,md}
 
chord_utils.py
    └─ decode_frame(rt, bass, 7th, 9th, 11th, 13th) → "C:maj7"
    └─ frames_to_mir_eval_format() → (intervals, labels)
    └─ viterbi_decode_crf() / build_crf_transition_matrix()
       (kept here so evaluation.py can import them without a circular dependency)
 
evaluation.py
    └─ compute_wcsr_sequence()  ← single sequence
    └─ compute_wcsr_dataset()   ← full DataLoader
```
 
---
 
## Required File Layout
 
All seven Python files must live in the same directory:
 
```
your_project/
├── chord_utils.py          ← pipeline
├── evaluation.py           ← pipeline
├── generate_splits.py      ← pipeline
├── train_unified.py        ← pipeline
├── preprocessing.py        ← from either model branch (identical in both)
├── chordformer_model.py    ← from the Conformer branch
├── mambaformer_model.py    ← from the Mamba branch
├── bello_dataset/
│   ├── audio/
│   └── chordlab/
└── splits/                 ← created by generate_splits.py
```
 
---
 
## WCSR Metric Tiers
 
| Tier | What is compared |
|---|---|
| `root` | Root pitch class only |
| `thirds` | Root + major/minor quality |
| `majmin` | Major, minor, or no-chord |
| `triads` | Root + full triad quality |
| `sevenths` | Root + triad + 7th extension |
| `tetrads` | Full four-note chord |
| `mirex` | MIREX 2013 vocabulary (enharmonic equivalence) |
 
All tiers use `mir_eval.chord.weighted_accuracy`, which weights each
comparison by the *duration* of the reference segment — consistent with
standard MIR evaluation practice.
 
---
 
## Output Format
 
### `results/wcsr_comparison.md` (example)
 
| Tier | chordformer (mean ± std) | mambaformer (mean ± std) |
|---|---|---|
| **root** | 0.8412 ± 0.0123 (0.822 / 0.860) | 0.8287 ± 0.0145 (0.809 / 0.847) |
| **majmin** | 0.7934 ± 0.0211 | 0.7801 ± 0.0198 |
| **mirex** | 0.7712 ± 0.0189 | 0.7590 ± 0.0203 |
| … | … | … |
 
### `results/wcsr_comparison.csv`
 
```
Tier,chordformer_mean,chordformer_std,...,mambaformer_mean,...
root,0.8412,0.0123,...
```
 
### Per-model JSON (`results/chordformer_cv_results.json`)
 
Stores all raw fold scores and aggregated statistics for reproducibility.
 
---
 
## Design Decisions
 
### Why static JSON splits?
Both model branches must train and evaluate on **exactly the same songs**
in each fold.  Generating splits dynamically (even with the same seed) is
fragile when the on-disk dataset changes or when the two branches use
slightly different discovery logic.  Serialising the paths to JSON
eliminates this risk entirely.
 
### Why fold-aware cache names?
`BelloChordFormerDataset` caches preprocessed CQT + labels to disk.  If
both folds used the same cache key, fold 2 would silently load fold 1's
preprocessed data.  The `split_name=f"fold{fold_num}_train"` argument
ensures each fold gets its own cache file.
 
### Why re-initialise inside `train_one_fold`?
PyTorch schedulers and optimisers carry state (momentum buffers, LR
history, bad-epoch counters) that must be reset between folds.  Creating
new objects inside the fold function is the safest way to guarantee no
state leakage.
 
### Why are the CRF functions in `chord_utils.py` and not `train_unified.py`?
`evaluation.py` needs `viterbi_decode_crf` to decode predictions during
WCSR computation.  `train_unified.py` imports from `evaluation.py`.  Putting
the CRF functions in `train_unified.py` would therefore create a circular
import.  `chord_utils.py` is a pure-utility module with no local imports,
making it the correct home.  `train_unified.py` re-exports them for any
callers that expect them there.
 
### Why WCSR instead of frame accuracy?
Frame-wise accuracy treats every 23 ms frame equally regardless of chord
duration and is dominated by frequent short chords.  WCSR weights each
evaluation by the *actual duration* of the annotated segment, which is the
metric used in all published chord recognition literature (MIREX, ISMIR).
