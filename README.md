# bio-cv-pipeline

A model-agnostic **K-fold Stratified CV + retrain + test evaluation** pipeline for PyTorch binary classifiers.

Swap any architecture or data format — the pipeline structure, metrics, and output format stay fixed.

---

## Features

- **5-stage fixed pipeline**: CV → summary → epoch selection → full retrain → test evaluation
- **Dual-track epoch selection**: per-fold best AUC for mean±std reporting; cross-fold avg AUC for choosing the final retrain epoch
- **7 metrics**: Acc / Sn / Sp / Prec / F1 / MCC / AUC — computed consistently across all folds
- **Architecture-agnostic**: MLP, CNN, GNN, Transformer, dual-tower, or any custom model that outputs `(B, 2)` logits
- **Data-format-agnostic**: fixed vectors, variable-length sequences, PyG graphs, multi-modal inputs
- **Reproducible**: dual seeds (`SEED_CV=42`, `SEED_FINAL=123`) for CV split and final retrain

---

## Pipeline Overview

```
Stage 1 ── K-fold Stratified CV (on training set)
            ├─ Track A: best AUC epoch per fold → mean ± std across folds
            └─ Track B: avg AUC per epoch across folds → select retrain epoch

Stage 2 ── CV summary (console table + cv_summary.csv)

Stage 3 ── Select final retrain epoch
            └─ epoch with highest cross-fold avg AUC (or MANUAL_N_EPOCHS override)

Stage 4 ── Retrain on full training set

Stage 5 ── Evaluate on held-out test set → test_results.txt + final_model.pth
```

---

## File Structure

```
bio-cv-pipeline/
├── SKILL.md                        # Main workflow & adaptation guide
├── README.md
├── references/
│   ├── metrics.md                  # Canonical compute_metrics() implementation
│   └── pipeline_anatomy.md         # Detailed per-stage code skeletons
└── assets/
    └── template_cv_pipeline.py     # Ready-to-use template with 4 ZONE markers
```

---

## Quick Start

### 1. Copy the template

```bash
cp assets/template_cv_pipeline.py train_cv.py
```

### 2. Fill in the 4 ZONEs

| Zone       | What to fill                                              |
| ---------- | --------------------------------------------------------- |
| **ZONE A** | File paths, `BATCH_SIZE`, `EPOCHS`, etc.                  |
| **ZONE B** | `Dataset.__getitem__` + `collate_fn` for your data format |
| **ZONE C** | `build_model()` — instantiate your model class            |
| **ZONE D** | Batch unpack line in `train_epoch` / `evaluate`           |

Everything outside the ZONEs is invariant — do not modify.

### 3. Run

```bash
python train_cv.py
```

---

## Data Format Adaptation (ZONE B)

`__getitem__`, `collate`, and the batch unpack in `train_epoch`/`evaluate` must stay aligned:

| Format                   | `__getitem__` returns   | unpack                 |
| ------------------------ | ----------------------- | ---------------------- |
| Single fixed vector      | `(feat, label)`         | `inputs, y = batch`    |
| Dual-tower               | `(feat1, feat2, label)` | `f1, f2, y = batch`    |
| Variable-length sequence | `(seq [L,D], label)`    | `seq, mask, y = batch` |
| PyG graph                | `(Data, label)`         | `g, y = batch`         |

---

## Output Files

```
{SAVE_DIR}/
    best_model_fold{1..K}.pth    # Best AUC checkpoint per fold
    loss_fold{1..K}.png
    roc_fold{1..K}.png
    cv_summary.csv               # K rows + mean + std

{RESULTS_DIR}/
    final_model_epoch{N}.pth
    test_results.txt             # 7 metrics + confusion matrix
```

---

## Key Configuration

```python
EPOCHS          = 20
N_SPLITS        = 5
BATCH_SIZE      = 16
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
SEED_CV         = 42
SEED_FINAL      = 123
MANUAL_N_EPOCHS = None   # Override automatic epoch selection
RUN_FINAL_EVAL  = True   # Set False to run CV only
```

---

## Requirements

```
torch
scikit-learn
numpy
pandas
matplotlib
torch_geometric  # only if using PyG graphs
```

---

## License

MIT
