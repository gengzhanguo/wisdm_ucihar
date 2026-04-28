# Cross-Dataset HAR Transfer: UCI HAR → WISDM

Physics-grounded domain adaptation for human activity recognition across sensor setups.

## Overview

We train on **UCI HAR** (waist-mounted, 50 Hz, Butterworth-filtered) and evaluate on **WISDM v1.1** (front-pocket, 20 Hz, raw). No target labels are used during training.

The core finding: gravity axis projection mismatch (Wasserstein = 8.7 m/s²) dominates the domain shift. A five-operator data augmentation pipeline (**PhysicalDistortion**) addressing physical sensor differences brings CNN1D macro F1 from 0.120 to 0.723. Standard model-level methods (TTBN, DANN) hurt performance when applied on top.

## Datasets

- **UCI HAR**: 30 subjects, 50 Hz, 5 classes, 8,355 training windows after alignment
- **WISDM v1.1**: 36 subjects, 20 Hz, 5 classes, 5,307 test windows
- Aligned window size: 51 × 3 @ 20 Hz, units: m/s²

Download datasets and place under `data/uci_har/` and `data/wisdm/` respectively.

## Key Results

| Condition | CNN1D F1 | Avg F1 |
|-----------|----------|--------|
| C0 Raw (no adaptation) | 0.120 | 0.085 |
| C1 PhysicalDistortion | **0.723** | **0.578** |
| C2 + TTBN | 0.610 | 0.516 |
| C3 + DANN | 0.581 | 0.418 |
| C1 + HP search (CNN1D) | **0.762** | — |

MMD: 0.830 → 0.453 (45% reduction after PhysicalDistortion).

## Project Structure

```
.
├── src/                    # Training and augmentation code
│   ├── models/             # CNN1D, Transformer, FFN, TCN, BiGRU
│   └── augmentation/       # PhysicalDistortion pipeline
├── data/                   # Dataset loading utilities
├── figures/                # Generated plots
├── results/                # Experiment outputs (JSON)
├── report.tex              # Full paper (NeurIPS-style, 6-8 pages)
├── ppt.tex                 # Beamer slides (Madrid theme, 14 slides)
└── Class_Imbalance_Analysis.ipynb
```

## Running Experiments

```bash
# Install dependencies
pip install torch numpy scipy scikit-learn

# Run ablation study
python src/train.py --condition c0  # Raw
python src/train.py --condition c1  # PhysicalDistortion
python src/train.py --condition c2  # + TTBN
python src/train.py --condition c3  # + DANN

# Hyperparameter search (CNN1D)
python src/hp_search.py --n_trials 50
```

## Building the Paper and Slides

```bash
# Requires tectonic (https://tectonic-typesetting.github.io)
tectonic report.tex
tectonic ppt.tex
```

## PhysicalDistortion Pipeline

Five operators applied to UCI HAR training data:

1. **Orientation shift**: 90° rotation around Z + ±10° random wobble
2. **Gravity attenuation**: ×0.7523 (= 7.29/9.69 m/s²)
3. **Per-activity amplitude scaling**: empirical per-axis scale factors
4. **Gait spectral boost**: ×Uniform[1.2, 1.5] in 0.8–2 Hz (locomotion only)
5. **AR(1) colored noise**: α = 0.9865, stationary initialization

## Domain Shift Sources (by importance)

| Rank | Source | Magnitude |
|------|--------|-----------|
| 1 | Gravity axis projection | W₁ = 8.7 m/s² |
| 2 | Signal dynamic amplitude | 3–5× difference |
| 3 | Noise temporal structure | AR(1) α≈0.986 vs. white |
| 4 | Class prior shift | TVD = 0.36 |
| 5 | Gait spectral energy | WISDM +40% in 0.8–2 Hz |
| 6 | Inter-subject variability | max W₁ = 1.89 m/s² |
| 7 | Gravity magnitude | 9.69 vs. 7.29 m/s² |
