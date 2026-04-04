# Cross-Dataset Domain Shift in Human Activity Recognition

**WISDM v1.1 ↔ UCI HAR** — Quantitative analysis and adaptation benchmarking

---

## Overview

This project investigates **domain shift** between two widely-used accelerometer-based Human Activity Recognition (HAR) datasets: [WISDM v1.1](https://www.cis.fordham.edu/wisdm/dataset.php) and [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The work is split into two parts, both documented in [`RQ1_Technical_Report.md`](RQ1_Technical_Report.md):

| Part | Research Question | Key Result |
|---|---|---|
| **I** | What are the sources of domain shift, and how large are they? | 7 shift sources identified; raw MMD = 0.830 |
| **II** | Can the shift be corrected, and which adaptation layer works best? | Physics-grounded data augmentation reduces MMD by 45%; best F1 = 0.723 |

---

## Datasets

Download and place data as follows (not committed to Git — large files):

```
wisdm_ucihar/
├── WISDM_ar_latest/
│   └── WISDM_ar_v1.1/
│       └── WISDM_ar_v1.1_raw.txt          ← from wisdm.net
└── human+activity+recognition+using+smartphones/
    └── UCI HAR Dataset/
        ├── train/Inertial Signals/total_acc_*.txt
        └── test/Inertial Signals/total_acc_*.txt
```

Both datasets use a **5-class shared label scheme**: Walking, Upstairs, Downstairs, Sitting, Standing (Jogging and Laying excluded).

---

## Project Structure

```
├── data_io.py                  # Loading, alignment (50→20 Hz), windowing
├── physics_engine.py           # Acceleration magnitude, frequency centroid
├── domain_shift_metrics.py     # Wasserstein, KL divergence, MMD
├── comprehensive_analysis.py   # Per-axis, gravity, class prior, autocorrelation
├── visualize_distributions.py  # Global KDE overview
│
├── covariate_shift_engine.py   # 5-operator PhysicalDistortion pipeline
├── architectures.py            # FFN, CNN1D, BiGRU, TCN, Transformer
├── domain_adaptation.py        # TTBN + DANN implementation
├── hp_search.py                # Optuna hyperparameter search
├── final_benchmark.py          # 4-condition ablation (C0: raw → C3: DANN)
├── dann_tuning.py              # DANN-specific LR / weight grid search
├── visualize_stability.py      # PCA alignment, stability, confusion, radar
│
├── RQ1_Technical_Report.md     # Full report (Parts I and II)
├── figures/                    # Part I analysis figures (auto-generated)
└── results_final/figures/      # Part II visualizations (auto-generated)
```

---

## Quick Start

```bash
conda create -n har_env python=3.12
conda activate har_env
pip install numpy scipy pandas scikit-learn matplotlib seaborn torch optuna tqdm tensorboard

# Part I — Dataset Analysis
python visualize_distributions.py
python domain_shift_metrics.py
python comprehensive_analysis.py

# Part II — Domain Adaptation Benchmark
python hp_search.py --n-trials 40 --epochs 100 --out-dir results
python final_benchmark.py --config results/hp_best_configs.json --out-dir results_final
python visualize_stability.py --hp-config results/hp_best_configs.json --out-dir results_final/figures
```

GPU strongly recommended for Part II (tested on CUDA 12.x). CPU-only is supported but slow.

---

## Key Results (Part II)

### WISDM-Test Macro F1 — 4 Conditions × 5 Models

| Model | Params | C0: Raw | C1: +Distort | C2: +TTBN | C3: +DANN |
|---|---|---|---|---|---|
| **CNN1D** | **10K** | 0.120 | **0.723** | 0.610 | 0.581 |
| Transformer | 84K | 0.046 | 0.580 | 0.580 | 0.327 |
| FFN | 82K | 0.069 | 0.542 | 0.425 | 0.401 |
| TCN | 68K | 0.127 | 0.535 | 0.451 | 0.418 |
| BiGRU | 51K | 0.062 | 0.512 | 0.512 | 0.365 |

### Main Findings

1. **PhysicalDistortion** (5-operator physics pipeline) is the single most effective intervention, reducing MMD by 45.4% and lifting mean F1 from 0.085 → 0.574.
2. **Smaller models transfer better**: CNN1D at 10K params outperforms all 50–140K param models on cross-dataset transfer.
3. **TTBN and DANN do not help** in this low-data regime (~6.7K source windows): both degrade performance relative to data-level augmentation alone.
4. **Upstairs remains unsolved** (best F1 = 0.45) — requires phase-sensitive gait-cycle augmentation beyond window-level transforms.

---

## Figures

| Figure | Description |
|---|---|
| `figures/RQ1_Physical_Distributions.png` | Raw signal distributions by activity |
| `figures/per_axis_analysis.png` | Per-axis Wasserstein distances |
| `figures/gravity_bias_analysis.png` | Gravity projection shift (dominant source) |
| `figures/latent_space_overlap.png` | MMD in PCA feature space |
| `results_final/figures/distribution_alignment.png` | PCA: Raw UCI → Aug UCI → WISDM |
| `results_final/figures/stability_analysis.png` | F1 vs noise / rotation intensity |
| `results_final/figures/confusion_matrices.png` | 2×5 confusion matrix comparison |
| `results_final/figures/radar_charts.png` | Per-activity F1 radar charts |

---

## Report

The full technical report is in [`RQ1_Technical_Report.md`](RQ1_Technical_Report.md), covering:
- Part I: 7 quantified sources of domain shift with three statistical metrics each
- Part II: PhysicalDistortion design, HP search, 4-condition ablation, stability analysis, discussion

---

## References

- Kwapisz et al. (2011) — WISDM dataset
- Anguita et al. (2013) — UCI HAR dataset
- Ganin et al. (2016) — DANN
- Schneider et al. (2020) — Test-Time BN
- Akiba et al. (2019) — Optuna
