# Domain Shift in Human Activity Recognition

**WISDM v1.1 ↔ UCI HAR** — Quantitative analysis and adaptation benchmarking

> Full technical report with all figures: **[REPORT.md](REPORT.md)**

---

## Overview

This project studies **cross-dataset domain shift** between two smartphone accelerometer HAR datasets and evaluates three levels of domain adaptation:

| Stage | Method | WISDM-Test Macro F1 |
|---|---|---|
| No adaptation (baseline) | Train on raw UCI → test on WISDM | 0.085 |
| **Data-level** | PhysicalDistortion (5-operator pipeline) | **0.723** |
| Inference-level | + Test-Time Batch Normalisation | 0.516 |
| Training-level | + Domain-Adversarial (DANN) | 0.418 |

**Best model:** CNN1D with 10K parameters, trained on physically distorted UCI HAR, achieving Macro F1 = **0.723** on WISDM-Test (subjects 31–36).

---

## Project Structure

```
├── README.md
├── LICENSE
├── REPORT.md              ← Full technical report with all figures
│
├── src/                   ← All Python source code
│   ├── config.py          ← Centralised path configuration
│   ├── data_io.py         ← Data loading, alignment, windowing
│   ├── physics_engine.py
│   ├── domain_shift_metrics.py
│   ├── comprehensive_analysis.py
│   ├── visualize_distributions.py
│   ├── covariate_shift_engine.py   ← PhysicalDistortion pipeline
│   ├── architectures.py            ← FFN, CNN1D, BiGRU, TCN, Transformer
│   ├── domain_adaptation.py        ← TTBN + DANN
│   ├── hp_search.py                ← Optuna hyperparameter search
│   ├── final_benchmark.py          ← 4-condition ablation
│   ├── dann_tuning.py
│   └── visualize_stability.py      ← PCA, stability, confusion, radar
│
├── figures/               ← All generated figures (15 total)
├── results/               ← JSON results and Markdown summaries
└── data/
    ├── wisdm/             ← WISDM v1.1 raw data
    └── ucihar/            ← UCI HAR Inertial Signals
```

---

## Quick Start

```bash
conda create -n har_env python=3.12
conda activate har_env
pip install numpy scipy pandas scikit-learn matplotlib seaborn torch optuna tqdm tensorboard

cd src

# Dataset analysis (Section 4 of report)
python domain_shift_metrics.py
python comprehensive_analysis.py

# Hyperparameter search — ~30 min, GPU recommended
python hp_search.py --n-trials 40 --epochs 100

# Full ablation benchmark — ~10 min, GPU
python final_benchmark.py --config ../results/hp_best_configs.json

# Visualizations — ~5 min
python visualize_stability.py
```

---

## Key Figures

| Figure | Description |
|---|---|
| `figures/per_axis_analysis.png` | Per-axis Wasserstein — gravity axis dominates |
| `figures/gravity_bias_analysis.png` | UCI X-axis vs WISDM Y-axis gravity projection |
| `figures/latent_space_overlap.png` | Raw MMD = 0.830 in PCA feature space |
| `figures/distribution_alignment.png` | PhysicalDistortion closes the PCA gap |
| `figures/stability_analysis.png` | F1 vs noise / rotation intensity per model |
| `figures/confusion_matrices.png` | C0 vs C1 confusion patterns |
| `figures/radar_charts.png` | Per-activity F1 radar — CNN1D most balanced |

---

## Citation

If you use this code or report, please cite:

```bibtex
@misc{guo2026hardomainshift,
  author = {Guo, Gengzhan},
  title  = {Domain Shift in Human Activity Recognition: Analysis and Adaptation},
  year   = {2026},
  url    = {https://github.com/gengzhanguo/wisdm_ucihar}
}
```
