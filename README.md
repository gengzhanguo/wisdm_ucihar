# Domain Shift Analysis: WISDM vs UCI HAR

A systematic, physics-grounded quantitative analysis of domain shift between two widely-used Human Activity Recognition (HAR) benchmark datasets.

## Research Question

> **What are the key sources of domain shift across the WISDM dataset and other HAR datasets? What metrics can be used to quantify the differences?**

## Quick Summary

Seven distinct sources of domain shift were identified and quantified. The dominant source — typically invisible in prior work — is **sensor mounting orientation**, with per-axis Wasserstein distances up to **9.05 m/s²**, more than five times larger than the commonly-reported scalar magnitude shift (1.63 m/s²).

| Source | Metric | Value |
|---|---|---|
| Sensor orientation (X-axis) | Wasserstein | 9.06 m/s² |
| Sensor orientation (Y-axis) | Wasserstein | 8.72 m/s² |
| Class prior shift P(Y) | TVD | 0.489 |
| Feature-space global shift | MMD | 0.455 |
| Max inter-user shift | Wasserstein | 1.891 m/s² |
| Cross-dataset shift (scalar) | Wasserstein | 1.631 m/s² |

## Report

The full analysis is in [`RQ1_Technical_Report.md`](RQ1_Technical_Report.md), including:
- Experimental setup and physical alignment procedure
- Seven-source shift taxonomy with quantitative tables
- Discussion and recommendations for HAR domain adaptation research

## Figures

All figures are in the [`figures/`](figures/) directory.

| Figure | Description |
|---|---|
| `RQ1_Physical_Distributions.png` | Global KDE: Acc Magnitude & Frequency Centroid |
| `amplitude_dist_metrics.png` | Per-class Wasserstein + KDE |
| `freq_centroid_dist_metrics.png` | Frequency Centroid per-class analysis |
| `latent_space_overlap.png` | PCA-2D scatter + Multi-kernel MMD |
| `subject_wasserstein_heatmap.png` | 10×10 inter-user Wasserstein (WISDM) |
| `per_axis_analysis.png` | Per-axis (X/Y/Z) distribution analysis |
| `gravity_bias_analysis.png` | Gravity component & sensor orientation |
| `class_prior_shift.png` | P(Y) class prior comparison |
| `autocorrelation_analysis.png` | Gait ACF & Power Spectral Density |
| `comprehensive_summary.png` | Radar chart + full shift taxonomy |

## Code

| Script | Role |
|---|---|
| `data_io.py` | Data loading, resampling, unit conversion, windowing |
| `physics_engine.py` | Acc Magnitude and Frequency Centroid extraction |
| `domain_shift_metrics.py` | Wasserstein, KL, MMD; per-class analysis |
| `comprehensive_analysis.py` | Per-axis, gravity, prior shift, autocorrelation |
| `visualize_distributions.py` | Global KDE overview |

## Setup

```bash
conda activate <your_env>   # requires numpy, scipy, scikit-learn, matplotlib, seaborn
cd wisdm_ucihar
python visualize_distributions.py
python domain_shift_metrics.py
python comprehensive_analysis.py
```

## Data

Both datasets are included in this repository (only the files required for this analysis).

### WISDM v1.1

| Field | Detail |
|---|---|
| **Version** | WISDM Activity Recognition Dataset v1.1 |
| **File included** | `WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt` (48 MB) |
| **Subjects** | 36 |
| **Sampling rate** | 20 Hz |
| **Sensor** | Tri-axial accelerometer (Android phone, front trouser pocket) |
| **Activities** | Walking, Jogging, Upstairs, Downstairs, Sitting, Standing |
| **Samples** | 1,098,207 raw rows |
| **Original source** | https://www.cis.fordham.edu/wisdm/dataset.php |
| **Citation** | Kwapisz et al. (2011), ACM SIGKDD Explorations, 12(2), 74–82 |

> **Note:** The `.arff` transformed file and `.Zone.Identifier` Windows metadata files are excluded via `.gitignore`.

### UCI HAR Dataset

| Field | Detail |
|---|---|
| **Version** | UCI HAR Dataset (original, 2012) |
| **Files included** | `total_acc_[xyz]_{train,test}.txt`, labels, subject IDs, README (see structure below) |
| **Files excluded** | `X_train.txt` / `X_test.txt` (561 pre-computed features, not used), `body_acc_*`, `body_gyro_*` (not used in this analysis) |
| **Subjects** | 30 |
| **Sampling rate** | 50 Hz (resampled to 20 Hz in this analysis) |
| **Sensor** | Tri-axial accelerometer + gyroscope (Samsung Galaxy S II, waist-mounted) |
| **Activities** | Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying |
| **Windows** | 7,352 train + 2,947 test (128 samples @ 50 Hz = 2.56 s, 50% overlap) |
| **Original source** | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| **Citation** | Anguita et al. (2013), ESANN |

## References

- Anguita et al. (2013). *A public domain dataset for HAR using smartphones.* ESANN.
- Kwapisz et al. (2011). *Activity recognition using cell phone accelerometers.* ACM SIGKDD.
- Trofimov et al. (2024). *A benchmark for domain adaptation in smartphone-based HAR.* Scientific Data.
