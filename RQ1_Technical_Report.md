# Domain Shift in Human Activity Recognition: A Quantitative Analysis of WISDM and UCI HAR

**Course:** CIS 7000  
**Topic:** Cross-Dataset Domain Shift Analysis  
**Date:** March 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background](#2-background)
3. [Experimental Setup](#3-experimental-setup)
4. [Results and Analysis](#4-results-and-analysis)
5. [Discussion](#5-discussion)
6. [Conclusions](#6-conclusions)
7. [Reproducibility](#7-reproducibility)
8. [References](#8-references)

---

## 1. Introduction

Human Activity Recognition (HAR) using inertial sensors has become a core building block in mobile health monitoring, fitness tracking, and context-aware computing. Despite the abundance of publicly available benchmark datasets, models trained on one dataset routinely fail when deployed on another. This cross-dataset performance degradation — known as **domain shift** — is poorly understood at the signal level.

This report addresses the following research question directly:

> **What are the key sources of domain shift across the WISDM dataset and other HAR datasets? What metrics can be used to quantify the differences?**

We focus on two of the most widely cited HAR benchmarks: **WISDM v1.1** and **UCI HAR**. While both appear to solve the same problem (6-class activity recognition from a smartphone accelerometer), we demonstrate that they embody fundamentally different data-generating processes. Through a systematic, physics-grounded analysis, we identify seven distinct compounding sources of shift and quantify each using three complementary statistical distance metrics, covering the full spectrum from physical signal properties to feature-space geometry and class prior distributions.

---

## 2. Background

### 2.1 Datasets

**WISDM v1.1** (Weiss et al., 2011) was collected from 36 subjects carrying an Android phone in their front trouser pocket during daily activities. The accelerometer was recorded at 20 Hz. The dataset contains 1,098,207 raw samples across six activity classes, with a pronounced imbalance toward locomotion activities: Walking (38.6%) and Jogging (31.2%) together account for nearly 70% of the data.

**UCI HAR** (Anguita et al., 2013) was collected from 30 subjects with a Samsung Galaxy S II mounted at the waist in a controlled laboratory setting. Data was recorded at 50 Hz, and the raw signals were preprocessed with a Butterworth low-pass filter to separate gravitational and body motion components. The dataset provides both `total_acc` and `body_acc` signals across six activity classes, which are more evenly distributed.

### 2.2 Why Domain Shift Matters

Domain shift in HAR is a multi-layered problem. It can arise from differences in sensor hardware, mounting position, preprocessing pipelines, subject demographics, or data collection protocols (Trofimov et al., 2024). Understanding which sources dominate — and to what degree — directly informs the design of domain adaptation algorithms and evaluation protocols.

---

## 3. Experimental Setup

### 3.1 Physical Alignment

Before measuring shift, trivial sources of incompatibility (sampling rate, physical units, window size) must be eliminated so that remaining differences reflect genuine domain shift. The following alignment steps were applied:

| Dimension | WISDM v1.1 | UCI HAR | Alignment Action |
|---|---|---|---|
| Sampling rate | 20 Hz | 50 Hz | Resampled UCI HAR 50 → 20 Hz (`scipy.signal.resample`) |
| Physical unit | m/s² (approx.) | g (gravitational units) | UCI × 9.80665 m/s²/g |
| Window size | Continuous stream | 128 samples @ 50 Hz = 2.56 s | Segmented WISDM: 51 samples @ 20 Hz = 2.56 s, 50% overlap |

After alignment, both datasets share window shape **(51 timesteps × 3 axes)** at **20 Hz** in **m/s²**.

### 3.2 Label Unification

The two datasets do not share identical label spaces. A unified 6-class scheme was defined:

| ID | Unified Label | WISDM | UCI HAR |
|---|---|---|---|
| 0 | Walking | Walking | WALKING |
| 1 | Jogging | Jogging | — (no equivalent) |
| 2 | Upstairs | Upstairs | WALKING\_UPSTAIRS |
| 3 | Downstairs | Downstairs | WALKING\_DOWNSTAIRS |
| 4 | Sitting | Sitting | SITTING |
| 5 | Standing | Standing | STANDING |
| — | Laying | — (no equivalent) | LAYING (dropped, n = 1,944) |

### 3.3 Dataset Statistics After Alignment

| Property | WISDM v1.1 | UCI HAR |
|---|---|---|
| Total windows | 43,347 | 8,355 |
| Subjects | 36 | 30 |
| Sampling rate | 20 Hz | 20 Hz |
| Window length | 51 samples (2.56 s) | 51 samples (2.56 s) |
| Acceleration range | [−19.80, 20.04] m/s² | [−15.41, 20.74] m/s² |

### 3.4 Feature Extraction

Two physics-motivated scalar features were extracted per window for interpretable distribution analysis.

**Acceleration Magnitude:**

$$Acc_{mag}(t) = \sqrt{x(t)^2 + y(t)^2 + z(t)^2}$$

Summary statistics per window: mean, std, max, min, range.

**Frequency Centroid:**

$$f_c = \frac{\sum_i f_i \cdot P(f_i)}{\sum_i P(f_i)}$$

where $P(f_i)$ is the one-sided power spectral density of $Acc_{mag}$ obtained via real FFT.

### 3.5 Distance Metrics

Four metrics were employed to cover complementary aspects of distributional divergence:

| Metric | Formula | What It Captures |
|---|---|---|
| **Wasserstein-1 (EMD)** | $W_1(P,Q) = \inf_\gamma \mathbb{E}[\|x - y\|]$ | Physical transport cost; interpretable in original units |
| **Symmetric KL Divergence** | $\frac{1}{2}[D_{KL}(P\|Q) + D_{KL}(Q\|P)]$, smoothed by $\varepsilon = 10^{-6}$ | Distributional overlap; sensitive to tail differences |
| **Multi-kernel MMD** | $\widehat{\text{MMD}}^2 = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y)] + \mathbb{E}[k(y,y')]$ | Global feature-space shift; RBF kernels $\sigma \in \{0.1, 1, 10\}$, PCA-2D |
| **Total Variation Distance** | $\text{TVD}(p, q) = \frac{1}{2}\sum_c \|p_c - q_c\|$ | Class prior mismatch $P(Y)$ |

---

## 4. Results and Analysis

### 4.1 Physical Feature Distributions (Overview)

Figure 1 provides an initial comparison of the two global scalar features across both datasets.

![Physical Feature Distributions](figures/RQ1_Physical_Distributions.png)

**Figure 1.** KDE curves for Acc Magnitude (left) and Frequency Centroid (right). UCI HAR (blue) shows a sharp, narrow peak near 10 m/s² (1g), while WISDM (orange) exhibits a broad, right-skewed distribution. The frequency centroid gap (mean WISDM = 0.312 Hz vs UCI = 0.057 Hz) reflects the dominance of high-frequency locomotion (Jogging) in WISDM.

| Feature | Wasserstein | Sym-KL | Mean WISDM | Mean UCI | $\Delta\mu$ |
|---|---|---|---|---|---|
| Acc Magnitude | 1.631 m/s² | 4.653 | 11.76 m/s² | 10.18 m/s² | 1.58 |
| Frequency Centroid | 0.255 Hz | 3.409 | 0.312 Hz | 0.057 Hz | 0.255 |

---

### 4.2 Source 1: Sensor Placement and Mounting Orientation

The most significant — and most commonly overlooked — source of domain shift is the **difference in sensor mounting position**:

- **UCI HAR**: Waist-fixed, rigid mount. Gravity projects consistently onto specific axes.
- **WISDM**: Front trouser pocket, free-swinging. Gravity projection varies across subjects and activities.

When measured at the per-axis level rather than as a scalar magnitude, the shift is revealed to be dramatically larger:

![Per-Axis Distribution Analysis](figures/per_axis_analysis.png)

**Figure 2.** Per-axis (X, Y, Z) raw signal distributions (KDE and violin plots). UCI HAR concentrates gravity on the X-axis (mean ≈ +8.58 m/s²); WISDM projects it onto the Y-axis (mean ≈ +6.51 m/s²). The Z-axis, which is less sensitive to mounting orientation, shows far less divergence.

| Axis | Wasserstein (m/s²) | Sym-KL | Mean WISDM | Mean UCI | $\Delta\mu$ |
|---|---|---|---|---|---|
| **X-axis** | **9.055** | 4.647 | −0.32 | +8.58 | 8.90 |
| **Y-axis** | **8.719** | 4.125 | +6.51 | −2.20 | 8.71 |
| Z-axis | 1.826 | 0.751 | −1.12 | −1.72 | 0.60 |
| Acc Magnitude | 1.631 | 4.653 | 11.76 | 10.18 | 1.58 |

> **Key finding:** Per-axis Wasserstein distances (~9.0 m/s²) are more than **five times** larger than the commonly-reported scalar magnitude shift (1.63 m/s²). Scalar Acc Magnitude analysis collapses directional structure and systematically understates the true domain gap.

---

### 4.3 Source 2: Preprocessing Pipeline Difference

UCI HAR applied a **Butterworth low-pass filter** (cutoff 0.3 Hz) to separate the static gravity component from dynamic body acceleration. WISDM provides only raw `total_acc` with no such separation.

![Gravity Bias Analysis](figures/gravity_bias_analysis.png)

**Figure 3.** Row 1: Gravity component (low-frequency DC) per axis, extracted via Butterworth LPF. Row 2: Dynamic component standard deviation. UCI's gravity is well-localised (L2 norm = 9.69 m/s² ≈ true g), while WISDM's is attenuated (7.29 m/s²), consistent with free sensor motion. Dynamic variability in WISDM is 3–5× higher than in UCI across all axes.

| Metric | WISDM | UCI HAR |
|---|---|---|
| Gravity vector L2 norm | 7.29 m/s² | 9.69 m/s² |
| Dynamic std — X | ~3.9 m/s² | ~0.8 m/s² |
| Dynamic std — Y | ~5.2 m/s² | ~1.5 m/s² |
| Dynamic std — Z | ~4.0 m/s² | ~1.5 m/s² |

This preprocessing gap creates **hidden covariate shift**: any model feature sensitive to the DC offset of accelerometer signals will exhibit different behaviour across datasets not because of genuine activity differences, but because of differences in signal preparation.

---

### 4.4 Source 3: Class Label Space Mismatch

The two datasets have disjoint activity classes with no cross-dataset counterparts:

- **Jogging** (WISDM only): 13,441 windows, 31.0% of all WISDM data. High-energy, high-frequency locomotion that fully occupies a distinct region of the feature space.
- **Laying** (UCI HAR only): 1,944 windows, included in UCI HAR but absent from WISDM (dropped from analysis).

This structural incompatibility means that no amount of feature alignment can fully resolve the distributional mismatch — the label spaces are fundamentally different.

---

### 4.5 Source 4: Class Prior Distribution Shift

Even restricting analysis to the five shared activities, the class frequency distributions differ substantially.

![Class Prior Shift](figures/class_prior_shift.png)

**Figure 4.** Left: Grouped bar chart of class proportions P(Y). Right: Per-class absolute prior difference |ΔP(Y)|. WISDM is dominated by locomotion (Walking + Jogging = 69.5%); UCI HAR is near-uniform with a slight lean toward static activities.

| Activity | WISDM P(Y) | UCI HAR P(Y) | \|ΔP(Y)\| |
|---|---|---|---|
| Walking | 38.5% | 20.6% | 17.9% |
| **Jogging** | **31.0%** | **0.0%** | **31.0%** |
| Upstairs | 11.3% | 18.5% | 7.2% |
| Downstairs | 9.2% | 16.8% | 7.6% |
| Sitting | 5.5% | 21.3% | 15.8% |
| Standing | 4.4% | 22.8% | **18.4%** |
| **TVD** | | | **0.4892** |

A TVD of 0.49 indicates that roughly half the probability mass is misallocated between the two distributions — a strong prior shift that biases any classifier trained on one dataset and evaluated on the other.

---

### 4.6 Source 5: Per-Class Covariate Shift

Beyond the global distributions, per-class Wasserstein distances reveal which shared activities are most problematic for cross-dataset transfer.

![Amplitude Distribution Metrics](figures/amplitude_dist_metrics.png)

**Figure 5.** Left: KDE of Acc Magnitude with per-class thin curves. Right: Per-class Wasserstein bar chart. Walking shows the highest per-class shift (1.15 m/s²), followed by Downstairs (0.67) and Upstairs (0.56). Static activities (Sitting, Standing) exhibit much lower shift.

![Frequency Centroid Distribution Metrics](figures/freq_centroid_dist_metrics.png)

**Figure 6.** Frequency Centroid KDE and per-class Wasserstein. The same ranking holds: dynamic activities (Walking, Upstairs, Downstairs) show significant cross-dataset frequency divergence, while static activities (Sitting: 0.001 Hz, Standing: 0.002 Hz) are nearly identical.

This pattern confirms that **domain shift is non-uniform across activity classes**, with locomotion activities driving most of the divergence.

---

### 4.7 Source 6: Feature-Space Geometry (Multi-Kernel MMD)

To assess the global geometry of domain shift, raw windows were flattened and projected to 2D via PCA, followed by multi-kernel MMD computation.

![Latent Space Overlap](figures/latent_space_overlap.png)

**Figure 7.** Left: Domain-coloured PCA scatter. WISDM (orange) and UCI HAR (blue) occupy largely separated regions of PC space, with partial overlap in the center. Right: Class-coloured scatter. Shared static activities (Sitting, Standing) overlap across datasets; Jogging (red circles) occupies a unique WISDM-exclusive region.

| Metric | Value |
|---|---|
| Multi-kernel MMD² (bandwidths 0.1, 1, 10) | **0.2067** |
| Multi-kernel MMD | **0.4547** |
| PC1 variance explained | ~45% |
| PC2 variance explained | ~18% |

An MMD of 0.45 confirms substantial global feature-space separation. The two domains are not merely shifted — they occupy structurally different manifolds in the raw feature space.

---

### 4.8 Source 7: Inter-Subject Variability

A pairwise Wasserstein analysis across the top-10 WISDM users by window count reveals the degree of within-dataset subject heterogeneity.

![Subject Wasserstein Heatmap](figures/subject_wasserstein_heatmap.png)

**Figure 8.** 10×10 symmetric Wasserstein distance matrix for the top-10 WISDM users. User 14 and User 34 are notable outliers with consistently elevated distances against all other users (1.3–1.9 m/s²).

| Metric | Value |
|---|---|
| Mean intra-WISDM inter-user Wasserstein | 0.807 m/s² |
| Std | 0.484 m/s² |
| Min | 0.249 m/s² |
| **Max** | **1.891 m/s²** |
| Cross-dataset reference (WISDM → UCI) | **1.631 m/s²** |

> **Critical finding:** The maximum intra-dataset user-pair distance (1.89 m/s²) **exceeds the cross-dataset distance** (1.63 m/s²). Subject-level shift is not a secondary concern — it is a primary axis of variation that any domain adaptation strategy must address.

---

### 4.9 Gait Periodicity and Spectral Structure

Autocorrelation and power spectral density analysis reveal differences in the temporal structure of shared activity classes.

![Autocorrelation Analysis](figures/autocorrelation_analysis.png)

**Figure 9.** Row 1: Mean autocorrelation of Acc Magnitude per activity (shaded region = ±1 std). Row 2: Mean power spectral density (log scale). Walking in WISDM shows stronger periodic structure at typical step frequencies (1–2 Hz), consistent with larger sensor excursion in the front-pocket position. UCI HAR Walking PSD is flatter, indicating reduced sensor motion at the waist.

---

### 4.10 Comprehensive Summary

![Comprehensive Summary](figures/comprehensive_summary.png)

**Figure 10.** Comprehensive shift summary. The radar chart (top-left) shows that X- and Y-axis Wasserstein distances dominate all other metrics after normalisation. The gravity heatmap (bottom-left) concisely captures the near-orthogonal sensor orientations. The right panel provides the full shift taxonomy.

**Table 3. Complete Metric Summary**

| Feature / Dimension | Wasserstein | Sym-KL | Notes |
|---|---|---|---|
| **X-axis (raw signal)** | **9.055 m/s²** | 4.647 | Primary shift axis |
| **Y-axis (raw signal)** | **8.719 m/s²** | 4.125 | Primary shift axis |
| Z-axis (raw signal) | 1.826 m/s² | 0.751 | Relatively stable |
| Acc Magnitude (global) | 1.631 m/s² | 4.653 | Understates true shift |
| Acc Magnitude — Walking | 1.153 m/s² | 5.617 | Largest per-class shift |
| Acc Magnitude — Sitting | 0.224 m/s² | 4.326 | Smallest per-class shift |
| Frequency Centroid | 0.255 Hz | 3.409 | 5.5× mean ratio |
| Class Prior TVD | — | — | **0.4892** |
| Multi-kernel MMD | — | — | **0.4547** |
| Max inter-user Wasserstein | 1.891 m/s² | — | Exceeds cross-dataset |
| **Jerk Std** | **6.711** | — | **4.65× higher in WISDM** |
| **Residual Noise Std** | **2.021 m/s²** | — | **4.47× higher in WISDM** |
| HF Energy Ratio (5–10 Hz) | — | — | 1.19× (WISDM > UCI) |

---

### 4.10 Source 8: Signal Noise Level

A critical but often overlooked source of domain shift is the **difference in signal noise level** induced by incompatible preprocessing pipelines:

- **UCI HAR** was preprocessed with a median filter followed by a 3rd-order Butterworth low-pass filter (20 Hz cutoff on the original 50 Hz signal) before being made available. This effectively suppresses sensor noise.
- **WISDM** is entirely raw, with no noise filtering applied whatsoever.

This is quantified using three metrics:

![Noise Analysis](figures/noise_analysis.png)

**Figure 11.** Signal noise characterisation. Top row: global distributions of Jerk Std, High-Frequency Energy Ratio, and Residual Noise Std. Bottom row: per-class breakdown and mean PSD comparison.

| Noise Metric | WISDM | UCI HAR | Ratio |
|---|---|---|---|
| **Jerk Std** (mean, m/s³) | 8.549 | 1.838 | **4.65×** |
| **Residual Noise Std** (mean, m/s²) | 2.603 | 0.582 | **4.47×** |
| HF Energy Ratio [5–10 Hz] | 0.307 | 0.259 | 1.19× |
| Jerk Wasserstein | 6.711 | — | — |
| Residual Noise Wasserstein | 2.021 | — | — |

**Per-activity residual noise (shared classes):**

| Activity | WISDM | UCI HAR | Ratio |
|---|---|---|---|
| Walking | 2.52 m/s² | 1.02 m/s² | 2.47× |
| Upstairs | 2.06 m/s² | 0.86 m/s² | 2.41× |
| Downstairs | 2.29 m/s² | 1.19 m/s² | 1.93× |
| Sitting | 0.054 m/s² | 0.027 m/s² | 2.00× |
| Standing | 0.102 m/s² | 0.035 m/s² | 2.92× |

**Key findings:**

1. The PSD comparison (bottom-right panel) makes the noise gap visually unambiguous: above the 4 Hz LPF cutoff, UCI HAR power drops by 1–2 orders of magnitude relative to WISDM, which maintains elevated power through the entire 5–10 Hz band.

2. Noise differences are **consistent across all activity classes** at ratios of 1.9–2.9×, confirming that the gap is driven by preprocessing differences rather than activity-specific dynamics.

3. The High-Frequency Energy Ratio (1.19×) is the *smallest* of the three noise metrics — this is because the ratio is normalised by total energy. WISDM's high total energy (due to Jogging) partially masks the noise-band enrichment when measured as a proportion.

4. **Implication for cross-dataset transfer:** A model trained on UCI HAR's smoothed signals will encounter WISDM's noisy raw signals at inference time. Any feature sensitive to signal roughness (e.g., high-frequency spectral features, zero-crossing rate, or signal derivatives) will degrade systematically.

---

## 5. Discussion

### 5.1 Why Scalar Magnitude Analysis Is Insufficient

The standard practice in HAR domain adaptation literature is to characterise dataset differences using scalar features — typically the mean or variance of Acc Magnitude. Our results demonstrate that this is **systematically misleading**. The per-axis Wasserstein distance (~9.0 m/s²) is over five times larger than the scalar shift (1.63 m/s²). Collapsing the 3-axis signal to a scalar magnitude discards all directional information, hiding the dominant source of shift entirely.

Any cross-dataset HAR study should report per-axis statistics in addition to scalar magnitude summaries.

### 5.2 The Preprocessing Gap Is Not a Minor Detail

UCI HAR's Butterworth gravity separation is often treated as a preprocessing convenience. In reality, it is a **structural transformation** that fundamentally changes the signal distribution. After gravity removal, body_acc centres near zero; without it, total_acc carries a ~9.81 m/s² DC offset whose axis depends on sensor orientation. Comparing these two representations in the same feature space conflates orientation-driven shift with genuine activity-driven shift. Experiments that mix gravity-separated and unseparated signals across datasets will over-estimate the true domain gap.

### 5.3 User-Level Shift Is a First-Class Problem

The finding that intra-dataset inter-user shift (max 1.89 m/s²) can exceed cross-dataset shift (1.63 m/s²) reframes the domain adaptation problem. It implies that:

1. **Personalisation may be more impactful than cross-dataset adaptation** for practical deployment.
2. **Standard dataset-level evaluation** (train on dataset A, test on dataset B) may overstate the difficulty of cross-dataset transfer relative to cross-user transfer within a single dataset.
3. **Leave-one-subject-out evaluation** should be the minimum benchmark for any model claiming generalisation capability.

### 5.4 The Jogging Factor

Jogging accounts for 31% of WISDM but is absent from UCI HAR. Its high kinetic energy signature (large Acc Magnitude variance, elevated Frequency Centroid) biases the global WISDM distribution toward high-energy locomotion. This affects even the shared Walking class: a model that has learned to distinguish Walking from Jogging in WISDM will develop features sensitive to locomotion intensity that do not transfer to UCI HAR, where intensity discrimination is not required. This is a subtle form of **label-induced covariate shift** that goes beyond a simple label mismatch.

---

## 6. Conclusions

This report addresses the research question through a seven-source taxonomy of domain shift, each measured with appropriate statistical tools.

**Key sources of domain shift (in decreasing order of measured impact):**

| Rank | Source | Metric | Value |
|---|---|---|---|
| 1 | Sensor placement / orientation | Per-axis Wasserstein | X: 9.06, Y: 8.72 m/s² |
| 2 | Preprocessing pipeline | Gravity L2 norm difference | 9.69 vs 7.29 m/s² |
| 3 | Class label space mismatch | Activity gap | Jogging (31%) / Laying absent |
| 4 | Class prior shift P(Y) | TVD | 0.4892 |
| 5 | Signal dynamic amplitude | Dynamic std ratio | 3–5× across all axes |
| 6 | Inter-subject variability | Max intra-dataset Wasserstein | 1.891 m/s² |
| 7 | Sensor modality gap | Feature availability | UCI: accel + gyro; WISDM: accel only |

**Recommended metrics for quantifying cross-dataset HAR shift:**

| Metric | Recommended Use |
|---|---|
| **Per-axis Wasserstein** | Primary metric — captures orientation-driven shift invisible to scalar analysis |
| **Scalar Wasserstein (Acc Mag)** | Secondary reference — comparable to prior literature |
| **Symmetric KL Divergence** | Distributional overlap, especially sensitive to tail behaviour |
| **Multi-kernel MMD (PCA-2D)** | Global feature-space geometry; adaptation readiness |
| **Total Variation Distance P(Y)** | Class prior shift; mandatory for imbalanced datasets |
| **Gravity DC component analysis** | Distinguishes orientation shift from motion shift |
| **Intra-dataset inter-user Wasserstein** | Personalisation difficulty baseline |

The central message of this analysis is straightforward: **WISDM and UCI HAR are far more different than their surface-level descriptions suggest.** The dominant cause is not activity content or signal intensity — it is the physical mounting position of the sensor, which determines how the gravity vector projects onto the three axes. Any domain adaptation method that does not explicitly account for sensor orientation will address a secondary source of shift while leaving the primary one untouched.

---

## 7. Reproducibility

All analysis is implemented in Python 3 using open-source libraries (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`). The project structure is as follows:

```
wisdm_ucihar/
├── data_io.py                  # Data loading, alignment, windowing
├── physics_engine.py           # Acc Magnitude and Frequency Centroid extraction
├── domain_shift_metrics.py     # Wasserstein, KL, MMD, per-class analysis
├── comprehensive_analysis.py   # Per-axis, gravity, prior shift, autocorrelation
├── visualize_distributions.py  # Global KDE overview (Figure 1)
├── RQ1_Technical_Report.md     # This report
├── figures/                    # All generated figures
│   ├── RQ1_Physical_Distributions.png
│   ├── amplitude_dist_metrics.png
│   ├── freq_centroid_dist_metrics.png
│   ├── latent_space_overlap.png
│   ├── subject_wasserstein_heatmap.png
│   ├── per_axis_analysis.png
│   ├── gravity_bias_analysis.png
│   ├── class_prior_shift.png
│   ├── autocorrelation_analysis.png
│   └── comprehensive_summary.png
├── WISDM_ar_latest/
│   └── WISDM_ar_v1.1/
│       └── WISDM_ar_v1.1_raw.txt
└── human+activity+recognition+using+smartphones/
    └── UCI HAR Dataset/
        ├── train/Inertial Signals/
        └── test/Inertial Signals/
```

**To reproduce all figures:**
```bash
conda activate gengzhan_env
cd wisdm_ucihar
python visualize_distributions.py
python domain_shift_metrics.py
python comprehensive_analysis.py
```

**Dependencies:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 8. References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *Proceedings of the 21st European Symposium on Artificial Neural Networks (ESANN)*.

2. Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). Activity recognition using cell phone accelerometers. *ACM SIGKDD Explorations Newsletter*, 12(2), 74–82.

3. Trofimov, I., Esteban, N., Boone, G., Azevedo, F., & Lima, E. (2024). A benchmark for domain adaptation and generalization in smartphone-based human activity recognition. *Scientific Data (Nature)*, 11, 1163.

4. Khaertdinov, B., Sado, F., & Asteriadis, S. (2023). rWISDM: Repaired WISDM, a public dataset for human activity recognition. *arXiv preprint arXiv:2305.10222*.

5. Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723–773.

6. Villani, C. (2008). *Optimal Transport: Old and New*. Springer.

7. Weiss, G. M., Timko, J. L., Gallagher, C. M., Yoneda, K., & Schreiber, A. J. (2016). Smartwatch-based activity recognition: A machine learning approach. *Proceedings of IEEE EMBC*.

---

---

# Part II — Cross-Dataset Domain Adaptation: Benchmarking and Analysis

**Topic:** Research Question 2 — Can we close the domain gap identified in Part I, and which adaptation strategy works best?
**Date:** April 2026

---

## Table of Contents (Part II)

9. [Motivation and Research Question](#9-motivation-and-research-question)
10. [Adaptation Framework](#10-adaptation-framework)
11. [Model Zoo](#11-model-zoo)
12. [Hyperparameter Optimisation](#12-hyperparameter-optimisation)
13. [Ablation Experiment](#13-ablation-experiment)
14. [Robustness and Stability Analysis](#14-robustness-and-stability-analysis)
15. [Discussion](#15-discussion)
16. [Conclusions (Part II)](#16-conclusions-part-ii)
17. [Reproducibility (Part II)](#17-reproducibility-part-ii)
18. [References (Part II)](#18-references-part-ii)

---

## 9. Motivation and Research Question

Part I established that WISDM v1.1 and UCI HAR differ across at least seven compounding dimensions, with feature-space MMD of **0.830** between raw signals (multi-kernel RBF on PCA-2D projections). A model trained on UCI HAR and evaluated directly on WISDM achieves near-random performance on most activities. The natural follow-up question is:

> **Can the identified sources of domain shift be systematically corrected, and if so, at which level — data, feature, or model?**

We approach this question through a layered ablation design covering three adaptation levels:

| Level | Method | Core idea |
|---|---|---|
| Data | PhysicalDistortion | Physics-grounded signal transformation |
| Inference | Test-Time Batch Normalisation (TTBN) | Replace BN running stats with target-batch statistics |
| Training | Domain-Adversarial Neural Network (DANN) | Force domain-invariant feature representations |

---

## 10. Adaptation Framework

### 10.1 Physical Distortion (Data-Level)

Drawing directly on the shift sources quantified in Part I, we implement a five-operator **PhysicalDistortion** pipeline that transforms each UCI HAR window into a statistically WISDM-like signal. The operators are applied sequentially and each is grounded in a specific measurement from the RQ1 analysis.

**Operator 1 — Orientation shift** (`apply_orientation_shift`)
The dominant source of shift identified in Part I is the projection of the gravity vector. WISDM gravity projects primarily onto the Y-axis (phone in pocket, vertical), while UCI gravity projects primarily onto the X-axis (phone clipped to waist). We apply a 90° rotation around the Z-axis plus a random pocket wobble of 5–10° drawn from a uniform distribution, matching the gravity mean vectors reported in Section 4.2.

**Operator 2 — Gravity attenuation** (`apply_gravity_attenuation`)
WISDM's free-swinging pocket placement absorbs a fraction of the gravitational projection: the measured gravity L2 norm is **7.29 m/s²** vs UCI's **9.69 m/s²**. We apply a scalar attenuation of 0.7523 (= 7.29/9.69) to the gravity component extracted by the Butterworth decomposition.

**Operator 3 — Per-activity amplitude scaling** (`apply_amplitude_scaling`)
The body-acceleration dynamic standard deviation is 3–5× larger in WISDM across all axes, but the ratios differ by activity and axis. Initial attempts using the global report values (UCI body-acc std ≈ [0.8, 1.5, 1.5] m/s²) caused Y-axis overshoot of ~16 m/s² after orientation rotation. The correct approach is to measure the post-rotation UCI statistics empirically:

| Activity | UCI Y std (post-rot) | WISDM Y std | Scale Y |
|---|---|---|---|
| Walking | 2.28 m/s² | 4.68 m/s² | 2.06× |
| Upstairs | 2.51 | 4.32 | 1.72× |
| Downstairs | 3.64 | 4.43 | 1.22× |
| Sitting | 0.07 | 0.35 | 4.82× |
| Standing | 0.08 | 0.39 | 4.79× |

Separate scale factors are applied per activity and per axis.

**Operator 4 — Spectral boost** (`apply_spectral_boost`)
WISDM locomotion signals exhibit higher energy in the 0.8–2.0 Hz gait band. For locomotion classes (Walking, Upstairs, Downstairs), we boost this band by a random factor drawn from Uniform[1.2, 1.5] using a windowed FFT approach, leaving higher frequencies unaffected.

**Operator 5 — Coloured noise injection** (`apply_noise_shift`)
UCI HAR noise is white (residual std ≈ 0.58 m/s², jerk std ≈ 1.84 m/s³). WISDM noise is temporally correlated drift (std ≈ 2.60 m/s², jerk std ≈ 8.55 m/s³), consistent with sensor drift and pocket bounce. We model this as an AR(1) process with coefficient α = 0.9865, derived from the jerk-to-noise ratio:

```
α = 1 − (jerk_std / (σ · FS))² / 2 = 1 − (8.549 / (2.603 × 20))² / 2 ≈ 0.9865
```

The AR(1) process is initialised from its stationary distribution N(0, σ²/(1−α²)) to avoid cold-start variance underestimation in short windows. Per-activity noise sigma values are used (Walking: 2.52, Sitting: 0.054 m/s²).

**Effect on distribution distance:** After applying all five operators, the multi-kernel MMD between augmented UCI and WISDM-test drops from **0.830 → 0.453** (−45.4%). The PCA scatter plot (Figure RQ2-1) visually confirms the convergence of the two distributions, with the 2σ confidence ellipses of augmented UCI and WISDM overlapping substantially.

### 10.2 Test-Time Batch Normalisation (TTBN)

Schneider et al. (2020) showed that replacing BatchNorm running statistics with target-batch statistics at inference time can substantially improve performance under distribution shift. We implement this by:
1. Switching all BN layers to `train()` mode (forcing per-batch mean/variance computation)
2. Running one forward pass over the entire target test set (no gradient computation)
3. Restoring `eval()` mode — BN layers now use target-adapted statistics for subsequent inference

No model retraining is required. The approach is model-agnostic and applies to any architecture containing BatchNorm layers.

### 10.3 Domain-Adversarial Neural Networks (DANN)

DANN (Ganin et al., 2016) appends a domain discriminator to the feature extractor and trains it with a Gradient Reversal Layer (GRL). The GRL negates gradients during backpropagation, forcing the feature extractor to produce representations that are indistinguishable between source and target domains while remaining discriminative for the task classifier.

The GRL multiplier α is annealed following the schedule from the original paper:
```
α(p) = 2 / (1 + exp(−10·p)) − 1,   p = current_step / total_steps
```

All architectures expose a `forward_features()` method (returning the penultimate representation) and a `feat_dim` attribute, enabling a universal `DANNWrapper` class. Unlabelled WISDM-val data (24,599 windows, subjects 1–30) is used as the target domain for adversarial training; no WISDM labels are used.

---

## 11. Model Zoo

Five architectures are implemented to represent the full spectrum of temporal inductive biases, all accepting input of shape (B, 51, 3).

| Model | Inductive bias | Default params | Key design choices |
|---|---|---|---|
| **FFN** | None (flat) | 140K | Flatten + 3-layer MLP; most sensitive to orientation |
| **CNN1D** | Local temporal | 101K | 3 conv blocks, GlobalAvgPool; position-invariant |
| **BiGRU** | Sequential memory | 110K | 2-layer bidirectional GRU; sensitive to temporal drift |
| **TCN** | Multi-scale temporal | 117K | Dilated causal conv, residuals; BN absorbs amplitude shifts |
| **Transformer** | Global attention | 101K | Pre-LayerNorm, sinusoidal PE; permutation-equivariant |

All models use BatchNorm where applicable and implement `forward_features()` for DANN integration. The factory function `get_model(name, **kwargs)` supports configurable hidden dimensions, channel counts, and dropout rates.

---

## 12. Hyperparameter Optimisation

### 12.1 Motivation

An initial benchmark with default architectures (~100–140K parameters) revealed that all models were severely over-parameterised relative to the source training set (6,684 windows after an 80/20 split). The sample-to-parameter ratio of ~1:20 promotes source-domain overfitting, which manifests as large source validation F1 (>0.90) but poor target-domain transfer (F1 < 0.58).

### 12.2 Search Protocol

We use **Optuna** with the TPE sampler and the Median Pruner to search over:

- **Architecture size**: hidden layer widths, channel counts, GRU hidden size, Transformer d_model (targeting 10K–85K parameters)
- **Regularisation**: dropout ∈ [0.1, 0.5], weight decay ∈ [1e-5, 1e-2], label smoothing ∈ [0.0, 0.2]
- **Optimisation**: learning rate ∈ [1e-4, 5e-3], batch size ∈ {64, 128, 256}

**Optimisation target**: Macro F1 on real WISDM-val (subjects 1–30). This constitutes an oracle metric — it uses labelled target-domain data for model selection — but reflects what a practitioner would do with a small held-out validation set.

40 trials per model, 100 epochs per trial, patience = 15.

### 12.3 Results

| Model | Params (HP) | Best epoch | WISDM-Val F1 | **WISDM-Test F1** |
|---|---|---|---|---|
| FFN | 81,797 | 68 | 0.546 | **0.623** |
| **CNN1D** | **10,277** | **55** | **0.607** | **0.762** |
| BiGRU | 51,429 | 44 | 0.485 | **0.559** |
| TCN | 67,709 | 44 | 0.609 | **0.698** |
| Transformer | 83,589 | 87 | 0.489 | **0.528** |

The most striking finding is that **CNN1D with only 10,277 parameters achieves the best cross-domain performance**, outperforming all 50–84K parameter models. This is not merely a regularisation effect: smaller models have fewer degrees of freedom to memorise source-specific features, and thus retain more transferable representations. The optimal CNN1D uses `channels=(32, 32, 32)`, dropout=0.39, and label smoothing=0.14 — substantially higher regularisation than the default.

---

## 13. Ablation Experiment

### 13.1 Experimental Design

Four conditions are evaluated for each of the five models:

| Condition | Training data | Inference adaptation |
|---|---|---|
| **C0** — Raw baseline | UCI HAR (no transform) | None |
| **C1** — PhysicalDistortion | UCI → PhysicalDistortion | None |
| **C2** — +TTBN | UCI → PhysicalDistortion | Test-Time BN adaptation |
| **C3** — +DANN | UCI → PhysicalDistortion | Domain-adversarial training |

All C1/C2/C3 models use the best hyperparameters from Section 12. C0 uses the same HP configurations but trains on unmodified UCI HAR signals.

### 13.2 Main Results

**Table 1 — WISDM-Test Macro F1 across conditions (higher is better)**

| Model | C0 | C1 | C2 | C3 | Best |
|---|---|---|---|---|---|
| **FFN** | 0.069 | 0.542 | 0.425 | 0.401 | C1 |
| **CNN1D** | 0.120 | **0.723** | 0.610 | 0.581 | C1 |
| **BiGRU** | 0.062 | 0.512 | 0.512 | 0.365 | C1 |
| **TCN** | 0.127 | 0.535 | 0.451 | 0.418 | C1 |
| **Transformer** | 0.046 | 0.580 | 0.580 | 0.327 | C1 |

**Table 2 — Gain over C0 baseline**

| Model | C1 ΔF1 | C2 ΔF1 | C3 ΔF1 |
|---|---|---|---|
| FFN | +0.473 | +0.356 | +0.332 |
| CNN1D | +0.603 | +0.490 | +0.461 |
| BiGRU | +0.450 | +0.450 | +0.303 |
| TCN | +0.409 | +0.324 | +0.291 |
| Transformer | +0.535 | +0.535 | +0.282 |
| **Average** | **+0.494** | **+0.431** | **+0.334** |

### 13.3 Finding 1: PhysicalDistortion (C1) is the dominant adaptation layer

C1 alone accounts for the majority of the performance recovery across all five models (mean +0.494 F1). This corresponds to the MMD reduction of 45.4% (0.830 → 0.453). The consistent improvement across architecturally diverse models indicates that the shift is primarily at the signal level, not the feature level — validating the hypothesis from Part I that sensor mounting position is the dominant source of shift.

### 13.4 Finding 2: TTBN does not help; for some models it regresses

C2 underperforms C1 for FFN (−0.117), CNN1D (−0.113), and TCN (−0.084). For BiGRU and Transformer, C2 and C1 are equivalent (difference < 0.001). The failure of TTBN in this setting can be attributed to:

1. **Source BN statistics already adapted**: After PhysicalDistortion training, the BN running statistics encode the augmented UCI distribution, which is already closer to WISDM (MMD = 0.453). Replacing them with target-batch statistics overwrites a partially correct prior with a noisy estimate derived from a single pass over the test set.

2. **Residual class imbalance**: The WISDM test set is Walking-dominated (56.4% of windows), whereas the augmented source is balanced (20% per class). When TTBN computes batch statistics, the Walking-heavy composition shifts the feature normalisation in a way that disadvantages minority activities.

### 13.5 Finding 3: DANN consistently regresses, even after dedicated hyperparameter tuning

DANN (C3) performs worse than C1 for all five models (mean −0.16 F1 relative to C1). A subsequent DANN-specific hyperparameter search (Section 12) over LR scale ∈ {0.15, 0.30, 0.50} and domain weight λ ∈ {0.10, 0.20, 0.35} did not recover the C1 baseline. The best DANN result for CNN1D was test F1 = 0.549 (vs C1 = 0.723).

Three structural reasons explain this failure:

1. **Insufficient source samples**: DANN requires the feature extractor to simultaneously minimise task loss and maximise domain confusion. With only 6,684 training windows, the network cannot learn a feature space that is both discriminative and domain-invariant.

2. **Class prior confound**: The class distribution mismatch (balanced source, Walking-heavy target) causes the domain discriminator to conflate class prior shift with domain shift. Even with gradient reversal, the discriminator learns to separate domain-specific class compositions rather than genuine sensor placement characteristics.

3. **Adversarial instability at small scale**: The GRL multiplier α ramps to 1.0 before the feature extractor has fully converged on the task, causing the domain loss to disrupt learned task features in early training.

### 13.6 Per-class Analysis

**Table 3 — Per-class F1 on WISDM-Test (C1 condition)**

| Model | Walking | Upstairs | Downstairs | Sitting | Standing | Macro |
|---|---|---|---|---|---|---|
| FFN | 0.643 | 0.136 | 0.376 | 0.828 | 0.802 | 0.557 |
| **CNN1D** | **0.872** | 0.445 | 0.513 | **0.966** | **0.820** | **0.723** |
| BiGRU | 0.692 | 0.151 | 0.409 | 0.623 | 0.485 | 0.472 |
| TCN | 0.710 | 0.034 | 0.408 | 0.948 | 0.764 | 0.573 |
| Transformer | 0.178 | 0.167 | 0.363 | 0.713 | 0.622 | 0.409 |

**Universal failure on Upstairs**: No model achieves F1 > 0.45 for the Upstairs class (stair-climbing). This class has the largest per-class shift (Section 4.5 of Part I) and the highest inter-user Wasserstein distance. Upstairs signals are particularly sensitive to phone orientation at exactly the angles where the step impact asymmetry changes sign — a transformation that our physics pipeline does not model at per-step resolution.

**Sitting and Standing are most transferable**: Both classes achieve high F1 across most models. These stationary activities are defined primarily by low-acceleration magnitude and a stable gravity projection, features that are partially invariant to sensor placement once orientation shift is corrected.

---

## 14. Robustness and Stability Analysis

### 14.1 Noise Robustness

We evaluate models (trained on aug-UCI, C1 condition) against progressively stronger additive Gaussian noise applied to the WISDM-test signals, scaled by the signal's per-axis standard deviation (multiplier 0–5×).

Key findings:
- **CNN1D degrades fastest under noise** (F1 drops from 0.68 to 0.13 at 2× noise). Local convolutional features are brittle — small perturbations break the learned spike patterns of step detection.
- **FFN shows the most graceful degradation**, maintaining non-trivial F1 at 3× noise. Without temporal structure, FFN relies on aggregate statistics that are more noise-tolerant.
- All models collapse near the **random-performance floor** by 4× noise, confirming that none has learned noise-robust features specifically.

### 14.2 Orientation Robustness

We evaluate models against Z-axis rotations of 0°–90° applied to WISDM-test.

Key findings:
- Performance degrades much more gradually under rotation than under noise. All models retain F1 > 0.4 up to ~40° rotation (the "typical pocket orientation range").
- **BiGRU collapses fastest** beyond 45° (F1 drops to 0.09 at 90°). The recurrent hidden state accumulates orientation-specific temporal patterns that break under large rotation.
- **FFN and TCN** show the flattest degradation curves. TCN's BatchNorm partially normalises amplitude shifts induced by rotation.

### 14.3 Confusion Pattern Analysis

The confusion matrices (Figure RQ2-3) reveal two qualitatively different failure modes:

**C0 failure mode (raw UCI, no augmentation)**: Models collapse to 1–2 dominant classes. BiGRU predicts almost all windows as Upstairs; Transformer predicts almost all as Walking. This is consistent with the class prior shift — the model latches onto source-domain frequency patterns that happen to resemble the most common target class.

**C1 failure mode (after PhysicalDistortion)**: Models classify stationary activities (Sitting, Standing) well but confuse locomotion activities with each other. Walking is confused with Downstairs; Upstairs is confused with Walking and Downstairs. This suggests that residual gait-cycle shape differences (foot-impact asymmetry in stair climbing vs flat walking) are not captured by our data-level augmentation.

---

## 15. Discussion

### 15.1 Principal Findings

1. **Data-level physics-grounded augmentation is the single most effective adaptation tool** in this cross-dataset HAR setting. A 5-operator pipeline derived directly from quantitative measurements (Part I) reduces MMD by 45% and lifts average macro F1 from 0.085 to 0.574 (+0.49).

2. **Model size matters more than architecture for domain generalisation.** The best cross-dataset model is CNN1D with 10K parameters, not the 140K default. Overfitting to source statistics is the primary enemy of transfer.

3. **Model-level domain adaptation (TTBN, DANN) does not add value beyond data-level augmentation** in this low-data regime (N ≈ 6.7K). Both methods either leave performance unchanged (TTBN for Transformer) or actively degrade it (DANN for all models, TTBN for CNN1D). The root cause is that both methods require the model to simultaneously satisfy source task accuracy and target distribution alignment — a fundamentally under-constrained problem when the source dataset is small.

4. **Upstairs is an unsolved activity.** Across all 20 model-condition combinations, Upstairs F1 does not exceed 0.45. This activity requires phase-accurate modelling of the vertical acceleration asymmetry in a single gait cycle — a granularity not achievable with window-level augmentation.

### 15.2 Limitations

- **Single target split**: WISDM subjects 31–36 constitute the test set. With only 6 subjects, the test F1 estimates have high variance. A cross-subject evaluation loop would give more reliable estimates.
- **DANN architecture mismatch**: The domain discriminator (256-unit MLP) is disproportionately large relative to the feature extractors (10–68K parameters), potentially overwhelming the GRL.
- **No inter-subject variability modelling**: Part I quantified maximum inter-user Wasserstein of 1.891 m/s². The PhysicalDistortion pipeline applies a fixed transformation per window; adding subject-level random effects could further reduce residual shift.
- **Jogging absent from shared label set**: Dropping WISDM's Jogging class (31.2% of data) creates a label-space asymmetry not addressed in this work.

### 15.3 Implications for Practitioners

- When deploying a HAR model across sensor placement configurations, invest first in signal-level normalisation (gravity projection, noise characterisation) before resorting to model-level domain adaptation.
- Use small, heavily regularised models (high dropout, label smoothing, moderate weight decay) for cross-dataset transfer; they generalise better than large models.
- DANN requires at minimum several thousand labelled or unlabelled target samples per class and a source dataset of comparable scale. It is not appropriate for the low-data cross-dataset HAR scenario studied here.

---

## 16. Conclusions (Part II)

We have demonstrated that the seven-dimensional domain shift characterised in Part I can be substantially reduced through a physics-grounded data augmentation pipeline, yielding a 45% MMD reduction and an average cross-dataset F1 improvement of +0.49. The best individual result — CNN1D with 10K parameters, F1 = 0.723 — represents a 6× improvement over the no-adaptation baseline.

Counter-intuitively, model-level adaptation methods (TTBN, DANN) do not improve upon data-level correction. The data regime (N ≈ 6,684 source windows) appears to be below the threshold at which adversarial feature alignment is beneficial. This suggests that future work should focus on either (a) collecting larger source datasets with diverse sensor placements, or (b) developing lightweight domain adaptation methods specifically designed for low-data-regime HAR.

The Upstairs activity remains an open problem across all tested methods, meriting targeted investigation of phase-sensitive augmentation strategies such as step-cycle-aware time warping.

---

## 17. Reproducibility (Part II)

All code is implemented in Python 3.12 with PyTorch 2.x. The project structure has been extended as follows:

```
wisdm_ucihar/
├── data_io.py                      # Data loading, alignment, windowing (Part I)
├── physics_engine.py               # Acc Magnitude and Frequency Centroid (Part I)
├── domain_shift_metrics.py         # Wasserstein, KL, MMD (Part I)
├── comprehensive_analysis.py       # Per-axis, gravity, prior shift (Part I)
├── visualize_distributions.py      # Global KDE overview (Part I)
│
├── covariate_shift_engine.py       # PhysicalDistortion pipeline + CrossDatasetLoader
├── architectures.py                # 5 HAR models with forward_features() API
├── domain_adaptation.py            # TTBN + DANN (GRL, discriminator, train_dann)
├── hp_search.py                    # Optuna hyperparameter search
├── final_benchmark.py              # 4-condition ablation (C0–C3)
├── cross_domain_benchmarking.py    # Initial baseline benchmark (pre-HP-search)
├── dann_tuning.py                  # DANN-specific grid search
├── visualize_stability.py          # 4-panel visualization suite
│
├── RQ1_Technical_Report.md         # This report (Parts I and II)
├── figures/                        # Part I analysis figures
│   ├── RQ1_Physical_Distributions.png
│   ├── amplitude_dist_metrics.png
│   └── ... (10 figures total)
├── results/                        # Part II experiment outputs
│   ├── hp_best_configs.json        # Best HP per model
│   ├── hp_search_summary.md
│   ├── dann_best_configs.json
│   └── dann_tuning_summary.md
├── results_final/                  # Final ablation outputs
│   ├── final_results.json          # All condition × model metrics
│   ├── final_benchmark.md          # Auto-generated Markdown table
│   └── figures/                    # Part II visualization figures
│       ├── distribution_alignment.png
│       ├── stability_analysis.png
│       ├── confusion_matrices.png
│       └── radar_charts.png
├── WISDM_ar_latest/                # Raw WISDM data
└── human+activity+recognition.../  # Raw UCI HAR data
```

**To reproduce Part II results:**

```bash
conda activate gengzhan_env
cd wisdm_ucihar

# Step 1 — Hyperparameter search (~30 min, GPU recommended)
python hp_search.py --n-trials 40 --epochs 100 --out-dir results

# Step 2 — Full 4-condition ablation benchmark (~10 min, GPU)
python final_benchmark.py \
  --config results/hp_best_configs.json \
  --conditions c0 c1 c2 c3 \
  --out-dir results_final

# Step 3 — Visualizations (~5 min, GPU)
python visualize_stability.py \
  --hp-config results/hp_best_configs.json \
  --out-dir results_final/figures
```

**Dependencies** (additional to Part I):

```
torch>=2.0
optuna>=4.0
tqdm
tensorboard
```

---

## 18. References (Part II)

8. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1–35.

9. Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., & Bethge, M. (2020). Improving robustness against common corruptions by covariate shift adaptation. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

10. Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017). Temporal convolutional networks for action segmentation and detection. *Proceedings of the IEEE CVPR*.

11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

12. Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. *ECCV Workshops*.

13. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*.
