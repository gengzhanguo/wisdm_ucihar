# Cross-Domain HAR Benchmark

_Generated: 2026-04-03 19:37_

## Experimental Setup

| Item | Detail |
|---|---|
| Source (train) | UCI HAR → PhysicalDistortion (pseudo-WISDM), 80% split |
| Source (val) | UCI HAR augmented, 20% split — used for early stopping |
| Target (test) | WISDM subjects 31–36 (held-out, never seen in training) |
| Activities | 5 shared classes: Walking, Upstairs, Downstairs, Sitting, Standing |
| Metric | Top-1 Accuracy + Macro F1-score |

## Distribution Distances (√MMD²)

| Split pair | √MMD² |
|---|---|
| `raw_uci → wisdm_test` | 0.8194 |
| `aug_uci → wisdm_test` | 0.4528 |
| `aug_uci → wisdm_val` | 0.3426 |
| `wisdm_val → wisdm_test` | 0.3054 |

> PhysicalDistortion reduces source→target MMD by **44.7%** (0.8194 → 0.4528)


## Main Results

| Model | Params | Aug-UCI(train) | Raw-UCI | WISDM-Val | WISDM-Test ★ | Train(s) |
| --- | --- | --- | --- | --- | --- | --- |
| | | Acc / F1 | Acc / F1 | Acc / F1 | Acc / F1 | |
| **FFN** | 140,037 | 93.0% / 0.936 | 29.6% / 0.219 | 46.5% / 0.489 | 55.8% / 0.557 | 9 |
| **CNN1D** | 100,677 | 90.4% / 0.911 | 25.5% / 0.166 | 50.7% / 0.469 | 68.8% / 0.551 | 4 |
| **BIGRU** | 109,573 | 90.3% / 0.910 | 24.1% / 0.165 | 43.5% / 0.390 | 53.3% / 0.472 | 8 |
| **TCN** | 116,997 | 92.0% / 0.927 | 17.8% / 0.104 | 46.9% / 0.486 | 60.6% / 0.573 | 10 |
| **TRANSFORMER** | 100,549 | 86.2% / 0.867 | 21.1% / 0.136 | 29.3% / 0.376 | 32.1% / 0.409 | 11 |

## Covariate Shift Sensitivity Analysis

Δ-Acc = Raw-UCI acc − WISDM-Test acc &nbsp;(↓ = more shift-sensitive)  
Δ-F1  = Raw-UCI F1  − WISDM-Test F1

| Model | Raw-UCI Acc | WISDM-Test Acc | Δ-Acc | Raw-UCI F1 | WISDM-Test F1 | Δ-F1 |
|---|---|---|---|---|---|---|
| **FFN** | 29.6% | 55.8% | -26.2% | 0.219 | 0.557 | -0.338 |
| **CNN1D** | 25.5% | 68.8% | -43.3% | 0.166 | 0.551 | -0.385 |
| **BIGRU** | 24.1% | 53.3% | -29.2% | 0.165 | 0.472 | -0.307 |
| **TCN** | 17.8% | 60.6% | -42.8% | 0.104 | 0.573 | -0.468 |
| **TRANSFORMER** | 21.1% | 32.1% | -11.0% | 0.136 | 0.409 | -0.273 |

## Augmentation Gain Analysis

Augmentation gain = WISDM-Test F1 (trained on aug-UCI) − WISDM-Test F1 (trained on raw-UCI).

> **Note:** 'Trained on raw-UCI' baseline requires separate run with `--no-distortion`; column shown as N/A until available.

| Model | Aug-UCI train F1 | WISDM-Test F1 | Source→Target gap |
|---|---|---|---|
| **FFN** | 0.936 | 0.557 | -0.379 |
| **CNN1D** | 0.911 | 0.551 | -0.359 |
| **BIGRU** | 0.910 | 0.472 | -0.439 |
| **TCN** | 0.927 | 0.573 | -0.355 |
| **TRANSFORMER** | 0.867 | 0.409 | -0.458 |

## Per-Class F1 on WISDM-Test (★)

| Model | Walking | Upstairs | Downstairs | Sitting | Standing | Macro F1 |
|---|---|---|---|---|---|---|
| **FFN** | 0.643 | 0.136 | 0.376 | 0.828 | 0.802 | **0.557** |
| **CNN1D** | 0.873 | 0.051 | 0.472 | 0.777 | 0.584 | **0.551** |
| **BIGRU** | 0.692 | 0.151 | 0.409 | 0.623 | 0.485 | **0.472** |
| **TCN** | 0.710 | 0.034 | 0.408 | 0.948 | 0.764 | **0.573** |
| **TRANSFORMER** | 0.178 | 0.167 | 0.363 | 0.713 | 0.622 | **0.409** |

## MMD vs Performance Summary

Source (aug-UCI) → Target (WISDM-Test) MMD = **0.4528**  

Lower WISDM-Test F1 relative to Aug-UCI F1 indicates the model is sensitive to the residual domain gap (MMD = 0.4528).

| Model | WISDM-Test F1 | Rank |
|---|---|---|
| **TCN** | 0.573 | 1 |
| **FFN** | 0.557 | 2 |
| **CNN1D** | 0.551 | 3 |
| **BIGRU** | 0.472 | 4 |
| **TRANSFORMER** | 0.409 | 5 |

---
_Cross-domain HAR benchmark — WISDM v1.1 / UCI HAR_
