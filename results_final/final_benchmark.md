# Final Cross-Domain HAR Benchmark

_Generated: 2026-04-03 20:55_

## Experimental Conditions

| ID | Description | Training data | Adaptation |
|---|---|---|---|
| C0 | Raw-UCI baseline    | UCI HAR (no transform) | None |
| C1 | PhysicalDistortion  | UCI HAR → pseudo-WISDM | None |
| C2 | +Test-Time BN       | UCI HAR → pseudo-WISDM | TTBN at inference |
| C3 | +DANN               | UCI HAR → pseudo-WISDM | Domain-adversarial training |

## Distribution Distances (MMD)

| Split pair | MMD |
|---|---|
| `raw_uci→wisdm_test` | 0.8297 |
| `aug_uci→wisdm_test` | 0.4528 |
| `wisdm_val→wisdm_test` | 0.3054 |

> PhysicalDistortion reduces source→target MMD by **45.4%**


## Main Results — WISDM-Test Macro F1

| Model | Params | Raw-UCI | Aug-UCI | Aug-UCI + TTBN | Aug-UCI + DANN |
|---|---|---|---|---|---|
| **FFN** | 81,797 | 0.069 / 12.5% | 0.542 / 52.8% | 0.425 / 44.2% | 0.401 / 43.2% |
| **CNN1D** | 10,277 | 0.120 / 18.3% | 0.723 / 76.9% | 0.610 / 65.3% | 0.581 / 64.7% |
| **BIGRU** | 51,429 | 0.062 / 11.6% | 0.512 / 47.6% | 0.512 / 47.6% | 0.365 / 56.7% |
| **TCN** | 67,709 | 0.127 / 18.5% | 0.535 / 56.5% | 0.451 / 47.2% | 0.418 / 54.9% |
| **TRANSFORMER** | 83,589 | 0.046 / 6.7% | 0.580 / 47.0% | 0.580 / 47.0% | 0.327 / 31.8% |

_Format: Macro F1 / Accuracy_


## Ablation — Gain Over C0 (Raw-UCI)

| Model | C0 F1 | C1 ΔF1 | C2 ΔF1 | C3 ΔF1 |
|---|---|---|---|---|
| **FFN** | 0.069 | +0.473 | +0.356 | +0.332 |
| **CNN1D** | 0.120 | +0.603 | +0.490 | +0.461 |
| **BIGRU** | 0.062 | +0.450 | +0.450 | +0.303 |
| **TCN** | 0.127 | +0.409 | +0.324 | +0.291 |
| **TRANSFORMER** | 0.046 | +0.535 | +0.535 | +0.282 |


## Per-Class F1 on WISDM-Test (Aug-UCI + DANN)

| Model | Walking | Upstairs | Downstairs | Sitting | Standing | Macro F1 |
|---|---|---|---|---|---|---|
| **FFN** | 0.552 | 0.147 | 0.314 | 0.627 | 0.364 | **0.401** |
| **CNN1D** | 0.798 | 0.349 | 0.276 | 0.803 | 0.678 | **0.581** |
| **BIGRU** | 0.833 | 0.233 | 0.102 | 0.316 | 0.342 | **0.365** |
| **TCN** | 0.760 | 0.287 | 0.105 | 0.550 | 0.389 | **0.418** |
| **TRANSFORMER** | 0.357 | 0.223 | 0.258 | 0.351 | 0.447 | **0.327** |

## Model Ranking (Best Condition)

| Rank | Model | Best F1 | Best Condition |
|---|---|---|---|
| 1 | **CNN1D** | 0.7232 | Aug-UCI (distort) |
| 2 | **TRANSFORMER** | 0.5804 | Aug-UCI (distort) |
| 3 | **FFN** | 0.5418 | Aug-UCI (distort) |
| 4 | **TCN** | 0.5352 | Aug-UCI (distort) |
| 5 | **BIGRU** | 0.5118 | Aug-UCI (distort) |

---
_Final benchmark — WISDM v1.1 / UCI HAR domain adaptation_
