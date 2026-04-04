# DANN Hyperparameter Tuning Summary

_Generated: 2026-04-03 21:59_

Grid: `lr_scale` ∈ {0.15, 0.30, 0.50} × `dann_weight` ∈ {0.10, 0.20, 0.35}

Optimisation target: WISDM-Val Macro F1


## CNN1D

Best: `lr_scale=0.5`, `dann_weight=0.1`, DANN-LR=4.29e-04  →  val F1=0.4291, test F1=0.4945

| lr_scale | dann_weight | Val F1 | Test F1 |
|---|---|---|---|
| 0.5 | 0.1 | 0.4291 | 0.4945 | ← best
| 0.5 | 0.35 | 0.4126 | 0.5490 |
| 0.15 | 0.35 | 0.4078 | 0.4772 |
| 0.15 | 0.2 | 0.4022 | 0.5226 |
| 0.3 | 0.1 | 0.3961 | 0.3799 |
| 0.5 | 0.2 | 0.3799 | 0.3194 |
| 0.3 | 0.2 | 0.3617 | 0.4140 |
| 0.15 | 0.1 | 0.3327 | 0.2627 |
| 0.3 | 0.35 | 0.3262 | 0.2624 |

## Final Best DANN Configs

| Model | DANN LR | dann_weight | Val F1 | Test F1 |
|---|---|---|---|---|
| **CNN1D** | 4.29e-04 | 0.1 | 0.4291 | 0.4945 |

---
_DANN HP tuning — WISDM v1.1 / UCI HAR_
