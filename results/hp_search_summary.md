# HP Search Summary

_Generated: 2026-04-03 20:19_

Optimisation target: **WISDM-Val Macro F1** (oracle, using target-domain labels for model selection).

## Best Configuration Per Model

| Model | Params | Best Epoch | WISDM-Val F1 | WISDM-Test F1 | Key Hyperparams |
|---|---|---|---|---|---|
| **FFN** | 81,797 | 68 | 0.5457 | 0.6228 | `lr=1e-03, wd=1e-03, drop=0.17, ls=0.15, bs=128` |
| **CNN1D** | 10,277 | 55 | 0.6071 | 0.7616 | `lr=9e-04, wd=4e-03, drop=0.39, ls=0.14, bs=128` |
| **BIGRU** | 51,429 | 44 | 0.4849 | 0.5588 | `lr=5e-04, wd=4e-05, drop=0.13, ls=0.02, bs=64` |
| **TCN** | 67,709 | 44 | 0.6085 | 0.6976 | `lr=2e-03, wd=2e-05, drop=0.23, ls=0.11, bs=256` |
| **TRANSFORMER** | 83,589 | 87 | 0.4892 | 0.5281 | `lr=5e-03, wd=8e-05, drop=0.13, ls=0.04, bs=64` |

## Architecture Config Per Model

| Model | Architecture kwargs |
|---|---|
| **FFN** | `{'hidden_dims': (256, 128, 64), 'dropout': 0.16546313440034688}` |
| **CNN1D** | `{'channels': (32, 32, 32), 'dropout': 0.391641942262628}` |
| **BIGRU** | `{'hidden_size': 80, 'num_layers': 1, 'dropout': 0.12877513242219973}` |
| **TCN** | `{'n_channels': 56, 'kernel_size': 3, 'dropout': 0.2294961261317829}` |
| **TRANSFORMER** | `{'d_model': 64, 'nhead': 2, 'num_layers': 1, 'dim_feedforward': 512, 'dropout': 0.12900908479202913}` |

---
_HAR HP search — Optuna TPE + Median Pruner_
