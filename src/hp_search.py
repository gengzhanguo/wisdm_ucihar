#!/usr/bin/env python3
"""
hp_search.py — Hyperparameter search for cross-domain HAR
==========================================================
Uses Optuna (TPE sampler) to find the best hyperparameters for each model.

Optimisation target : Macro F1 on WISDM-Val (subjects 1-30).
  • This is an oracle metric (uses target-domain labels for model selection).
  • In practice this approximates what a practitioner would do with a small
    labelled validation set from the target domain.

Search space (per model)
------------------------
Global (all models):
  lr            log-uniform  [1e-4, 5e-3]
  weight_decay  log-uniform  [1e-5, 1e-2]
  dropout       uniform      [0.1, 0.5]
  label_smoothing  uniform   [0.0, 0.2]
  batch_size    categorical  [64, 128, 256]

Model-specific (size):
  FFN         hidden_dims  — 3 choices of (h1, h2, h3)
  CNN1D       channels     — 3 choices of (c1, c2, c3)
  BiGRU       hidden_size  — int [24, 128], num_layers [1, 2]
  TCN         n_channels   — int [16, 96], kernel_size [3, 5]
  Transformer d_model      — categorical [32, 48, 64]
              num_layers   — [1, 2, 3], dim_feedforward factor [2, 4, 8]

Output
------
  hp_search_results.json     — all trials per model
  hp_best_configs.json       — best config per model (use in final benchmark)
  hp_search_summary.md       — Markdown table

Usage
-----
  python hp_search.py                         # 30 trials/model, all models
  python hp_search.py --models tcn transformer --n-trials 50
  python hp_search.py --fast                  # 10 trials, 20 epochs each
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from architectures import get_model, count_params, _REGISTRY
from covariate_shift_engine import (
    CrossDatasetLoader, PhysicalDistortion, SHARED_LABEL_NAMES
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Data  (loaded once, shared across all trials)
# ─────────────────────────────────────────────────────────────────────────────

_SPLITS: dict | None = None


def get_splits(seed: int = 42) -> dict:
    global _SPLITS
    if _SPLITS is not None:
        return _SPLITS

    print("[HP] Loading data (once) …")
    rng  = np.random.default_rng(seed)
    dist = PhysicalDistortion(rng=rng)
    loader = CrossDatasetLoader(distortion=dist, verbose=True)
    src, wisdm_val, wisdm_test = loader.get_all()

    X_s, y_s = src["X"], src["y"]
    idx_tr, idx_va = train_test_split(
        np.arange(len(X_s)), test_size=0.2, stratify=y_s, random_state=seed
    )

    _SPLITS = {
        "aug_train":  (X_s[idx_tr], y_s[idx_tr]),
        "aug_val":    (X_s[idx_va], y_s[idx_va]),
        "wisdm_val":  (wisdm_val["X"],  wisdm_val["y"]),
        "wisdm_test": (wisdm_test["X"], wisdm_test["y"]),
    }
    print(f"  aug_train={len(idx_tr):,}  aug_val={len(idx_va):,}  "
          f"wisdm_val={len(wisdm_val['X']):,}  wisdm_test={len(wisdm_test['X']):,}")
    return _SPLITS


def _make_loader(X, y, batch_size, shuffle=True, drop_last=False):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────────────────────

# FFN hidden_dims presets — (small, medium, large)
_FFN_PRESETS = [
    (64, 64),
    (128, 64),
    (128, 128, 64),
    (256, 128, 64),
    (256, 256, 128),
]

# CNN1D channels presets
_CNN_PRESETS = [
    (32, 32, 32),
    (32, 64, 64),
    (64, 64, 64),
    (64, 128, 64),
    (64, 128, 128),
]


def suggest_model_kwargs(trial: optuna.Trial, name: str) -> dict:
    """Return model constructor kwargs suggested by Optuna."""
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    if name == "ffn":
        idx = trial.suggest_categorical("preset_idx", list(range(len(_FFN_PRESETS))))
        return {"hidden_dims": _FFN_PRESETS[idx], "dropout": dropout}

    elif name == "cnn1d":
        idx = trial.suggest_categorical("preset_idx", list(range(len(_CNN_PRESETS))))
        return {"channels": _CNN_PRESETS[idx], "dropout": dropout}

    elif name == "bigru":
        hidden_size = trial.suggest_int("hidden_size", 24, 96, step=8)
        num_layers  = trial.suggest_categorical("num_layers", [1, 2])
        return {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}

    elif name == "tcn":
        n_channels  = trial.suggest_int("n_channels", 16, 80, step=8)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 4, 5])
        return {"n_channels": n_channels, "kernel_size": kernel_size, "dropout": dropout}

    elif name == "transformer":
        d_model = trial.suggest_categorical("d_model", [32, 48, 64])
        nhead_choices = [h for h in [2, 4, 8] if d_model % h == 0]
        nhead   = trial.suggest_categorical("nhead", nhead_choices)
        n_layers = trial.suggest_int("num_layers", 1, 3)
        ff_factor = trial.suggest_categorical("ff_factor", [2, 4, 8])
        return {
            "d_model":         d_model,
            "nhead":           nhead,
            "num_layers":      n_layers,
            "dim_feedforward": d_model * ff_factor,
            "dropout":         dropout,
        }
    else:
        return {"dropout": dropout}


def suggest_train_kwargs(trial: optuna.Trial) -> dict:
    """Suggest training hyperparameters."""
    return {
        "lr":               trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "label_smoothing":  trial.suggest_float("label_smoothing", 0.0, 0.2),
        "batch_size":       trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training (compact, optimised for speed during HP search)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_f1(model, X, y, device, batch_size=512):
    from sklearn.metrics import f1_score
    model.eval()
    loader = _make_loader(X, y, batch_size, shuffle=False)
    preds  = []
    for X_b, _ in loader:
        preds.append(model(X_b.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return float(f1_score(y, preds, average="macro", zero_division=0))


def train_and_eval(
    model_name:      str,
    model_kwargs:    dict,
    train_kwargs:    dict,
    splits:          dict,
    epochs:          int,
    patience:        int,
    device:          torch.device,
    seed:            int = 0,
    trial:           optuna.Trial | None = None,
) -> dict:
    """
    Train model and return metrics on all splits.
    Uses WISDM-val F1 as the primary metric (oracle).
    """
    torch.manual_seed(seed)

    model     = get_model(model_name, **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_kwargs.get("label_smoothing", 0.0)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_kwargs["lr"],
        weight_decay=train_kwargs["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=train_kwargs["lr"] * 0.01
    )

    tr_loader = _make_loader(
        *splits["aug_train"],
        batch_size=train_kwargs["batch_size"],
        shuffle=True, drop_last=True,
    )

    best_wisdm_f1  = -1.0
    best_state     = None
    patience_cnt   = 0
    stopped_at     = epochs
    loss_curve     = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
            n += len(X_b)
        scheduler.step()
        loss_curve.append(round(epoch_loss / n, 5))

        # Evaluate on WISDM-val (oracle criterion)
        wv_f1 = _eval_f1(model, *splits["wisdm_val"], device)

        if wv_f1 > best_wisdm_f1:
            best_wisdm_f1 = wv_f1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        # Optuna pruning (prune unpromising trials early)
        if trial is not None:
            trial.report(wv_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_cnt >= patience:
            stopped_at = epoch
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # Evaluate on all splits
    results = {
        "best_wisdm_val_f1": round(best_wisdm_f1, 5),
        "stopped_at":        stopped_at,
        "n_params":          count_params(model),
        "loss_curve":        loss_curve,
    }
    for split_key in ("aug_val", "wisdm_val", "wisdm_test"):
        X, y = splits[split_key]
        from sklearn.metrics import accuracy_score, f1_score
        model.eval()
        with torch.no_grad():
            loader = _make_loader(X, y, 512, shuffle=False)
            preds  = []
            for X_b, _ in loader:
                preds.append(model(X_b.to(device)).argmax(1).cpu())
        preds = torch.cat(preds).numpy()
        results[f"{split_key}_acc"] = round(float(accuracy_score(y, preds)), 5)
        results[f"{split_key}_f1"]  = round(float(f1_score(y, preds, average="macro", zero_division=0)), 5)
        results[f"{split_key}_per_class_f1"] = [
            round(v, 4) for v in
            f1_score(y, preds, average=None, zero_division=0).tolist()
        ]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(model_name, splits, epochs, patience, device, seed):
    def objective(trial: optuna.Trial) -> float:
        model_kwargs = suggest_model_kwargs(trial, model_name)
        train_kwargs = suggest_train_kwargs(trial)
        try:
            res = train_and_eval(
                model_name, model_kwargs, train_kwargs,
                splits, epochs, patience, device,
                seed=seed, trial=trial,
            )
            # Log to trial user_attrs for later inspection
            trial.set_user_attr("n_params",        res["n_params"])
            trial.set_user_attr("stopped_at",       res["stopped_at"])
            trial.set_user_attr("wisdm_test_f1",    res.get("wisdm_test_f1", -1))
            trial.set_user_attr("model_kwargs",     str(model_kwargs))
            trial.set_user_attr("train_kwargs",     str(train_kwargs))
            return res["best_wisdm_val_f1"]
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial failed: {e}")
            return 0.0

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main search loop
# ─────────────────────────────────────────────────────────────────────────────

def run_search(
    model_names: list[str],
    n_trials:    int,
    epochs:      int,
    patience:    int,
    device:      torch.device,
    seed:        int,
    out_dir:     Path,
) -> dict:
    splits      = get_splits(seed)
    all_results = {}
    best_configs = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  HP Search: {name.upper()}  ({n_trials} trials × {epochs} epochs)")
        print(f"{'='*60}")
        t0 = time.time()

        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner  = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=2
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"har_{name}",
        )

        obj = make_objective(name, splits, epochs, patience, device, seed)
        study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

        elapsed = time.time() - t0
        best    = study.best_trial

        print(f"\n  Best trial #{best.number}:")
        print(f"    WISDM-val F1 : {best.value:.4f}")
        print(f"    Params       : {best.user_attrs.get('n_params', '?'):,}")
        print(f"    Stopped at   : ep {best.user_attrs.get('stopped_at', '?')}")
        print(f"    model_kwargs : {best.user_attrs.get('model_kwargs', best.params)}")
        print(f"    train_kwargs : {best.user_attrs.get('train_kwargs', '')}")
        print(f"    Search time  : {elapsed:.0f}s")

        # Reconstruct best kwargs from params
        best_model_kw = suggest_model_kwargs_from_params(name, best.params)
        best_train_kw = {
            "lr":              best.params.get("lr", 1e-3),
            "weight_decay":    best.params.get("weight_decay", 1e-4),
            "label_smoothing": best.params.get("label_smoothing", 0.0),
            "batch_size":      best.params.get("batch_size", 256),
        }

        # Final evaluation with best config
        print(f"\n  Running final eval with best config …")
        final_res = train_and_eval(
            name, best_model_kw, best_train_kw,
            splits, epochs * 2, patience * 2,   # more epochs for final run
            device, seed=seed,
        )
        print(f"    WISDM-val F1  : {final_res['wisdm_val_f1']:.4f}")
        print(f"    WISDM-test F1 : {final_res['wisdm_test_f1']:.4f}")
        print(f"    aug-val F1    : {final_res['aug_val_f1']:.4f}")
        print(f"    Stopped at ep : {final_res['stopped_at']}")
        print(f"    Params        : {final_res['n_params']:,}")

        all_results[name] = {
            "n_trials":      n_trials,
            "best_val_f1":   best.value,
            "best_trial_no": best.number,
            "final_eval":    final_res,
            "trials": [
                {
                    "number":    t.number,
                    "value":     t.value,
                    "state":     str(t.state),
                    "params":    t.params,
                    "n_params":  t.user_attrs.get("n_params"),
                    "stopped_at": t.user_attrs.get("stopped_at"),
                }
                for t in study.trials
            ],
        }
        best_configs[name] = {
            "model_kwargs": best_model_kw,
            "train_kwargs": best_train_kw,
            "wisdm_val_f1_search": best.value,
            "wisdm_val_f1_final":  final_res["wisdm_val_f1"],
            "wisdm_test_f1_final": final_res["wisdm_test_f1"],
            "n_params":            final_res["n_params"],
            "best_epoch":          final_res["stopped_at"],
        }

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hp_search_results.json").write_text(json.dumps(all_results, indent=2))
    (out_dir / "hp_best_configs.json").write_text(json.dumps(best_configs, indent=2))

    md = _generate_summary(best_configs)
    (out_dir / "hp_search_summary.md").write_text(md)

    print("\n" + "─" * 60)
    print(md)
    return best_configs


def suggest_model_kwargs_from_params(name: str, params: dict) -> dict:
    """Reconstruct model kwargs dict from Optuna trial params dict."""
    dropout = params.get("dropout", 0.3)

    if name == "ffn":
        idx = params.get("preset_idx", 2)
        return {"hidden_dims": _FFN_PRESETS[idx], "dropout": dropout}
    elif name == "cnn1d":
        idx = params.get("preset_idx", 1)
        return {"channels": _CNN_PRESETS[idx], "dropout": dropout}
    elif name == "bigru":
        return {
            "hidden_size": params.get("hidden_size", 64),
            "num_layers":  params.get("num_layers", 2),
            "dropout":     dropout,
        }
    elif name == "tcn":
        return {
            "n_channels":  params.get("n_channels", 64),
            "kernel_size": params.get("kernel_size", 4),
            "dropout":     dropout,
        }
    elif name == "transformer":
        d_model = params.get("d_model", 64)
        return {
            "d_model":         d_model,
            "nhead":           params.get("nhead", 4),
            "num_layers":      params.get("num_layers", 2),
            "dim_feedforward": d_model * params.get("ff_factor", 4),
            "dropout":         dropout,
        }
    return {"dropout": dropout}


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary
# ─────────────────────────────────────────────────────────────────────────────

def _generate_summary(best_configs: dict) -> str:
    import datetime
    lines = [
        "# HP Search Summary",
        f"\n_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "Optimisation target: **WISDM-Val Macro F1** (oracle, using target-domain labels for model selection).\n",
        "## Best Configuration Per Model\n",
        "| Model | Params | Best Epoch | WISDM-Val F1 | WISDM-Test F1 | Key Hyperparams |",
        "|---|---|---|---|---|---|",
    ]
    for name, cfg in best_configs.items():
        mk  = cfg["model_kwargs"]
        tk  = cfg["train_kwargs"]
        kw_str = (
            f"lr={tk['lr']:.0e}, wd={tk['weight_decay']:.0e}, "
            f"drop={mk.get('dropout', '?'):.2f}, "
            f"ls={tk.get('label_smoothing',0):.2f}, bs={tk['batch_size']}"
        )
        lines.append(
            f"| **{name.upper()}** "
            f"| {cfg['n_params']:,} "
            f"| {cfg['best_epoch']} "
            f"| {cfg['wisdm_val_f1_final']:.4f} "
            f"| {cfg['wisdm_test_f1_final']:.4f} "
            f"| `{kw_str}` |"
        )

    lines += [
        "\n## Architecture Config Per Model\n",
        "| Model | Architecture kwargs |",
        "|---|---|",
    ]
    for name, cfg in best_configs.items():
        lines.append(f"| **{name.upper()}** | `{cfg['model_kwargs']}` |")

    lines.append("\n---\n_HAR HP search — Optuna TPE + Median Pruner_\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna HP search for cross-domain HAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+",
                        default=list(_REGISTRY.keys()),
                        choices=list(_REGISTRY.keys()))
    parser.add_argument("--n-trials",  type=int, default=30,
                        help="Optuna trials per model.")
    parser.add_argument("--epochs",    type=int, default=80,
                        help="Max epochs per trial.")
    parser.add_argument("--patience",  type=int, default=12,
                        help="Early stopping patience (WISDM-val F1).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir",   type=Path, default=Path(__file__).resolve().parent.parent / 'results')
    parser.add_argument("--fast",      action="store_true",
                        help="10 trials, 25 epochs (smoke test).")
    args = parser.parse_args()

    if args.fast:
        args.n_trials = 10
        args.epochs   = 25
        args.patience = 8

    return args


if __name__ == "__main__":
    args = _parse_args()
    print(f"[HP] device={args.device}  trials/model={args.n_trials}  "
          f"epochs/trial={args.epochs}  patience={args.patience}")

    best = run_search(
        model_names = args.models,
        n_trials    = args.n_trials,
        epochs      = args.epochs,
        patience    = args.patience,
        device      = torch.device(args.device),
        seed        = args.seed,
        out_dir     = args.out_dir,
    )

    print("\n[HP] Done. Best WISDM-test F1 per model:")
    ranked = sorted(best.items(), key=lambda x: x[1]["wisdm_test_f1_final"], reverse=True)
    for name, cfg in ranked:
        print(f"  {name:<14} val={cfg['wisdm_val_f1_final']:.4f}  "
              f"test={cfg['wisdm_test_f1_final']:.4f}  "
              f"params={cfg['n_params']:,}  ep={cfg['best_epoch']}")
