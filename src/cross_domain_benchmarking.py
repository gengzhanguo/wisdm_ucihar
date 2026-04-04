#!/usr/bin/env python3
"""
cross_domain_benchmarking.py
============================
Automated training and cross-domain evaluation of all HAR architectures.

Experiment design
-----------------
Train  : UCI HAR + PhysicalDistortion  →  "pseudo-WISDM" (augmented source)
         80% aug-UCI for training, 20% aug-UCI for early stopping
Test   : Real WISDM subjects 31-36  (never seen during training)

Sensitivity analysis (per model, 4 splits):
  1. Aug-UCI train split   — sanity check, upper bound on source
  2. Raw UCI               — no distortion; measures how much augmentation helps
  3. WISDM-Val             — subjects 1-30, reference generalisation score
  4. WISDM-Test            — subjects 31-36, primary target metric

MMD analysis:
  Computes MMD between source and target before/after PhysicalDistortion.
  Checks whether architectures with better domain robustness benefit more
  from augmentation (lower MMD → higher WISDM-Test F1).

Output:
  results/cross_domain_results.json
  results/cross_domain_benchmark.md

Usage
-----
    python cross_domain_benchmarking.py                   # all models, 100 epochs
    python cross_domain_benchmarking.py --fast            # 25 epochs, patience=8
    python cross_domain_benchmarking.py --models ffn tcn  # subset of models
    python cross_domain_benchmarking.py --device cuda     # GPU training
    python cross_domain_benchmarking.py --no-mmd          # skip MMD (saves ~1 min)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from architectures import get_model, count_params, _REGISTRY
from covariate_shift_engine import (
    CrossDatasetLoader, PhysicalDistortion, SHARED_LABEL_NAMES, compute_mmd
)

# Optional tqdm
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    models:       list[str] = field(default_factory=lambda: list(_REGISTRY.keys()))
    epochs:       int       = 100
    batch_size:   int       = 256
    lr:           float     = 1e-3
    weight_decay: float     = 1e-4
    patience:     int       = 15
    seed:         int       = 42
    device:       str       = "cpu"
    out_dir:      Path      = Path("results")
    run_mmd:      bool      = True
    mmd_samples:  int       = 1500
    grad_clip:    float     = 1.0
    log_every:    int       = 10    # print metrics every N epochs
    tb_wisdm_every: int     = 5     # log WISDM-val metrics to TB every N epochs


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )


def load_all_splits(cfg: Config) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load and return all five data splits as (X, y) numpy arrays.

    Returns
    -------
    dict with keys:
        "aug_train"   : 80% of augmented UCI  (train)
        "aug_val"     : 20% of augmented UCI  (early-stop val)
        "raw_uci"     : UCI HAR, no distortion
        "wisdm_val"   : WISDM subjects 1–30
        "wisdm_test"  : WISDM subjects 31–36  (primary target)
    """
    print("[Data] Loading all splits …")

    rng  = np.random.default_rng(cfg.seed)
    dist = PhysicalDistortion(rng=rng)

    # Augmented UCI + WISDM
    loader = CrossDatasetLoader(distortion=dist, verbose=True)
    src, wisdm_val, wisdm_test = loader.get_all()

    # 80/20 stratified split of augmented UCI
    X_s, y_s = src["X"], src["y"]
    idx_tr, idx_va = train_test_split(
        np.arange(len(X_s)),
        test_size=0.2,
        stratify=y_s,
        random_state=cfg.seed,
    )

    # Raw UCI (no distortion)
    loader_raw  = CrossDatasetLoader(apply_distortion=False, verbose=False)
    raw_src     = loader_raw.get_source()

    splits = {
        "aug_train":  (X_s[idx_tr], y_s[idx_tr]),
        "aug_val":    (X_s[idx_va], y_s[idx_va]),
        "raw_uci":    (raw_src["X"], raw_src["y"]),
        "wisdm_val":  (wisdm_val["X"],  wisdm_val["y"]),
        "wisdm_test": (wisdm_test["X"], wisdm_test["y"]),
    }

    print(f"\n  {'Split':<14} {'N':>7}  {'Shape'}")
    print(f"  {'─'*40}")
    for name, (X, y) in splits.items():
        print(f"  {name:<14} {len(X):>7,}  {X.shape}")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# MMD analysis  (data-level, model-independent)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mmd_distances(
    splits: dict,
    cfg: Config,
) -> dict[str, float]:
    """
    Compute MMD between key split pairs to quantify distribution distances.
    All values are √MMD² (non-negative).
    """
    print("\n[MMD] Computing distribution distances …")
    pairs = [
        ("raw_uci",   "wisdm_test", "raw_uci → wisdm_test"),
        ("aug_train", "wisdm_test", "aug_uci → wisdm_test"),
        ("aug_train", "wisdm_val",  "aug_uci → wisdm_val"),
        ("wisdm_val", "wisdm_test", "wisdm_val → wisdm_test"),
    ]
    mmd = {}
    for k1, k2, label in pairs:
        val = compute_mmd(
            splits[k1][0], splits[k2][0],
            n_sample=cfg.mmd_samples,
        )
        mmd[label] = val
        print(f"  {label:<38}  MMD = {val:.4f}")

    if "raw_uci → wisdm_test" in mmd and "aug_uci → wisdm_test" in mmd:
        raw = mmd["raw_uci → wisdm_test"]
        aug = mmd["aug_uci → wisdm_test"]
        print(f"\n  MMD reduction from PhysicalDistortion: "
              f"{raw:.4f} → {aug:.4f}  "
              f"({(raw-aug)/raw*100:.1f}% reduction)")
    return mmd


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_total    = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_b), y_b)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X_b)
        n_total    += len(X_b)
    return total_loss / n_total


def train_model(
    name:   str,
    cfg:    Config,
    splits: dict,
) -> tuple[nn.Module, dict]:
    """
    Train a single model and return (best_model, training_log).

    Early stopping is based on macro F1 on the augmented UCI val split
    (no WISDM labels used during training / model selection).

    TensorBoard layout (runs/<name>/):
        Loss/train            — CE loss per epoch
        Acc/aug_val           — accuracy on aug-UCI val
        F1/aug_val            — macro F1 on aug-UCI val (early-stop criterion)
        Acc/wisdm_val         — accuracy on WISDM val (every tb_wisdm_every epochs)
        F1/wisdm_val          — macro F1 on WISDM val
        F1_class/wisdm_val/<activity> — per-class F1 on WISDM val
        LR/lr                 — learning rate schedule
    """
    print(f"\n{'='*58}")
    print(f"  Training: {name.upper()}  ({cfg.epochs} epochs, "
          f"patience={cfg.patience})")
    print(f"{'='*58}")

    device    = torch.device(cfg.device)
    model     = get_model(name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
    )

    tr_loader = DataLoader(
        _to_tensor_dataset(*splits["aug_train"]),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=0,
    )

    # TensorBoard writer (one sub-dir per model)
    tb_writer = None
    if _HAS_TB:
        tb_dir = cfg.out_dir / "runs" / name
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        # Log model graph with dummy input
        try:
            dummy = torch.zeros(1, 51, 3).to(device)
            tb_writer.add_graph(model, dummy)
        except Exception:
            pass

    best_val_f1  = -1.0
    best_state   = None
    patience_cnt = 0
    log          = {"epochs": [], "train_loss": [], "val_f1": [], "val_acc": []}
    t_start      = time.time()

    epoch_iter = range(1, cfg.epochs + 1)
    if _HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc=f"  {name}", ncols=80, leave=False)

    for epoch in epoch_iter:
        train_loss = _train_one_epoch(
            model, tr_loader, optimizer, criterion, device, cfg.grad_clip
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ── Evaluate on aug_val (early-stop criterion) ───────────────────
        val_res = evaluate_split(model, *splits["aug_val"], device)

        log["epochs"].append(epoch)
        log["train_loss"].append(round(train_loss, 5))
        log["val_f1"].append(round(val_res["f1"], 5))
        log["val_acc"].append(round(val_res["acc"], 5))

        # ── TensorBoard: per-epoch scalars ───────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train",   train_loss,    epoch)
            tb_writer.add_scalar("F1/aug_val",   val_res["f1"], epoch)
            tb_writer.add_scalar("Acc/aug_val",  val_res["acc"],epoch)
            tb_writer.add_scalar("LR/lr",        current_lr,    epoch)

        # ── TensorBoard: WISDM-val metrics (periodic, more expensive) ────
        if tb_writer is not None and (epoch % cfg.tb_wisdm_every == 0 or epoch == 1):
            wv = evaluate_split(model, *splits["wisdm_val"], device)
            tb_writer.add_scalar("F1/wisdm_val",  wv["f1"],  epoch)
            tb_writer.add_scalar("Acc/wisdm_val", wv["acc"], epoch)
            for cls_id, cls_name in SHARED_LABEL_NAMES.items():
                if cls_id < len(wv["per_class_f1"]):
                    tb_writer.add_scalar(
                        f"F1_class/{cls_name.lower()}/wisdm_val",
                        wv["per_class_f1"][cls_id],
                        epoch,
                    )

        # ── Early stopping ───────────────────────────────────────────────
        if val_res["f1"] > best_val_f1:
            best_val_f1  = val_res["f1"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if not _HAS_TQDM and (epoch % cfg.log_every == 0 or epoch == 1):
            print(f"  [{epoch:3d}/{cfg.epochs}]  loss={train_loss:.4f}  "
                  f"val_f1={val_res['f1']:.4f}  best={best_val_f1:.4f}  "
                  f"pat={patience_cnt}/{cfg.patience}")
        elif _HAS_TQDM:
            epoch_iter.set_postfix(
                loss=f"{train_loss:.4f}",
                vf1=f"{val_res['f1']:.3f}",
                best=f"{best_val_f1:.3f}",
                pat=f"{patience_cnt}",
            )

        if patience_cnt >= cfg.patience:
            if not _HAS_TQDM:
                print(f"  Early stop at epoch {epoch} "
                      f"(best val_f1={best_val_f1:.4f})")
            log["stopped_at"] = epoch
            break

    elapsed = time.time() - t_start
    log["best_val_f1"]  = round(best_val_f1, 5)
    log["train_time_s"] = round(elapsed, 1)

    # Restore best weights
    model.load_state_dict(best_state)

    # ── TensorBoard: final evaluation on all splits ──────────────────────
    if tb_writer is not None:
        for split_key, split_label in EVAL_SPLITS:
            res = evaluate_split(model, *splits[split_key], device)
            tb_writer.add_scalar(f"Final/acc_{split_key}",  res["acc"], 0)
            tb_writer.add_scalar(f"Final/f1_{split_key}",   res["f1"],  0)
            for cls_id, cls_name in SHARED_LABEL_NAMES.items():
                if cls_id < len(res["per_class_f1"]):
                    tb_writer.add_scalar(
                        f"Final_classF1/{cls_name.lower()}/{split_key}",
                        res["per_class_f1"][cls_id], 0
                    )
        tb_writer.add_hparams(
            hparam_dict={
                "model":        name,
                "epochs":       cfg.epochs,
                "batch_size":   cfg.batch_size,
                "lr":           cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
            metric_dict={
                "hparam/wisdm_test_f1":  log.get("wisdm_test_f1", best_val_f1),
                "hparam/best_val_f1":    best_val_f1,
            },
        )
        tb_writer.flush()
        tb_writer.close()

    # ── Save checkpoint ──────────────────────────────────────────────────
    ckpt_dir = cfg.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / f"{name}_best.pt")

    print(f"\n  Finished in {elapsed:.0f}s  best_val_f1={best_val_f1:.4f}"
          + (f"  TB → {cfg.out_dir}/runs/{name}" if tb_writer else ""))
    return model, log


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_split(
    model:      nn.Module,
    X:          np.ndarray,
    y:          np.ndarray,
    device:     torch.device,
    batch_size: int = 512,
) -> dict:
    """
    Evaluate model on a (X, y) split.

    Returns
    -------
    dict with: acc, f1, per_class_f1, confusion_matrix, preds
    """
    model.eval()
    loader = DataLoader(
        _to_tensor_dataset(X, y),
        batch_size=batch_size, shuffle=False,
    )
    preds = []
    for X_b, _ in loader:
        preds.append(model(X_b.to(device)).argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()

    acc          = float(accuracy_score(y, preds))
    macro_f1     = float(f1_score(y, preds, average="macro",    zero_division=0))
    per_class_f1 = f1_score(y, preds, average=None, zero_division=0).tolist()
    cm           = confusion_matrix(y, preds, labels=list(range(5))).tolist()

    return {
        "acc":           acc,
        "f1":            macro_f1,
        "per_class_f1":  per_class_f1,
        "confusion":     cm,
        "preds":         preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

# Display order and labels for the four evaluation splits
EVAL_SPLITS = [
    ("aug_train",  "Aug-UCI(train)"),
    ("raw_uci",    "Raw-UCI"),
    ("wisdm_val",  "WISDM-Val"),
    ("wisdm_test", "WISDM-Test ★"),
]


def run_benchmark(cfg: Config) -> dict:
    """
    Main entry point: trains all models, evaluates on all splits,
    computes MMD, and writes results.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"\n[Config] device={cfg.device}  seed={cfg.seed}  "
          f"epochs={cfg.epochs}  patience={cfg.patience}")
    print(f"[Config] models: {cfg.models}\n")

    # ── Data ───────────────────────────────────────────────────────────────
    splits = load_all_splits(cfg)

    # ── MMD ────────────────────────────────────────────────────────────────
    mmd_values: dict[str, float] = {}
    if cfg.run_mmd:
        mmd_values = compute_mmd_distances(splits, cfg)

    # ── Per-model loop ─────────────────────────────────────────────────────
    results: dict = {}
    for name in cfg.models:
        model, train_log = train_model(name, cfg, splits)

        # Evaluate on all splits
        split_metrics: dict = {}
        print(f"\n  Evaluation — {name.upper()}")
        print(f"  {'Split':<22}  {'Acc':>7}  {'MacroF1':>9}  {'Per-class F1'}")
        print(f"  {'─'*72}")

        for split_key, split_label in EVAL_SPLITS:
            X, y = splits[split_key]
            res  = evaluate_split(model, X, y, device)

            pf1_str = "  ".join(f"{v:.2f}" for v in res["per_class_f1"])
            print(f"  {split_label:<22}  {res['acc']*100:>6.2f}%  "
                  f"{res['f1']:>9.4f}  [{pf1_str}]")

            split_metrics[split_key] = {
                "label":        split_label,
                "n":            int(len(y)),
                "acc":          round(res["acc"], 5),
                "macro_f1":     round(res["f1"],  5),
                "per_class_f1": [round(v, 4) for v in res["per_class_f1"]],
                "confusion":    res["confusion"],
            }

        # Feed WISDM-Test F1 back into train_log for hparam panel
        if "wisdm_test" in split_metrics:
            train_log["wisdm_test_f1"] = split_metrics["wisdm_test"]["macro_f1"]

        results[name] = {
            "params":     count_params(model),
            "train_log":  train_log,
            "splits":     split_metrics,
        }

    # ── Persistence ────────────────────────────────────────────────────────
    output = {"config": {
                  "models":       cfg.models,
                  "epochs":       cfg.epochs,
                  "batch_size":   cfg.batch_size,
                  "lr":           cfg.lr,
                  "weight_decay": cfg.weight_decay,
                  "patience":     cfg.patience,
                  "seed":         cfg.seed,
              },
              "mmd":     mmd_values,
              "results": results}

    json_path = cfg.out_dir / "cross_domain_results.json"
    json_path.write_text(json.dumps(output, indent=2))

    md_path = cfg.out_dir / "cross_domain_benchmark.md"
    md      = _generate_markdown(results, mmd_values)
    md_path.write_text(md)

    print(f"\n[Output] JSON   → {json_path}")
    print(f"[Output] Report → {md_path}")
    print("\n" + "─" * 70)
    print(md)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_markdown(
    results:    dict,
    mmd_values: dict,
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    model_names = list(results.keys())
    label_map   = {k: v for k, v in SHARED_LABEL_NAMES.items()}

    lines = [
        "# Cross-Domain HAR Benchmark",
        f"\n_Generated: {now}_\n",

        "## Experimental Setup\n",
        "| Item | Detail |",
        "|---|---|",
        "| Source (train) | UCI HAR → PhysicalDistortion (pseudo-WISDM), 80% split |",
        "| Source (val) | UCI HAR augmented, 20% split — used for early stopping |",
        "| Target (test) | WISDM subjects 31–36 (held-out, never seen in training) |",
        "| Activities | 5 shared classes: Walking, Upstairs, Downstairs, Sitting, Standing |",
        "| Metric | Top-1 Accuracy + Macro F1-score |\n",
    ]

    # ── MMD table ─────────────────────────────────────────────────────────
    if mmd_values:
        lines += [
            "## Distribution Distances (√MMD²)\n",
            "| Split pair | √MMD² |",
            "|---|---|",
        ]
        for k, v in mmd_values.items():
            lines.append(f"| `{k}` | {v:.4f} |")

        if ("raw_uci → wisdm_test" in mmd_values
                and "aug_uci → wisdm_test" in mmd_values):
            raw = mmd_values["raw_uci → wisdm_test"]
            aug = mmd_values["aug_uci → wisdm_test"]
            pct = (raw - aug) / raw * 100
            lines.append(
                f"\n> PhysicalDistortion reduces source→target MMD by "
                f"**{pct:.1f}%** ({raw:.4f} → {aug:.4f})\n"
            )

    # ── Main results table ─────────────────────────────────────────────────
    lines += ["\n## Main Results\n"]
    col_heads = ["Model", "Params"]
    for _, label in EVAL_SPLITS:
        col_heads.append(f"{label}")
    col_heads.append("Train(s)")

    lines.append("| " + " | ".join(col_heads) + " |")
    lines.append("| " + " | ".join(["---"] * len(col_heads)) + " |")
    lines.append("| | | " + " | ".join(["Acc / F1"] * len(EVAL_SPLITS)) + " | |")

    for name in model_names:
        r   = results[name]
        row = [f"**{name.upper()}**", f"{r['params']:,}"]
        for split_key, _ in EVAL_SPLITS:
            s   = r["splits"].get(split_key, {})
            acc = s.get("acc", float("nan"))
            f1  = s.get("macro_f1", float("nan"))
            row.append(f"{acc*100:.1f}% / {f1:.3f}")
        row.append(f"{r['train_log'].get('train_time_s', '—'):.0f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Covariate shift sensitivity ─────────────────────────────────────────
    lines += [
        "\n## Covariate Shift Sensitivity Analysis\n",
        "Δ-Acc = Raw-UCI acc − WISDM-Test acc &nbsp;(↓ = more shift-sensitive)  \n"
        "Δ-F1  = Raw-UCI F1  − WISDM-Test F1\n",
        "| Model | Raw-UCI Acc | WISDM-Test Acc | Δ-Acc | Raw-UCI F1 | WISDM-Test F1 | Δ-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in model_names:
        r    = results[name]
        raw  = r["splits"].get("raw_uci", {})
        tgt  = r["splits"].get("wisdm_test", {})
        ra   = raw.get("acc", float("nan"))
        ta   = tgt.get("acc", float("nan"))
        rf1  = raw.get("macro_f1", float("nan"))
        tf1  = tgt.get("macro_f1", float("nan"))
        da   = ra - ta
        df1  = rf1 - tf1
        lines.append(
            f"| **{name.upper()}** "
            f"| {ra*100:.1f}% | {ta*100:.1f}% | {da*100:+.1f}% "
            f"| {rf1:.3f} | {tf1:.3f} | {df1:+.3f} |"
        )

    # ── Augmentation gain analysis ──────────────────────────────────────────
    lines += [
        "\n## Augmentation Gain Analysis\n",
        "Augmentation gain = WISDM-Test F1 (trained on aug-UCI) − "
        "WISDM-Test F1 (trained on raw-UCI).\n",
        "> **Note:** 'Trained on raw-UCI' baseline requires separate run "
        "with `--no-distortion`; column shown as N/A until available.\n",
        "| Model | Aug-UCI train F1 | WISDM-Test F1 | Source→Target gap |",
        "|---|---|---|---|",
    ]
    for name in model_names:
        r    = results[name]
        af1  = r["splits"].get("aug_train", {}).get("macro_f1", float("nan"))
        tf1  = r["splits"].get("wisdm_test", {}).get("macro_f1", float("nan"))
        gap  = tf1 - af1
        lines.append(
            f"| **{name.upper()}** | {af1:.3f} | {tf1:.3f} | {gap:+.3f} |"
        )

    # ── Per-class F1 on WISDM-Test ──────────────────────────────────────────
    lines += [
        "\n## Per-Class F1 on WISDM-Test (★)\n",
        "| Model | "
        + " | ".join(label_map[i] for i in range(5))
        + " | Macro F1 |",
        "|---" * (7) + "|",
    ]
    for name in model_names:
        r   = results[name]
        pc  = r["splits"].get("wisdm_test", {}).get("per_class_f1", [float("nan")] * 5)
        mf1 = r["splits"].get("wisdm_test", {}).get("macro_f1", float("nan"))
        row = [f"**{name.upper()}**"] + [f"{v:.3f}" for v in pc] + [f"**{mf1:.3f}**"]
        lines.append("| " + " | ".join(row) + " |")

    # ── MMD vs F1 correlation table ──────────────────────────────────────────
    if mmd_values and "aug_uci → wisdm_test" in mmd_values:
        aug_mmd = mmd_values["aug_uci → wisdm_test"]
        lines += [
            "\n## MMD vs Performance Summary\n",
            f"Source (aug-UCI) → Target (WISDM-Test) MMD = **{aug_mmd:.4f}**  \n",
            "Lower WISDM-Test F1 relative to Aug-UCI F1 indicates the model "
            "is sensitive to the residual domain gap (MMD = "
            f"{aug_mmd:.4f}).\n",
            "| Model | WISDM-Test F1 | Rank |",
            "|---|---|---|",
        ]
        ranked = sorted(
            model_names,
            key=lambda n: results[n]["splits"].get("wisdm_test", {}).get("macro_f1", 0),
            reverse=True,
        )
        for rank, name in enumerate(ranked, 1):
            tf1 = results[name]["splits"].get("wisdm_test", {}).get("macro_f1", float("nan"))
            lines.append(f"| **{name.upper()}** | {tf1:.3f} | {rank} |")

    lines.append("\n---\n_Cross-domain HAR benchmark — WISDM v1.1 / UCI HAR_\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Cross-domain HAR benchmark: train on pseudo-WISDM, test on WISDM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(_REGISTRY.keys()),
        choices=list(_REGISTRY.keys()),
        help="Models to train and evaluate.",
    )
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cpu / cuda / mps).",
    )
    parser.add_argument("--out-dir",  type=Path, default=Path(__file__).resolve().parent.parent / 'results')
    parser.add_argument("--no-mmd",   action="store_true",
                        help="Skip MMD computation (faster run).")
    parser.add_argument("--mmd-samples", type=int, default=1500)
    parser.add_argument("--log-every",      type=int, default=10)
    parser.add_argument("--tb-wisdm-every", type=int, default=5,
                        help="Log WISDM-val metrics to TensorBoard every N epochs.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test: 25 epochs, patience=8. Overrides --epochs/--patience.",
    )

    args = parser.parse_args()

    if args.fast:
        args.epochs   = 25
        args.patience = 8

    return Config(
        models       = args.models,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        seed         = args.seed,
        device       = args.device,
        out_dir      = args.out_dir,
        run_mmd      = not args.no_mmd,
        mmd_samples  = args.mmd_samples,
        log_every       = args.log_every,
        tb_wisdm_every  = args.tb_wisdm_every,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = _parse_args()
    t0  = time.time()
    run_benchmark(cfg)
    print(f"\n[Total] Elapsed: {(time.time()-t0)/60:.1f} min")
