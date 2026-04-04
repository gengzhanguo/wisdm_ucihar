#!/usr/bin/env python3
"""
final_benchmark.py — Full 4-condition Cross-Domain HAR Benchmark
=================================================================
Runs all 5 models under 4 conditions, generating a complete ablation
table that quantifies the contribution of each adaptation layer.

┌─────────────────────────────────────────────────────────────────┐
│  Condition  │  Training data     │  Adaptation at inference     │
├─────────────────────────────────────────────────────────────────┤
│  C0 (raw)   │  Raw UCI           │  None                        │
│  C1 (aug)   │  Aug UCI (Distort) │  None                        │
│  C2 (ttbn)  │  Aug UCI           │  Test-Time BN adaptation     │
│  C3 (dann)  │  Aug UCI           │  Domain-Adversarial training │
└─────────────────────────────────────────────────────────────────┘

Usage
-----
    python final_benchmark.py                          # all 4 conditions
    python final_benchmark.py --conditions c0 c1 c2   # skip DANN
    python final_benchmark.py --models cnn1d tcn
    python final_benchmark.py --fast                   # 25 ep, smoke test
    python final_benchmark.py --config results/hp_best_configs.json
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from architectures import get_model, count_params, _REGISTRY
from covariate_shift_engine import CrossDatasetLoader, PhysicalDistortion, compute_mmd
from domain_adaptation import apply_ttbn, train_dann, evaluate_full

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ── Activity names ────────────────────────────────────────────────────────────
ACTIVITY_NAMES = {0: "Walking", 1: "Upstairs", 2: "Downstairs", 3: "Sitting", 4: "Standing"}

# ── Conditions ────────────────────────────────────────────────────────────────
ALL_CONDITIONS = ["c0", "c1", "c2", "c3"]
COND_LABELS    = {
    "c0": "Raw-UCI (no distort)",
    "c1": "Aug-UCI (distort)",
    "c2": "Aug-UCI + TTBN",
    "c3": "Aug-UCI + DANN",
}


@dataclass
class Config:
    models:           list[str] = field(default_factory=lambda: list(_REGISTRY.keys()))
    conditions:       list[str] = field(default_factory=lambda: list(ALL_CONDITIONS))
    epochs:           int       = 100
    batch_size:       int       = 256
    lr:               float     = 1e-3
    weight_decay:     float     = 1e-4
    label_smoothing:  float     = 0.0
    patience:         int       = 15
    dann_epochs:      int       = 80
    dann_patience:    int       = 15
    dann_weight:      float     = 0.5
    seed:             int       = 42
    device:           str       = "cpu"
    out_dir:          Path      = Path("results_final")
    hp_config:        Path | None = None
    grad_clip:        float     = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(cfg: Config) -> dict:
    print("[Data] Loading splits …")
    rng  = np.random.default_rng(cfg.seed)
    dist = PhysicalDistortion(rng=rng)

    # Augmented source
    loader = CrossDatasetLoader(distortion=dist, verbose=True)
    src, wisdm_val, wisdm_test = loader.get_all()

    # Raw source
    raw_loader = CrossDatasetLoader(apply_distortion=False, verbose=False)
    raw_src    = raw_loader.get_source()

    Xs, ys = src["X"], src["y"]
    idx_tr, idx_va = train_test_split(
        np.arange(len(Xs)), test_size=0.2, stratify=ys, random_state=cfg.seed
    )

    Xr, yr = raw_src["X"], raw_src["y"]
    idx_rtr, idx_rva = train_test_split(
        np.arange(len(Xr)), test_size=0.2, stratify=yr, random_state=cfg.seed
    )

    splits = {
        "aug_train":  (Xs[idx_tr], ys[idx_tr]),
        "aug_val":    (Xs[idx_va], ys[idx_va]),
        "raw_train":  (Xr[idx_rtr], yr[idx_rtr]),
        "raw_val":    (Xr[idx_rva], yr[idx_rva]),
        "wisdm_val":  (wisdm_val["X"],  wisdm_val["y"]),
        "wisdm_test": (wisdm_test["X"], wisdm_test["y"]),
    }
    for k, (X, y) in splits.items():
        print(f"  {k:<14} {len(X):>7,}  {X.shape}")
    return splits


def load_hp_config(path: Path | None) -> dict:
    """Load best HP configs from hp_search output."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    configs = {}
    for name, cfg in data.items():
        configs[name] = {
            "model_kwargs": cfg.get("model_kwargs", {}),
            "train_kwargs": cfg.get("train_kwargs", {}),
        }
    print(f"[Config] Loaded HP configs for: {list(configs.keys())}")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Training (reusable for C0 raw and C1/C2/C3 aug)
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(X, y, batch_size, shuffle=True, drop_last=True):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0)


def train_supervised(
    model:           nn.Module,
    train_xy:        tuple,
    val_xy:          tuple,
    device:          torch.device,
    epochs:          int,
    patience:        int,
    lr:              float,
    weight_decay:    float,
    label_smoothing: float,
    batch_size:      int,
    grad_clip:       float,
    tb_writer        = None,
    tb_prefix:       str = "",
) -> tuple[nn.Module, dict]:
    """Supervised training with early stopping on val macro-F1."""
    from sklearn.metrics import f1_score

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    loader = _make_loader(*train_xy, batch_size=batch_size, drop_last=True)

    best_f1 = -1.0; best_state = None; patience_cnt = 0
    log = {"train_loss": [], "val_f1": [], "stopped_at": epochs}

    ep_iter = range(1, epochs + 1)
    if _HAS_TQDM:
        ep_iter = tqdm(ep_iter, desc=f"    train", ncols=72, leave=False)

    for epoch in ep_iter:
        model.train()
        total_loss = 0.0; n = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(Xb); n += len(Xb)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Val evaluation (aug-val for C1/C2/C3, raw-val for C0)
        model.eval()
        Xv, yv = val_xy
        with torch.no_grad():
            vl = _make_loader(Xv, yv, 512, shuffle=False, drop_last=False)
            preds = torch.cat([model(Xb.to(device)).argmax(1).cpu() for Xb, _ in vl]).numpy()
        val_f1 = float(f1_score(yv, preds, average="macro", zero_division=0))

        avg_loss = total_loss / n
        log["train_loss"].append(round(avg_loss, 5))
        log["val_f1"].append(round(val_f1, 5))

        if tb_writer:
            tb_writer.add_scalar(f"{tb_prefix}Loss/train", avg_loss, epoch)
            tb_writer.add_scalar(f"{tb_prefix}F1/val",     val_f1,   epoch)
            tb_writer.add_scalar(f"{tb_prefix}LR",         current_lr, epoch)

        if _HAS_TQDM:
            ep_iter.set_postfix(loss=f"{avg_loss:.4f}", vf1=f"{val_f1:.3f}", best=f"{best_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            log["stopped_at"] = epoch
            break

    model.load_state_dict(best_state)
    log["best_val_f1"] = round(best_f1, 5)
    return model, log


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    splits    = load_splits(cfg)
    hp_configs = load_hp_config(cfg.hp_config)

    # MMD distances
    print("\n[MMD] Computing distribution distances …")
    mmd = {
        "raw_uci→wisdm_test":  compute_mmd(splits["raw_train"][0], splits["wisdm_test"][0], n_sample=1500),
        "aug_uci→wisdm_test":  compute_mmd(splits["aug_train"][0], splits["wisdm_test"][0], n_sample=1500),
        "wisdm_val→wisdm_test": compute_mmd(splits["wisdm_val"][0], splits["wisdm_test"][0], n_sample=1500),
    }
    for k, v in mmd.items():
        print(f"  {k:<36}  MMD = {v:.4f}")
    mmd_reduction = (mmd["raw_uci→wisdm_test"] - mmd["aug_uci→wisdm_test"]) / mmd["raw_uci→wisdm_test"] * 100
    print(f"  Distortion MMD reduction: {mmd_reduction:.1f}%")

    # TB root writer
    tb_root = None
    if _HAS_TB:
        tb_root = SummaryWriter(log_dir=str(cfg.out_dir / "runs" / "_overview"))

    results: dict[str, dict] = {}

    for model_name in cfg.models:
        print(f"\n{'━'*64}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'━'*64}")

        # Resolve HP config
        hp = hp_configs.get(model_name, {})
        model_kwargs = hp.get("model_kwargs", {})
        train_kw     = hp.get("train_kwargs", {})
        lr            = train_kw.get("lr",              cfg.lr)
        weight_decay  = train_kw.get("weight_decay",    cfg.weight_decay)
        label_smooth  = train_kw.get("label_smoothing", cfg.label_smoothing)
        batch_size    = train_kw.get("batch_size",      cfg.batch_size)

        print(f"  HP: lr={lr:.1e}  wd={weight_decay:.1e}  "
              f"ls={label_smooth:.2f}  bs={batch_size}  "
              f"model_kwargs={model_kwargs}")

        results[model_name] = {"conditions": {}}
        tb_model = None
        if _HAS_TB:
            tb_model = SummaryWriter(log_dir=str(cfg.out_dir / "runs" / model_name))

        # ── C0: Raw UCI, no distortion ──────────────────────────────────────
        if "c0" in cfg.conditions:
            print(f"\n  ── C0: Raw-UCI (no distortion) ──")
            m0 = get_model(model_name, **model_kwargs).to(device)
            t0 = time.time()
            m0, log0 = train_supervised(
                m0, splits["raw_train"], splits["raw_val"], device,
                cfg.epochs, cfg.patience, lr, weight_decay, label_smooth, batch_size,
                cfg.grad_clip,
                tb_writer=tb_model, tb_prefix=f"C0/{model_name}/",
            )
            elapsed0 = time.time() - t0
            metrics0 = evaluate_full(m0, *splits["wisdm_test"], device)
            print(f"    WISDM-Test F1={metrics0['macro_f1']:.4f}  "
                  f"Acc={metrics0['acc']*100:.1f}%  "
                  f"ep={log0['stopped_at']}  t={elapsed0:.0f}s")
            results[model_name]["conditions"]["c0"] = {
                "label": COND_LABELS["c0"], "train_log": log0,
                "wisdm_test": {k: round(v if isinstance(v, float) else 0, 5)
                               for k, v in metrics0.items() if k != "per_class_f1"},
                "wisdm_test_per_class": [round(v, 4) for v in metrics0["per_class_f1"]],
                "train_time_s": round(elapsed0, 1),
            }

        # ── C1: Aug UCI ─────────────────────────────────────────────────────
        if any(c in cfg.conditions for c in ["c1", "c2", "c3"]):
            print(f"\n  ── C1: Aug-UCI (PhysicalDistortion) ──")
            m1 = get_model(model_name, **model_kwargs).to(device)
            t1 = time.time()
            m1, log1 = train_supervised(
                m1, splits["aug_train"], splits["aug_val"], device,
                cfg.epochs, cfg.patience, lr, weight_decay, label_smooth, batch_size,
                cfg.grad_clip,
                tb_writer=tb_model, tb_prefix=f"C1/{model_name}/",
            )
            elapsed1 = time.time() - t1
            metrics1 = evaluate_full(m1, *splits["wisdm_test"], device)
            print(f"    WISDM-Test F1={metrics1['macro_f1']:.4f}  "
                  f"Acc={metrics1['acc']*100:.1f}%  "
                  f"ep={log1['stopped_at']}  t={elapsed1:.0f}s")

            if "c1" in cfg.conditions:
                results[model_name]["conditions"]["c1"] = {
                    "label": COND_LABELS["c1"], "train_log": log1,
                    "wisdm_test": {k: round(v if isinstance(v, float) else 0, 5)
                                   for k, v in metrics1.items() if k != "per_class_f1"},
                    "wisdm_test_per_class": [round(v, 4) for v in metrics1["per_class_f1"]],
                    "train_time_s": round(elapsed1, 1),
                }

        # ── C2: Aug UCI + TTBN ──────────────────────────────────────────────
        if "c2" in cfg.conditions:
            print(f"\n  ── C2: Aug-UCI + Test-Time BN ──")
            m2 = copy.deepcopy(m1)
            apply_ttbn(m2, splits["wisdm_test"][0], device)   # adapt to test batch
            metrics2 = evaluate_full(m2, *splits["wisdm_test"], device)
            print(f"    WISDM-Test F1={metrics2['macro_f1']:.4f}  "
                  f"Acc={metrics2['acc']*100:.1f}%")
            results[model_name]["conditions"]["c2"] = {
                "label": COND_LABELS["c2"],
                "wisdm_test": {k: round(v if isinstance(v, float) else 0, 5)
                               for k, v in metrics2.items() if k != "per_class_f1"},
                "wisdm_test_per_class": [round(v, 4) for v in metrics2["per_class_f1"]],
            }
            if tb_model:
                tb_model.add_scalar(f"Final/C2_wisdm_test_f1", metrics2["macro_f1"], 0)

        # ── C3: Aug UCI + DANN ──────────────────────────────────────────────
        if "c3" in cfg.conditions:
            print(f"\n  ── C3: Aug-UCI + DANN ──")
            m3 = get_model(model_name, **model_kwargs).to(device)
            t3 = time.time()
            m3 = train_dann(
                m3, splits, device,
                epochs        = cfg.dann_epochs,
                patience      = cfg.dann_patience,
                lr            = lr,
                weight_decay  = weight_decay,
                batch_size    = batch_size,
                label_smoothing = label_smooth,
                dann_weight   = cfg.dann_weight,
                seed          = cfg.seed,
                verbose       = True,
            )
            elapsed3 = time.time() - t3
            metrics3 = evaluate_full(m3, *splits["wisdm_test"], device)
            print(f"    WISDM-Test F1={metrics3['macro_f1']:.4f}  "
                  f"Acc={metrics3['acc']*100:.1f}%  t={elapsed3:.0f}s")
            results[model_name]["conditions"]["c3"] = {
                "label": COND_LABELS["c3"],
                "wisdm_test": {k: round(v if isinstance(v, float) else 0, 5)
                               for k, v in metrics3.items() if k != "per_class_f1"},
                "wisdm_test_per_class": [round(v, 4) for v in metrics3["per_class_f1"]],
                "train_time_s": round(elapsed3, 1),
            }
            if tb_model:
                tb_model.add_scalar(f"Final/C3_wisdm_test_f1", metrics3["macro_f1"], 0)

        results[model_name]["n_params"] = count_params(get_model(model_name, **model_kwargs))

        if tb_model:
            tb_model.flush(); tb_model.close()

    # ── Save ────────────────────────────────────────────────────────────────
    out = {"config": {
               "models": cfg.models, "conditions": cfg.conditions,
               "epochs": cfg.epochs, "seed": cfg.seed,
           },
           "mmd":     mmd,
           "mmd_reduction_pct": round(mmd_reduction, 2),
           "results": results}

    jpath = cfg.out_dir / "final_results.json"
    jpath.write_text(json.dumps(out, indent=2))

    md   = generate_report(results, mmd, mmd_reduction, cfg)
    mdpath = cfg.out_dir / "final_benchmark.md"
    mdpath.write_text(md)

    if tb_root:
        tb_root.flush(); tb_root.close()

    print(f"\n[Output] JSON   → {jpath}")
    print(f"[Output] Report → {mdpath}")
    print("\n" + "─" * 64)
    print(md)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: dict, mmd: dict, mmd_reduction: float, cfg: Config) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    conds_present = cfg.conditions
    model_names   = list(results.keys())

    lines = [
        "# Final Cross-Domain HAR Benchmark",
        f"\n_Generated: {now}_\n",

        "## Experimental Conditions\n",
        "| ID | Description | Training data | Adaptation |",
        "|---|---|---|---|",
        "| C0 | Raw-UCI baseline    | UCI HAR (no transform) | None |",
        "| C1 | PhysicalDistortion  | UCI HAR → pseudo-WISDM | None |",
        "| C2 | +Test-Time BN       | UCI HAR → pseudo-WISDM | TTBN at inference |",
        "| C3 | +DANN               | UCI HAR → pseudo-WISDM | Domain-adversarial training |\n",

        "## Distribution Distances (MMD)\n",
        "| Split pair | MMD |",
        "|---|---|",
    ]
    for k, v in mmd.items():
        lines.append(f"| `{k}` | {v:.4f} |")
    lines.append(
        f"\n> PhysicalDistortion reduces source→target MMD by **{mmd_reduction:.1f}%**\n"
    )

    # ── Main results table ──
    lines += ["\n## Main Results — WISDM-Test Macro F1\n"]
    cond_cols = [c for c in ALL_CONDITIONS if c in conds_present]
    header = ["Model", "Params"] + [COND_LABELS[c].split("(")[0].strip() for c in cond_cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for name in model_names:
        r   = results[name]
        row = [f"**{name.upper()}**", f"{r.get('n_params', '?'):,}"]
        for c in cond_cols:
            cdata = r["conditions"].get(c, {})
            wt    = cdata.get("wisdm_test", {})
            f1    = wt.get("macro_f1", float("nan"))
            acc   = wt.get("acc", float("nan"))
            row.append(f"{f1:.3f} / {acc*100:.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n_Format: Macro F1 / Accuracy_\n")

    # ── Ablation: gain from each condition ──
    lines += ["\n## Ablation — Gain Over C0 (Raw-UCI)\n"]
    if "c0" in conds_present:
        lines += [
            "| Model | C0 F1 | C1 ΔF1 | C2 ΔF1 | C3 ΔF1 |",
            "|---|---|---|---|---|",
        ]
        for name in model_names:
            r   = results[name]
            f0  = r["conditions"].get("c0", {}).get("wisdm_test", {}).get("macro_f1", float("nan"))
            row = [f"**{name.upper()}**", f"{f0:.3f}"]
            for c in ["c1", "c2", "c3"]:
                if c in conds_present:
                    fc = r["conditions"].get(c, {}).get("wisdm_test", {}).get("macro_f1", float("nan"))
                    delta = fc - f0
                    row.append(f"{delta:+.3f}")
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ── Per-class F1 on best condition (C3 > C2 > C1 > C0) ──
    best_cond = next((c for c in ["c3","c2","c1","c0"] if c in conds_present), None)
    if best_cond:
        lines += [
            f"\n## Per-Class F1 on WISDM-Test ({COND_LABELS[best_cond]})\n",
            "| Model | " + " | ".join(ACTIVITY_NAMES[i] for i in range(5)) + " | Macro F1 |",
            "|" + "---|" * 7,
        ]
        for name in model_names:
            r  = results[name]
            pc = r["conditions"].get(best_cond, {}).get("wisdm_test_per_class", [float("nan")] * 5)
            f1 = r["conditions"].get(best_cond, {}).get("wisdm_test", {}).get("macro_f1", float("nan"))
            row = [f"**{name.upper()}**"] + [f"{v:.3f}" for v in pc] + [f"**{f1:.3f}**"]
            lines.append("| " + " | ".join(row) + " |")

    # ── Ranking ──
    lines += ["\n## Model Ranking (Best Condition)\n", "| Rank | Model | Best F1 | Best Condition |", "|---|---|---|---|"]
    ranking = []
    for name in model_names:
        best_f1 = -1.0; best_c = "—"
        for c in conds_present:
            f1 = results[name]["conditions"].get(c, {}).get("wisdm_test", {}).get("macro_f1", -1.0)
            if f1 > best_f1:
                best_f1 = f1; best_c = c
        ranking.append((name, best_f1, best_c))
    for rank, (name, f1, c) in enumerate(sorted(ranking, key=lambda x: -x[1]), 1):
        lines.append(f"| {rank} | **{name.upper()}** | {f1:.4f} | {COND_LABELS.get(c, c)} |")

    lines.append("\n---\n_Final benchmark — WISDM v1.1 / UCI HAR domain adaptation_\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models",     nargs="+", default=list(_REGISTRY.keys()), choices=list(_REGISTRY.keys()))
    p.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--dann-epochs",   type=int,   default=80)
    p.add_argument("--dann-patience", type=int,   default=15)
    p.add_argument("--dann-weight",   type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",    type=Path,  default=Path("results_final"))
    p.add_argument("--config",     type=Path,  default=None,
                   help="Path to hp_best_configs.json from hp_search.py")
    p.add_argument("--fast",       action="store_true", help="25 epochs, patience=8, skip DANN")
    args = p.parse_args()

    if args.fast:
        args.epochs       = 25
        args.patience     = 8
        args.dann_epochs  = 20
        args.dann_patience = 6
        args.conditions   = [c for c in args.conditions if c != "c3"]

    return Config(
        models          = args.models,
        conditions      = args.conditions,
        epochs          = args.epochs,
        patience        = args.patience,
        dann_epochs     = args.dann_epochs,
        dann_patience   = args.dann_patience,
        dann_weight     = args.dann_weight,
        seed            = args.seed,
        device          = args.device,
        out_dir         = args.out_dir,
        hp_config       = args.config,
    )


if __name__ == "__main__":
    cfg = _parse()
    print(f"[Config] device={cfg.device}  conditions={cfg.conditions}  "
          f"models={cfg.models}")
    t_total = time.time()
    run_benchmark(cfg)
    print(f"\n[Total] Elapsed: {(time.time()-t_total)/60:.1f} min")
