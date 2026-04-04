#!/usr/bin/env python3
"""
dann_tuning.py — Focused hyperparameter search for DANN
=========================================================
Problem: DANN with the same LR as supervised training is unstable.
  - Task gradient + reversed domain gradient → larger effective update
  - dann_weight=0.5 makes domain loss dominate too early
  - Need lower LR and gentler domain weight for adversarial stability

Strategy: Grid search over DANN-specific params only.
  Base model config loaded from hp_best_configs.json (supervised HP search).
  Grid variables:
    dann_lr_scale  ∈ {0.15, 0.30, 0.50}   (multiply supervised LR)
    dann_weight    ∈ {0.10, 0.20, 0.35}   (domain loss weight λ)
  Fixed:
    patience = 20  (adversarial training is noisier, needs more patience)
    dann_epochs = 100

9 combos × 5 models × ~40s ≈ 30 min on GPU.

Output
------
  results/dann_best_configs.json  — best DANN config per model
  results/dann_tuning_summary.md  — grid table per model
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from architectures import get_model, count_params, _REGISTRY
from covariate_shift_engine import CrossDatasetLoader, PhysicalDistortion
from domain_adaptation import train_dann, evaluate_full

# ── Grid ─────────────────────────────────────────────────────────────────────
LR_SCALES    = [0.15, 0.30, 0.50]
DANN_WEIGHTS = [0.10, 0.20, 0.35]
DANN_EPOCHS  = 100
PATIENCE     = 20
SEED         = 42


def load_splits(seed: int = SEED) -> dict:
    rng  = np.random.default_rng(seed)
    dist = PhysicalDistortion(rng=rng)
    loader = CrossDatasetLoader(distortion=dist, verbose=False)
    src, wisdm_val, wisdm_test = loader.get_all()

    Xs, ys = src["X"], src["y"]
    idx_tr, idx_va = train_test_split(
        np.arange(len(Xs)), test_size=0.2, stratify=ys, random_state=seed
    )
    return {
        "aug_train":  (Xs[idx_tr], ys[idx_tr]),
        "aug_val":    (Xs[idx_va], ys[idx_va]),
        "wisdm_val":  (wisdm_val["X"],  wisdm_val["y"]),
        "wisdm_test": (wisdm_test["X"], wisdm_test["y"]),
    }


def load_base_configs(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] {path} not found — using default configs")
        return {}
    data = json.loads(path.read_text())
    return {name: {"model_kwargs": v.get("model_kwargs", {}),
                   "train_kwargs": v.get("train_kwargs", {})}
            for name, v in data.items()}


def run_tuning(
    models:     list[str],
    out_dir:    Path,
    device:     torch.device,
    hp_path:    Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[DANN Tuning] Loading data …")
    splits      = load_splits()
    base_configs = load_base_configs(hp_path)

    grid = list(itertools.product(LR_SCALES, DANN_WEIGHTS))
    print(f"[DANN Tuning] {len(grid)} combos × {len(models)} models "
          f"= {len(grid)*len(models)} runs\n")

    best_configs: dict  = {}
    all_results:  dict  = {}

    for name in models:
        print(f"\n{'='*60}")
        print(f"  Model: {name.upper()}")
        print(f"{'='*60}")

        hp       = base_configs.get(name, {})
        mkw      = hp.get("model_kwargs", {})
        tkw      = hp.get("train_kwargs", {})
        base_lr  = tkw.get("lr",              1e-3)
        wd       = tkw.get("weight_decay",    1e-4)
        ls       = tkw.get("label_smoothing", 0.0)
        bs       = tkw.get("batch_size",      128)

        print(f"  Base LR={base_lr:.1e}  WD={wd:.1e}  LS={ls:.2f}  BS={bs}")
        print(f"  {'lr_scale':>10} {'dann_w':>8} │ "
              f"{'val_F1':>8} {'test_F1':>8} {'stopped':>8} {'time':>6}")
        print(f"  {'─'*58}")

        grid_results = []
        best_val_f1  = -1.0
        best_cfg     = None

        for lr_scale, dann_weight in grid:
            dann_lr = base_lr * lr_scale
            t0 = time.time()

            model = get_model(name, **mkw).to(device)

            trained = train_dann(
                model, splits, device,
                epochs          = DANN_EPOCHS,
                patience        = PATIENCE,
                lr              = dann_lr,
                weight_decay    = wd,
                batch_size      = bs,
                label_smoothing = ls,
                dann_weight     = dann_weight,
                seed            = SEED,
                verbose         = False,
            )

            val_res  = evaluate_full(trained, *splits["wisdm_val"],  device)
            test_res = evaluate_full(trained, *splits["wisdm_test"], device)
            elapsed  = time.time() - t0

            val_f1  = val_res["macro_f1"]
            test_f1 = test_res["macro_f1"]

            # Find stopped_at from the training (we'll approximate with patience tracking)
            print(f"  {lr_scale:>10.2f} {dann_weight:>8.2f} │ "
                  f"{val_f1:>8.4f} {test_f1:>8.4f} {'?':>8} {elapsed:>5.0f}s")

            grid_results.append({
                "lr_scale":    lr_scale,
                "dann_lr":     dann_lr,
                "dann_weight": dann_weight,
                "val_f1":      round(val_f1,  5),
                "test_f1":     round(test_f1, 5),
                "time_s":      round(elapsed, 1),
                "per_class_f1": [round(v, 4) for v in test_res["per_class_f1"]],
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_cfg = {
                    "dann_lr_scale": lr_scale,
                    "dann_lr":       dann_lr,
                    "dann_weight":   dann_weight,
                    "val_f1":        round(val_f1,  5),
                    "test_f1":       round(test_f1, 5),
                    "per_class_f1":  [round(v, 4) for v in test_res["per_class_f1"]],
                    "model_kwargs":  mkw,
                    "base_train_kwargs": tkw,
                }

        print(f"\n  Best: lr_scale={best_cfg['dann_lr_scale']}  "
              f"dann_weight={best_cfg['dann_weight']}  "
              f"val_f1={best_cfg['val_f1']:.4f}  "
              f"test_f1={best_cfg['test_f1']:.4f}")

        best_configs[name] = best_cfg
        all_results[name]  = {"grid": grid_results, "best": best_cfg}

    # Save
    bp = out_dir / "dann_best_configs.json"
    bp.write_text(json.dumps(best_configs, indent=2))

    ap = out_dir / "dann_all_results.json"
    ap.write_text(json.dumps(all_results, indent=2))

    md = _make_summary(all_results, models)
    mp = out_dir / "dann_tuning_summary.md"
    mp.write_text(md)

    print(f"\n[Output] {bp}")
    print(f"[Output] {ap}")
    print(f"[Output] {mp}")
    print("\n" + "─" * 60)
    print(md)

    return best_configs


def _make_summary(all_results: dict, models: list[str]) -> str:
    import datetime
    lines = [
        "# DANN Hyperparameter Tuning Summary",
        f"\n_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "Grid: `lr_scale` ∈ {0.15, 0.30, 0.50} × `dann_weight` ∈ {0.10, 0.20, 0.35}\n",
        "Optimisation target: WISDM-Val Macro F1\n",
    ]

    for name in models:
        r     = all_results[name]
        best  = r["best"]
        lines += [
            f"\n## {name.upper()}\n",
            f"Best: `lr_scale={best['dann_lr_scale']}`, "
            f"`dann_weight={best['dann_weight']}`, "
            f"DANN-LR={best['dann_lr']:.2e}  →  "
            f"val F1={best['val_f1']:.4f}, test F1={best['test_f1']:.4f}\n",
            "| lr_scale | dann_weight | Val F1 | Test F1 |",
            "|---|---|---|---|",
        ]
        for g in sorted(r["grid"], key=lambda x: -x["val_f1"]):
            marker = " ← best" if (
                g["lr_scale"] == best["dann_lr_scale"] and
                g["dann_weight"] == best["dann_weight"]
            ) else ""
            lines.append(
                f"| {g['lr_scale']} | {g['dann_weight']} "
                f"| {g['val_f1']:.4f} | {g['test_f1']:.4f} |{marker}"
            )

    lines += [
        "\n## Final Best DANN Configs\n",
        "| Model | DANN LR | dann_weight | Val F1 | Test F1 |",
        "|---|---|---|---|---|",
    ]
    ranked = sorted(all_results.items(), key=lambda x: -x[1]["best"]["test_f1"])
    for name, r in ranked:
        b = r["best"]
        lines.append(
            f"| **{name.upper()}** | {b['dann_lr']:.2e} "
            f"| {b['dann_weight']} | {b['val_f1']:.4f} | {b['test_f1']:.4f} |"
        )

    lines.append("\n---\n_DANN HP tuning — WISDM v1.1 / UCI HAR_\n")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models",  nargs="+", default=list(_REGISTRY.keys()),
                   choices=list(_REGISTRY.keys()))
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / 'results')
    p.add_argument("--hp-config", type=Path, default=Path("results/hp_best_configs.json"))
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[DANN Tuning] device={args.device}  models={args.models}")
    print(f"  Grid: lr_scales={LR_SCALES}  dann_weights={DANN_WEIGHTS}")
    print(f"  Epochs={DANN_EPOCHS}  Patience={PATIENCE}\n")

    t0 = time.time()
    best = run_tuning(args.models, args.out_dir, device, args.hp_config)

    print(f"\n[Done] Total: {(time.time()-t0)/60:.1f} min")
    print("\nBest DANN configs (sorted by test F1):")
    ranked = sorted(best.items(), key=lambda x: -x[1]["test_f1"])
    for name, cfg in ranked:
        print(f"  {name:<14} val={cfg['val_f1']:.4f}  test={cfg['test_f1']:.4f}  "
              f"lr_scale={cfg['dann_lr_scale']}  dann_weight={cfg['dann_weight']}")
