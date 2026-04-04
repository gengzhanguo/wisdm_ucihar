#!/usr/bin/env python3
"""
visualize_stability.py — Four-panel experimental visualization
==============================================================
Generates four publication-quality figures:

  Fig 1  distribution_alignment.png
         PCA scatter of raw-UCI / aug-UCI / WISDM  showing how
         PhysicalDistortion moves the source into the target region.

  Fig 2  stability_analysis.png
         Two sub-plots:
           Left  — Macro F1 vs noise multiplier (1×…5×)
           Right — Macro F1 vs rotation angle   (0°…90°)
         Applied to WISDM-test; models trained on aug-UCI.

  Fig 3  confusion_matrices.png
         Normalised 5×5 confusion matrices for all 5 models
         evaluated on WISDM-test (C1 condition).

  Fig 4  radar_charts.png
         Per-activity F1 radar chart for each model (C1 condition)
         and an overlay comparing all models.

Usage
-----
    python visualize_stability.py                         # full run
    python visualize_stability.py --skip-train            # load cached ckpts
    python visualize_stability.py --out-dir figs --fast   # quick smoke test

All figures saved to --out-dir (default: results_final/figures).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import ConvexHull

from architectures import get_model, count_params, _REGISTRY
from covariate_shift_engine import CrossDatasetLoader, PhysicalDistortion

warnings.filterwarnings("ignore")

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

MODEL_NAMES    = list(_REGISTRY.keys())
ACTIVITY_NAMES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing"]
N_CLASSES      = 5

MODEL_COLORS = {
    "ffn":         "#E74C3C",   # red
    "cnn1d":       "#2ECC71",   # green
    "bigru":       "#3498DB",   # blue
    "tcn":         "#F39C12",   # orange
    "transformer": "#9B59B6",   # purple
}
MODEL_MARKERS = {
    "ffn": "o", "cnn1d": "s", "bigru": "^", "tcn": "D", "transformer": "P"
}
DATASET_COLORS = {
    "Raw UCI":  "#E74C3C",
    "Aug UCI":  "#F39C12",
    "WISDM":    "#2ECC71",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(seed: int = 42) -> dict:
    print("[Data] Loading …")
    rng  = np.random.default_rng(seed)
    dist = PhysicalDistortion(rng=rng)

    # Augmented source + WISDM
    loader = CrossDatasetLoader(distortion=dist, verbose=True)
    src, wisdm_val, wisdm_test = loader.get_all()

    # Raw source
    raw_loader = CrossDatasetLoader(apply_distortion=False, verbose=False)
    raw_src    = raw_loader.get_source()

    Xs, ys = src["X"], src["y"]
    idx_tr, idx_va = train_test_split(
        np.arange(len(Xs)), test_size=0.2, stratify=ys, random_state=seed
    )

    return {
        "aug_train":  (Xs[idx_tr], ys[idx_tr]),
        "aug_val":    (Xs[idx_va], ys[idx_va]),
        "raw_uci":    (raw_src["X"], raw_src["y"]),
        "wisdm_val":  (wisdm_val["X"],  wisdm_val["y"]),
        "wisdm_test": (wisdm_test["X"], wisdm_test["y"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model training / loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hp_configs(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        print(f"[HP] Loaded configs for: {list(data.keys())}")
        return data
    print(f"[HP] {path} not found — using defaults")
    return {}


def train_c1_models(
    splits:      dict,
    hp_configs:  dict,
    ckpt_dir:    Path,
    device:      torch.device,
    epochs:      int = 100,
    patience:    int = 15,
    seed:        int = 42,
) -> dict[str, nn.Module]:
    """Train all models under C1 condition (aug-UCI). Cache checkpoints."""
    from sklearn.metrics import f1_score as sk_f1

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    models = {}

    for name in MODEL_NAMES:
        ckpt = ckpt_dir / f"{name}_c1.pt"
        hp   = hp_configs.get(name, {})
        mkw  = hp.get("model_kwargs", {})
        tkw  = hp.get("train_kwargs", {})
        lr   = tkw.get("lr",              1e-3)
        wd   = tkw.get("weight_decay",    1e-4)
        ls   = tkw.get("label_smoothing", 0.0)
        bs   = tkw.get("batch_size",      256)

        model = get_model(name, **mkw).to(device)

        if ckpt.exists():
            print(f"  [Load] {name} ← {ckpt.name}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            models[name] = model
            continue

        print(f"  [Train] {name}  params={count_params(model):,}  "
              f"lr={lr:.1e}  epochs={epochs}")

        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        Xtr, ytr = splits["aug_train"]
        Xva, yva = splits["aug_val"]
        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr).float(),
                          torch.from_numpy(ytr).long()),
            batch_size=bs, shuffle=True, drop_last=True,
        )

        best_f1 = -1.0; best_state = None; patience_cnt = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                nn.CrossEntropyLoss(label_smoothing=ls)(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                vl = DataLoader(
                    TensorDataset(torch.from_numpy(Xva).float(),
                                  torch.from_numpy(yva).long()),
                    batch_size=512, shuffle=False,
                )
                preds = torch.cat([model(Xb.to(device)).argmax(1).cpu()
                                   for Xb, _ in vl]).numpy()
            vf1 = float(sk_f1(yva, preds, average="macro", zero_division=0))

            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= patience:
                break

        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)
        model.eval()
        models[name] = model
        print(f"    best_val_f1={best_f1:.4f}  stopped_ep={epoch}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device,
            batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                        batch_size=batch_size, shuffle=False)
    return torch.cat([model(Xb.to(device)).argmax(1).cpu()
                      for (Xb,) in loader]).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Distribution alignment (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a 2-sigma confidence ellipse for (x, y) points."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-9)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_distribution_alignment(splits: dict, out_path: Path,
                                 n_sample: int = 1500) -> None:
    print("[Fig1] PCA distribution alignment …")

    rng = np.random.default_rng(0)

    def _sample(X, y, n):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx].reshape(len(idx), -1), y[idx]

    X_raw,  y_raw  = _sample(*splits["raw_uci"],   n_sample)
    X_aug,  y_aug  = _sample(*splits["aug_train"],  n_sample)
    X_wis,  y_wis  = _sample(*splits["wisdm_test"], n_sample)

    X_all = np.vstack([X_raw, X_aug, X_wis])
    pca   = PCA(n_components=2, random_state=42)
    Z     = pca.fit_transform(X_all)
    ev    = pca.explained_variance_ratio_

    n0, n1, n2 = len(X_raw), len(X_aug), len(X_wis)
    Z_raw = Z[:n0];         Z_aug = Z[n0:n0+n1]; Z_wis = Z[n0+n1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: dataset overlay ─────────────────────────────────────────────
    ax = axes[0]
    datasets = [
        ("Raw UCI",  Z_raw, "#E74C3C", 0.35),
        ("Aug UCI",  Z_aug, "#F39C12", 0.35),
        ("WISDM",    Z_wis, "#2ECC71", 0.35),
    ]
    for label, Z_d, color, alpha in datasets:
        ax.scatter(Z_d[:, 0], Z_d[:, 1], c=color, s=8, alpha=alpha, label=label)
        _confidence_ellipse(Z_d[:, 0], Z_d[:, 1], ax, n_std=2,
                            edgecolor=color, facecolor="none",
                            linewidth=2.0, linestyle="--")

    # Centroid arrows: raw→aug→wisdm
    c_raw = Z_raw.mean(0); c_aug = Z_aug.mean(0); c_wis = Z_wis.mean(0)
    ax.annotate("", xy=c_aug, xytext=c_raw,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.annotate("", xy=c_wis, xytext=c_aug,
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.scatter(*c_raw, marker="*", s=200, c="#E74C3C", zorder=5)
    ax.scatter(*c_aug, marker="*", s=200, c="#F39C12", zorder=5)
    ax.scatter(*c_wis, marker="*", s=200, c="#2ECC71", zorder=5)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title("Dataset Overlap in PCA Space\n(★ = centroid, dashed = 2σ ellipse)")
    ax.legend(markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ── Right: per-activity scatter for WISDM vs Aug-UCI ─────────────────
    ax2 = axes[1]
    act_colors = ["#3498DB","#E67E22","#E74C3C","#9B59B6","#1ABC9C"]
    markers_d  = {"Aug UCI": "o", "WISDM": "s"}

    for ds_name, Z_d, y_d in [("Aug UCI", Z_aug, y_aug), ("WISDM", Z_wis, y_wis)]:
        for cls in range(N_CLASSES):
            mask = y_d == cls
            ax2.scatter(Z_d[mask, 0], Z_d[mask, 1],
                        c=act_colors[cls], s=10,
                        marker=markers_d[ds_name],
                        alpha=0.4 if ds_name == "WISDM" else 0.25,
                        label=f"{ACTIVITY_NAMES[cls]} ({ds_name})" if ds_name == "WISDM" else None)

    ax2.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax2.set_title("Per-Activity Distribution\n(circles=Aug-UCI, squares=WISDM)")

    # Compact legend (activity colors only)
    handles = [mpatches.Patch(color=act_colors[i], label=ACTIVITY_NAMES[i])
               for i in range(N_CLASSES)]
    handles += [plt.scatter([], [], marker="o", c="gray", s=30, alpha=0.5, label="Aug-UCI"),
                plt.scatter([], [], marker="s", c="gray", s=30, alpha=0.5, label="WISDM")]
    ax2.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Feature Space Distribution: Raw UCI → Aug UCI → WISDM\n"
                 "(PhysicalDistortion closes the domain gap)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Stability analysis
# ─────────────────────────────────────────────────────────────────────────────

def _perturb_noise(X: np.ndarray, scale: float, rng) -> np.ndarray:
    """Add white noise scaled by scale × original std."""
    sigma = X.std(axis=(0, 1), keepdims=True)
    return X + rng.normal(0, sigma * scale, X.shape).astype(np.float32)


def _perturb_rotation(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each window around the Z-axis by angle_deg degrees."""
    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta),  math.cos(theta), 0],
                  [0,                0,               1]], dtype=np.float32)
    return (X @ R.T)


def plot_stability(
    models: dict[str, nn.Module],
    splits: dict,
    device: torch.device,
    out_path: Path,
) -> None:
    print("[Fig2] Stability analysis …")

    Xt, yt = splits["wisdm_test"]
    rng    = np.random.default_rng(99)

    noise_scales  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    rotation_degs = [0, 10, 20, 30, 45, 60, 75, 90]

    # ── Collect results ───────────────────────────────────────────────────
    noise_f1s = {n: [] for n in MODEL_NAMES}
    rot_f1s   = {n: [] for n in MODEL_NAMES}

    for scale in noise_scales:
        X_noisy = _perturb_noise(Xt, scale, rng) if scale > 0 else Xt
        for name, model in models.items():
            preds = predict(model, X_noisy, device)
            f1    = float(f1_score(yt, preds, average="macro", zero_division=0))
            noise_f1s[name].append(f1)
        print(f"  noise ×{scale:.1f}  done")

    for angle in rotation_degs:
        X_rot = _perturb_rotation(Xt, angle) if angle > 0 else Xt
        for name, model in models.items():
            preds = predict(model, X_rot, device)
            f1    = float(f1_score(yt, preds, average="macro", zero_division=0))
            rot_f1s[name].append(f1)
        print(f"  rotation {angle}°  done")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for name in MODEL_NAMES:
        c  = MODEL_COLORS[name]
        mk = MODEL_MARKERS[name]
        ax1.plot(noise_scales, noise_f1s[name],
                 color=c, marker=mk, linewidth=2, markersize=6,
                 label=name.upper())
        ax2.plot(rotation_degs, rot_f1s[name],
                 color=c, marker=mk, linewidth=2, markersize=6,
                 label=name.upper())

    # Shade "target" range (WISDM natural noise / orientation variance)
    ax1.axvspan(0, 1.5, alpha=0.06, color="green",
                label="Natural WISDM noise range")
    ax2.axvspan(0, 30, alpha=0.06, color="green",
                label="Typical pocket orientation range")

    ax1.set_xlabel("Added noise multiplier (× signal std)")
    ax1.set_ylabel("WISDM-Test Macro F1")
    ax1.set_title("Robustness to Additive Noise\n(models trained on Aug-UCI)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    ax2.set_xlabel("Extra rotation angle (°) around Z-axis")
    ax2.set_ylabel("WISDM-Test Macro F1")
    ax2.set_title("Robustness to Orientation Shift\n(models trained on Aug-UCI)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)

    fig.suptitle("Covariate Shift Stability Analysis\n"
                 "(each point = WISDM-Test F1 after additional perturbation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    models: dict[str, nn.Module],
    splits: dict,
    device: torch.device,
    out_path: Path,
) -> None:
    print("[Fig3] Confusion matrices …")

    Xt, yt = splits["wisdm_test"]
    n_models = len(MODEL_NAMES)

    # 2 rows: top = raw UCI test, bottom = WISDM test
    fig, axes = plt.subplots(2, n_models, figsize=(4.2 * n_models, 8.5))
    # Row 0: raw UCI eval; Row 1: WISDM test eval
    eval_splits = [
        ("Raw-UCI",    splits["raw_uci"]),
        ("WISDM-Test", splits["wisdm_test"]),
    ]

    for row_i, (split_label, (Xe, ye)) in enumerate(eval_splits):
        for col_i, name in enumerate(MODEL_NAMES):
            ax  = axes[row_i][col_i]
            preds = predict(models[name], Xe, device)
            cm  = confusion_matrix(ye, preds, labels=list(range(N_CLASSES)))
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

            im  = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(N_CLASSES))
            ax.set_yticks(range(N_CLASSES))
            short = [a[:4] for a in ACTIVITY_NAMES]
            ax.set_xticklabels(short, fontsize=8, rotation=30, ha="right")
            ax.set_yticklabels(short, fontsize=8)

            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    v = cm_norm[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if v > 0.5 else "black")

            macro_f1 = float(f1_score(ye, preds, average="macro", zero_division=0))
            title_top = name.upper() if row_i == 0 else ""
            ax.set_title(f"{title_top}\n{split_label}  F1={macro_f1:.3f}",
                         fontsize=10, pad=4)
            if col_i == 0:
                ax.set_ylabel("True label", fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Predicted label", fontsize=9)

    fig.suptitle("Normalised Confusion Matrices — Cross-Dataset Transfer\n"
                 "(row=true class, col=predicted class)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Radar charts
# ─────────────────────────────────────────────────────────────────────────────

def _radar_axes(fig, rect, n_axes: int, labels: list[str]):
    """Create a radar (spider) subplot and return (ax, angles)."""
    angles = [n / float(n_axes) * 2 * math.pi for n in range(n_axes)]
    angles += angles[:1]   # close the polygon

    ax = fig.add_axes(rect, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    return ax, angles


def _draw_radar_polygon(ax, angles: list, values: list, color: str,
                        label: str, alpha_fill=0.15):
    vals = list(values) + [values[0]]
    ax.plot(angles, vals, color=color, linewidth=2, linestyle="solid", label=label)
    ax.fill(angles, vals, color=color, alpha=alpha_fill)


def plot_radar_charts(
    models: dict[str, nn.Module],
    splits: dict,
    device: torch.device,
    out_path: Path,
) -> None:
    print("[Fig4] Radar charts …")

    Xt, yt = splits["wisdm_test"]

    # Compute per-class F1 per model
    per_class: dict[str, list] = {}
    for name, model in models.items():
        preds = predict(model, Xt, device)
        per_class[name] = f1_score(yt, preds, average=None,
                                   zero_division=0, labels=list(range(N_CLASSES))).tolist()

    n_models = len(MODEL_NAMES)
    # Layout: n_models individual + 1 overlay
    n_cols = 3; n_rows = math.ceil((n_models + 1) / n_cols)
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows))
    fig.suptitle("Per-Activity F1 on WISDM-Test — Radar Charts\n"
                 "(models trained on Aug-UCI, C1 condition)",
                 fontsize=13, fontweight="bold", y=1.02)

    radar_size = 0.28   # subplot fraction

    for idx, name in enumerate(MODEL_NAMES):
        row = idx // n_cols
        col = idx % n_cols
        left   = col / n_cols + 0.03
        bottom = 1 - (row + 1) / n_rows + 0.04
        rect   = [left, bottom, radar_size, radar_size * 1.15]

        ax, angles = _radar_axes(fig, rect, N_CLASSES, ACTIVITY_NAMES)
        vals = per_class[name]
        _draw_radar_polygon(ax, angles, vals, MODEL_COLORS[name],
                            name.upper(), alpha_fill=0.2)

        # Mark per-class values
        for i, (a, v) in enumerate(zip(angles[:-1], vals)):
            ax.annotate(f"{v:.2f}", xy=(a, v),
                        fontsize=7, ha="center", va="center",
                        color=MODEL_COLORS[name],
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7))

        macro = np.mean(vals)
        ax.set_title(f"{name.upper()}\nMacro F1 = {macro:.3f}",
                     fontsize=10, pad=20, fontweight="bold",
                     color=MODEL_COLORS[name])

    # Overlay: all models on one radar
    row = n_models // n_cols
    col = n_models % n_cols
    left   = col / n_cols + 0.03
    bottom = 1 - (row + 1) / n_rows + 0.04
    rect   = [left, bottom, radar_size, radar_size * 1.15]

    ax_ov, angles = _radar_axes(fig, rect, N_CLASSES, ACTIVITY_NAMES)
    for name in MODEL_NAMES:
        _draw_radar_polygon(ax_ov, angles, per_class[name],
                            MODEL_COLORS[name], name.upper(), alpha_fill=0.08)
    ax_ov.set_title("All Models (Overlay)", fontsize=10, pad=20, fontweight="bold")
    ax_ov.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out-dir",     type=Path, default=Path(__file__).resolve().parent.parent / 'figures')
    parser.add_argument("--ckpt-dir",    type=Path, default=Path(__file__).resolve().parent.parent / 'results' / 'viz_checkpoints')
    parser.add_argument("--hp-config",   type=Path, default=Path(__file__).resolve().parent.parent / 'results' / 'hp_best_configs.json')
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-train",  action="store_true", help="Load checkpoints without retraining")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--n-sample",    type=int, default=1500, help="PCA sample size per dataset")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--fast",        action="store_true",
                        help="25 epochs, 500 PCA samples — smoke test")
    args = parser.parse_args()

    if args.fast:
        args.epochs   = 25
        args.patience = 8
        args.n_sample = 500

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Config] device={args.device}  out_dir={args.out_dir}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    splits = load_all_data(args.seed)

    # ── 2. Load HP configs ────────────────────────────────────────────────
    hp_configs = load_hp_configs(args.hp_config)

    # ── 3. Train / load models ────────────────────────────────────────────
    if args.skip_train:
        # Try to load all from ckpt_dir; re-train missing ones
        print("[Models] Loading from checkpoint dir (--skip-train) …")

    models = train_c1_models(
        splits, hp_configs, args.ckpt_dir, device,
        epochs=args.epochs, patience=args.patience, seed=args.seed,
    )

    # ── 4. Generate figures ───────────────────────────────────────────────
    print("\n[Figures] Generating …\n")

    plot_distribution_alignment(
        splits,
        out_path=args.out_dir / "distribution_alignment.png",
        n_sample=args.n_sample,
    )

    plot_stability(
        models, splits, device,
        out_path=args.out_dir / "stability_analysis.png",
    )

    plot_confusion_matrices(
        models, splits, device,
        out_path=args.out_dir / "confusion_matrices.png",
    )

    plot_radar_charts(
        models, splits, device,
        out_path=args.out_dir / "radar_charts.png",
    )

    print(f"\n✓ All figures saved to: {args.out_dir}")
    for f in sorted(args.out_dir.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<40} {size_kb:>5} KB")


if __name__ == "__main__":
    main()
