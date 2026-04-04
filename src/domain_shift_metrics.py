"""
domain_shift_metrics.py  —  Step 4: Triple Metric Calculation & Visualization
===============================================================================
Produces three figures:
  1. amplitude_dist_metrics.png      — KDE + per-class Wasserstein & KL
  2. freq_centroid_dist_metrics.png  — KDE + per-class Wasserstein & KL
  3. latent_space_overlap.png        — PCA 2D scatter + Multi-kernel MMD
  4. subject_wasserstein_heatmap.png — 10×10 inter-user Wasserstein (WISDM)

Metrics implemented:
  - Wasserstein-1  (scipy.stats.wasserstein_distance)
  - KL Divergence  symmetric, histogram approx, eps=1e-6
  - Multi-kernel MMD  RBF kernels, bandwidths=[0.1, 1, 10], on PCA-2D features
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import wasserstein_distance, gaussian_kde
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import load_wisdm, load_ucihar, UNIFIED_LABELS
from physics_engine import extract_features, acc_magnitude

from config import FIGURES_DIR as _cfg_figures
OUT_DIR = str(_cfg_figures)

# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────
C_WISDM = "#E07B39"
C_UCI   = "#3A7DC9"
CLASS_COLORS = {
    0: "#4CAF50", 1: "#F44336", 2: "#2196F3",
    3: "#FF9800", 4: "#9C27B0", 5: "#00BCD4",
}

# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def sym_kl(a: np.ndarray, b: np.ndarray, n_bins: int = 300) -> float:
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    eps = 1e-6
    ph, _ = np.histogram(a, bins=bins, density=True)
    qh, _ = np.histogram(b, bins=bins, density=True)
    ph = ph + eps;  ph /= ph.sum()
    qh = qh + eps;  qh /= qh.sum()
    return float(0.5 * (np.sum(ph * np.log(ph / qh)) +
                        np.sum(qh * np.log(qh / ph))))


def emd(a: np.ndarray, b: np.ndarray) -> float:
    return float(wasserstein_distance(a, b))


def mk_mmd_squared(X: np.ndarray, Y: np.ndarray,
                   bandwidths=(0.1, 1.0, 10.0)) -> float:
    """
    Multi-kernel MMD² with RBF kernels.
      MMD²(X,Y) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    Averaged over all bandwidths.
    """
    def rbf_kernel(A, B, bw):
        # squared Euclidean distances
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]   # (na, nb, d)
        sq   = (diff ** 2).sum(axis=-1)                     # (na, nb)
        return np.exp(-sq / (2 * bw ** 2))

    mmd2 = 0.0
    for bw in bandwidths:
        Kxx = rbf_kernel(X, X, bw).mean()
        Kyy = rbf_kernel(Y, Y, bw).mean()
        Kxy = rbf_kernel(X, Y, bw).mean()
        mmd2 += (Kxx - 2 * Kxy + Kyy)
    return float(mmd2 / len(bandwidths))


def make_kde_curve(data: np.ndarray, n_pts: int = 600):
    kde = gaussian_kde(data, bw_method="scott")
    x = np.linspace(data.min(), data.max(), n_pts)
    return x, kde(x)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading datasets …")
wisdm = load_wisdm(verbose=False)
uci   = load_ucihar(verbose=False)

feats_w = extract_features(wisdm["X"], wisdm["fs"])
feats_u = extract_features(uci["X"],   uci["fs"])

w_mag = feats_w[:, 0]   # mag_mean per window
u_mag = feats_u[:, 0]
w_fc  = feats_w[:, 5]   # freq_centroid
u_fc  = feats_u[:, 5]
w_lbl = wisdm["y"]
u_lbl = uci["y"]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 & 2: amplitude_dist_metrics.png / freq_centroid_dist_metrics.png
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metric_kde(w_feat, u_feat, w_lbl, u_lbl,
                   xlabel, title, fname):
    """KDE plot: overall + per-class breakdown, annotated with Wass & KL."""

    shared_classes = sorted(set(np.unique(w_lbl)) & set(np.unique(u_lbl)))

    # ── compute overall metrics ──
    ov_emd = emd(w_feat, u_feat)
    ov_kl  = sym_kl(w_feat, u_feat)

    # ── per-class metrics ──
    class_metrics = {}
    for c in shared_classes:
        wc = w_feat[w_lbl == c]
        uc = u_feat[u_lbl == c]
        if len(wc) > 10 and len(uc) > 10:
            class_metrics[c] = {
                "emd": emd(wc, uc),
                "kl":  sym_kl(wc, uc),
            }

    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.35)
    ax_kde  = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # ── left: overall KDE ──
    x_w, y_w = make_kde_curve(w_feat)
    x_u, y_u = make_kde_curve(u_feat)

    ax_kde.plot(x_w, y_w, color=C_WISDM, lw=2.4,
                label=f"WISDM  (n={len(w_feat):,})")
    ax_kde.fill_between(x_w, y_w, alpha=0.12, color=C_WISDM)
    ax_kde.plot(x_u, y_u, color=C_UCI, lw=2.4,
                label=f"UCI HAR (n={len(u_feat):,})")
    ax_kde.fill_between(x_u, y_u, alpha=0.12, color=C_UCI)

    ax_kde.axvline(w_feat.mean(), color=C_WISDM, ls="--", lw=1.2, alpha=0.85)
    ax_kde.axvline(u_feat.mean(), color=C_UCI,   ls="--", lw=1.2, alpha=0.85)

    # per-class KDE (thin lines)
    for c in shared_classes:
        wc = w_feat[w_lbl == c]
        uc = u_feat[u_lbl == c]
        if len(wc) > 30:
            xc, yc = make_kde_curve(wc)
            ax_kde.plot(xc, yc, color=CLASS_COLORS[c], lw=0.9,
                        ls=":", alpha=0.6, label=f"W·{UNIFIED_LABELS[c]}")
        if len(uc) > 30:
            xc, yc = make_kde_curve(uc)
            ax_kde.plot(xc, yc, color=CLASS_COLORS[c], lw=0.9,
                        ls="-.", alpha=0.6, label=f"U·{UNIFIED_LABELS[c]}")

    # annotation box
    ann = (f"Overall\n"
           f"  Wasserstein = {ov_emd:.4f}\n"
           f"  Sym-KL      = {ov_kl:.4f}\n"
           f"  Δμ          = {abs(w_feat.mean()-u_feat.mean()):.4f}")
    ax_kde.text(0.97, 0.97, ann,
                transform=ax_kde.transAxes, fontsize=8.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.45", fc="white",
                          ec="gray", alpha=0.9))

    ax_kde.set_xlabel(xlabel, fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.set_title(title, fontsize=12, fontweight="bold")
    ax_kde.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax_kde.grid(True, alpha=0.25)

    # ── right: per-class Wasserstein bar chart ──
    labels_ = [UNIFIED_LABELS[c] for c in class_metrics]
    emds_   = [class_metrics[c]["emd"] for c in class_metrics]
    colors_ = [CLASS_COLORS[c] for c in class_metrics]

    bars = ax_bar.barh(labels_, emds_, color=colors_, edgecolor="white",
                       height=0.55)
    for bar, val in zip(bars, emds_):
        ax_bar.text(val + max(emds_) * 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

    ax_bar.set_xlabel("Wasserstein Distance", fontsize=10)
    ax_bar.set_title("Per-Class Wasserstein\n(WISDM ↔ UCI HAR)", fontsize=11)
    ax_bar.grid(True, axis="x", alpha=0.25)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Physical Shift Quantification — {title.split()[0]}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
    print(f"    Overall  Wass={ov_emd:.4f}  KL={ov_kl:.4f}")
    for c, m in class_metrics.items():
        print(f"    {UNIFIED_LABELS[c]:<12}  Wass={m['emd']:.4f}  KL={m['kl']:.4f}")

print("\n── Figure 1: Acc Magnitude ──")
plot_metric_kde(w_mag, u_mag, w_lbl, u_lbl,
                "Acc Magnitude Mean  [m/s²]",
                "Acceleration Magnitude Distribution",
                "amplitude_dist_metrics.png")

print("\n── Figure 2: Freq Centroid ──")
plot_metric_kde(w_fc, u_fc, w_lbl, u_lbl,
                "Frequency Centroid  [Hz]",
                "Frequency Centroid Distribution",
                "freq_centroid_dist_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: latent_space_overlap.png  —  PCA 2D + Multi-kernel MMD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 3: Latent Space / PCA ──")

# Flatten windows: (N, 51, 3) → (N, 153)
X_w_flat = wisdm["X"].reshape(len(wisdm["X"]), -1).astype(np.float32)
X_u_flat = uci["X"].reshape(len(uci["X"]), -1).astype(np.float32)

# PCA on combined data
N_pca = min(10000, len(X_w_flat), len(X_u_flat))   # fit on balanced subset
rng = np.random.default_rng(42)
idx_w = rng.choice(len(X_w_flat), N_pca, replace=False)
idx_u = rng.choice(len(X_u_flat), N_pca, replace=False)

pca = PCA(n_components=2, random_state=42)
pca.fit(np.concatenate([X_w_flat[idx_w], X_u_flat[idx_u]], axis=0))

Z_w = pca.transform(X_w_flat)   # (N_w, 2)
Z_u = pca.transform(X_u_flat)   # (N_u, 2)

# MMD on a balanced subsample (kernel computation is O(n²))
N_mmd = 1500
idx_w_mmd = rng.choice(len(Z_w), N_mmd, replace=False)
idx_u_mmd = rng.choice(len(Z_u), N_mmd, replace=False)
mmd2 = mk_mmd_squared(Z_w[idx_w_mmd], Z_u[idx_u_mmd], bandwidths=(0.1, 1.0, 10.0))
print(f"  Multi-kernel MMD² = {mmd2:.6f}   (bandwidths=[0.1, 1, 10])")

# Scatter plot
N_plot = 3000
idx_w_p = rng.choice(len(Z_w), N_plot, replace=False)
idx_u_p = rng.choice(len(Z_u), N_plot, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ── left: domain-coloured scatter ──
ax = axes[0]
ax.scatter(Z_w[idx_w_p, 0], Z_w[idx_w_p, 1],
           c=C_WISDM, s=6, alpha=0.35, label=f"WISDM (n={len(Z_w):,})", rasterized=True)
ax.scatter(Z_u[idx_u_p, 0], Z_u[idx_u_p, 1],
           c=C_UCI,   s=6, alpha=0.35, label=f"UCI HAR (n={len(Z_u):,})", rasterized=True)

# MMD annotation
mmd_text = (f"Multi-kernel MMD²\n"
            f"bandwidths=[0.1, 1, 10]\n"
            f"MMD² = {mmd2:.6f}\n"
            f"MMD  = {mmd2**0.5:.6f}")
ax.text(0.97, 0.97, mmd_text,
        transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.9))

var_ratio = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)", fontsize=11)
ax.set_title("Domain-level PCA Scatter\n(WISDM vs UCI HAR)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, markerscale=3)
ax.grid(True, alpha=0.2)

# ── right: class-coloured scatter (shared classes only) ──
ax = axes[1]
for c in sorted(UNIFIED_LABELS):
    # WISDM
    mask_w = w_lbl == c
    if mask_w.sum() > 0:
        idx_sub = rng.choice(np.where(mask_w)[0],
                             min(600, mask_w.sum()), replace=False)
        ax.scatter(Z_w[idx_sub, 0], Z_w[idx_sub, 1],
                   c=CLASS_COLORS[c], s=7, alpha=0.4, marker="o",
                   label=f"W·{UNIFIED_LABELS[c]}", rasterized=True)
    # UCI
    mask_u = u_lbl == c
    if mask_u.sum() > 0:
        idx_sub = rng.choice(np.where(mask_u)[0],
                             min(300, mask_u.sum()), replace=False)
        ax.scatter(Z_u[idx_sub, 0], Z_u[idx_sub, 1],
                   c=CLASS_COLORS[c], s=7, alpha=0.4, marker="^",
                   label=f"U·{UNIFIED_LABELS[c]}", rasterized=True)

ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)", fontsize=11)
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)", fontsize=11)
ax.set_title("Class-level PCA Scatter\n(○ WISDM  △ UCI HAR)", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, ncol=2, markerscale=2.5)
ax.grid(True, alpha=0.2)

fig.suptitle("Latent Feature Space Overlap — PCA Projection",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUT_DIR, "latent_space_overlap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: subject_wasserstein_heatmap.png  —  inter-user 10×10 matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 4: Subject Wasserstein Heatmap (WISDM) ──")

# Pick top-10 WISDM users by sample count
subj_ids, counts = np.unique(wisdm["subject"], return_counts=True)
top10 = subj_ids[np.argsort(-counts)[:10]]
print(f"  Top-10 users (by window count): {top10.tolist()}")

# Per-user mag_mean distribution
user_mags = {}
for uid in top10:
    mask = wisdm["subject"] == uid
    user_mags[uid] = feats_w[mask, 0]   # mag_mean

# Build 10×10 symmetric Wasserstein matrix
n = len(top10)
W_mat = np.zeros((n, n), dtype=np.float32)
for i, ui in enumerate(top10):
    for j, uj in enumerate(top10):
        if i == j:
            W_mat[i, j] = 0.0
        elif j > i:
            d = emd(user_mags[ui], user_mags[uj])
            W_mat[i, j] = d
            W_mat[j, i] = d

labels_10 = [f"User {uid}\n(n={len(user_mags[uid])})" for uid in top10]

fig, ax = plt.subplots(figsize=(10, 8))
mask_diag = np.eye(n, dtype=bool)

hm = sns.heatmap(
    W_mat, ax=ax,
    xticklabels=labels_10, yticklabels=labels_10,
    annot=True, fmt=".3f", annot_kws={"size": 8},
    cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    mask=mask_diag,
    cbar_kws={"label": "Wasserstein Distance (m/s²)"},
)

# Diagonal: grey "self" cells
for i in range(n):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                color="#DDDDDD", lw=0))
    ax.text(i + 0.5, i + 0.5, "—", ha="center", va="center",
            fontsize=9, color="#888888")

# Stats annotation
off_diag = W_mat[~mask_diag]
stats_text = (f"Mean: {off_diag.mean():.3f}\n"
              f"Std:  {off_diag.std():.3f}\n"
              f"Max:  {off_diag.max():.3f}\n"
              f"Min:  {off_diag.min():.3f}")
ax.text(1.22, 0.98, stats_text,
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.9))

ax.set_title("Intra-Dataset Subject Variability\n"
             "Pairwise Acc Magnitude Wasserstein Distance (WISDM Top-10 Users)",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
path = os.path.join(OUT_DIR, "subject_wasserstein_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("  METRIC SUMMARY")
print("═"*55)
print(f"  {'Metric':<35} {'Value':>10}")
print(f"  {'-'*50}")
print(f"  {'Acc Mag Wasserstein (overall)':<35} {emd(w_mag,u_mag):>10.4f}")
print(f"  {'Acc Mag Sym-KL (overall)':<35} {sym_kl(w_mag,u_mag):>10.4f}")
print(f"  {'Freq Centroid Wasserstein (overall)':<35} {emd(w_fc,u_fc):>10.4f}")
print(f"  {'Freq Centroid Sym-KL (overall)':<35} {sym_kl(w_fc,u_fc):>10.4f}")
print(f"  {'Multi-kernel MMD² (PCA-2D)':<35} {mmd2:>10.6f}")
print(f"  {'Multi-kernel MMD  (PCA-2D)':<35} {mmd2**0.5:>10.6f}")
print(f"  {'Intra-user Wass mean (WISDM)':<35} {off_diag.mean():>10.4f}")
print(f"  {'Intra-user Wass std  (WISDM)':<35} {off_diag.std():>10.4f}")
print("═"*55)
