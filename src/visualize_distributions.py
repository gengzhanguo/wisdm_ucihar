"""
visualize_distributions.py  —  Step 3: Statistical Fitting & Visualization
============================================================================
Fits KDE to Acc_mag and Frequency Centroid distributions for WISDM and UCI HAR,
computes cross-dataset distance metrics, and saves RQ1_Physical_Distributions.png.

Distance metrics (pre-computed for annotation):
  - KL Divergence  (KL[P||Q] + KL[Q||P]) / 2  symmetric
  - Wasserstein-1  (Earth Mover's Distance)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_io import load_wisdm, load_ucihar
from physics_engine import extract_features, acc_magnitude, FEATURE_NAMES

# ─────────────────────────────────────────────
# Distance helpers
# ─────────────────────────────────────────────

def symmetric_kl(p_vals: np.ndarray, q_vals: np.ndarray,
                 n_bins: int = 200) -> float:
    """Symmetric KL divergence via histogram approximation."""
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    # Smooth to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    kl_pq = np.sum(p_hist * np.log(p_hist / q_hist))
    kl_qp = np.sum(q_hist * np.log(q_hist / p_hist))
    return float((kl_pq + kl_qp) / 2)


def emd(p_vals: np.ndarray, q_vals: np.ndarray) -> float:
    """Wasserstein-1 / Earth Mover's Distance."""
    return float(wasserstein_distance(p_vals, q_vals))


# ─────────────────────────────────────────────
# Load data & extract features
# ─────────────────────────────────────────────
print("Loading datasets...")
wisdm = load_wisdm(verbose=False)
uci   = load_ucihar(verbose=False)

feats_w = extract_features(wisdm["X"], wisdm["fs"])  # (N_w, 6)
feats_u = extract_features(uci["X"],   uci["fs"])    # (N_u, 6)

# Per-window Acc magnitude mean  (col 0) and freq centroid (col 5)
w_mag = feats_w[:, 0]   # mag_mean per window
u_mag = feats_u[:, 0]

w_fc  = feats_w[:, 5]   # freq_centroid per window
u_fc  = feats_u[:, 5]

# ─────────────────────────────────────────────
# Compute distance metrics
# ─────────────────────────────────────────────
print("Computing distance metrics...")

kl_mag  = symmetric_kl(w_mag, u_mag)
emd_mag = emd(w_mag, u_mag)

kl_fc   = symmetric_kl(w_fc, u_fc)
emd_fc  = emd(w_fc, u_fc)

print(f"  Acc Magnitude  — Sym-KL: {kl_mag:.4f}   EMD: {emd_mag:.4f}")
print(f"  Freq Centroid  — Sym-KL: {kl_fc:.4f}   EMD: {emd_fc:.4f}")

# ─────────────────────────────────────────────
# KDE fitting
# ─────────────────────────────────────────────
def make_kde(data: np.ndarray, n_pts: int = 500):
    kde = gaussian_kde(data, bw_method="scott")
    x = np.linspace(data.min(), data.max(), n_pts)
    return x, kde(x)

x_mag_w, kde_mag_w = make_kde(w_mag)
x_mag_u, kde_mag_u = make_kde(u_mag)

x_fc_w,  kde_fc_w  = make_kde(w_fc)
x_fc_u,  kde_fc_u  = make_kde(u_fc)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
WISDM_COLOR = "#E07B39"   # warm orange
UCI_COLOR   = "#3A7DC9"   # steel blue
ALPHA_FILL  = 0.15

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RQ1 – Physical Feature Distributions: WISDM vs UCI HAR",
             fontsize=14, fontweight="bold", y=1.01)

# ── Left: Acc Magnitude ──────────────────────────────────────────────────────
ax = axes[0]

ax.plot(x_mag_w, kde_mag_w, color=WISDM_COLOR, lw=2.2,
        label=f"WISDM  (n={len(w_mag):,})")
ax.fill_between(x_mag_w, kde_mag_w, alpha=ALPHA_FILL, color=WISDM_COLOR)

ax.plot(x_mag_u, kde_mag_u, color=UCI_COLOR, lw=2.2,
        label=f"UCI HAR (n={len(u_mag):,})")
ax.fill_between(x_mag_u, kde_mag_u, alpha=ALPHA_FILL, color=UCI_COLOR)

# Vertical means
ax.axvline(w_mag.mean(), color=WISDM_COLOR, lw=1.2, ls="--", alpha=0.8)
ax.axvline(u_mag.mean(), color=UCI_COLOR,   lw=1.2, ls="--", alpha=0.8)

# Distance annotation box
stats_text = (
    f"Sym-KL  = {kl_mag:.4f}\n"
    f"EMD     = {emd_mag:.4f} m/s²\n"
    f"μ_W = {w_mag.mean():.2f}  μ_U = {u_mag.mean():.2f}"
)
ax.text(0.97, 0.97, stats_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85))

ax.set_xlabel("Acc Magnitude Mean  [m/s²]", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Acceleration Magnitude Distribution", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: Frequency Centroid ─────────────────────────────────────────────────
ax = axes[1]

ax.plot(x_fc_w, kde_fc_w, color=WISDM_COLOR, lw=2.2,
        label=f"WISDM  (n={len(w_fc):,})")
ax.fill_between(x_fc_w, kde_fc_w, alpha=ALPHA_FILL, color=WISDM_COLOR)

ax.plot(x_fc_u, kde_fc_u, color=UCI_COLOR, lw=2.2,
        label=f"UCI HAR (n={len(u_fc):,})")
ax.fill_between(x_fc_u, kde_fc_u, alpha=ALPHA_FILL, color=UCI_COLOR)

ax.axvline(w_fc.mean(), color=WISDM_COLOR, lw=1.2, ls="--", alpha=0.8)
ax.axvline(u_fc.mean(), color=UCI_COLOR,   lw=1.2, ls="--", alpha=0.8)

stats_text = (
    f"Sym-KL  = {kl_fc:.4f}\n"
    f"EMD     = {emd_fc:.4f} Hz\n"
    f"μ_W = {w_fc.mean():.3f}  μ_U = {u_fc.mean():.3f}"
)
ax.text(0.97, 0.97, stats_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85))

ax.set_xlabel("Frequency Centroid  [Hz]", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Frequency Centroid Distribution", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "RQ1_Physical_Distributions.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
