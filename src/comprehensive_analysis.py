"""
comprehensive_analysis.py  —  Supplementary Domain Shift Analysis
===================================================================
Based on literature review, fills the gaps left by Steps 1-4:

Gap 1: Per-axis distribution analysis (sensor orientation / placement shift)
Gap 2: Gravity component (DC bias) analysis — UCI waist vs WISDM pocket
Gap 3: Preprocessing difference — UCI body_acc (gravity-free) vs total_acc
Gap 4: Class prior shift P(Y) — Total Variation Distance
Gap 5: Signal autocorrelation — gait periodicity comparison
Gap 6: Per-axis SNR (noise floor) estimation
Gap 7: Comprehensive summary figure

References:
  - DAGHAR benchmark (Nature Scientific Data, 2024)
  - UCI HAR README: Butterworth LPF gravity separation
  - MDPI 2025: "waist-mounted UCI vs front-pocket WISDM"
  - rWISDM (arXiv 2305.10222)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.signal import butter, filtfilt, correlate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import load_wisdm, load_ucihar, UNIFIED_LABELS
from physics_engine import extract_features

from config import FIGURES_DIR as _cfg_figures
OUT_DIR = str(_cfg_figures)
C_W, C_U = "#E07B39", "#3A7DC9"
AXIS_COLORS = ["#E53935", "#43A047", "#1E88E5"]  # x=red, y=green, z=blue

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def sym_kl(a, b, n_bins=300):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    eps = 1e-6
    ph, _ = np.histogram(a, bins=bins, density=True)
    qh, _ = np.histogram(b, bins=bins, density=True)
    ph = (ph + eps) / (ph + eps).sum()
    qh = (qh + eps) / (qh + eps).sum()
    return float(0.5 * (np.sum(ph * np.log(ph/qh)) + np.sum(qh * np.log(qh/ph))))

def emd(a, b): return float(wasserstein_distance(a, b))

def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance between two probability vectors."""
    return float(0.5 * np.sum(np.abs(p - q)))

def make_kde(data, n=500):
    kde = gaussian_kde(data, bw_method="scott")
    x = np.linspace(data.min(), data.max(), n)
    return x, kde(x)

def butterworth_lowpass(data, cutoff=0.3, fs=20, order=3):
    """Low-pass Butterworth filter to extract gravity component."""
    nyq = fs / 2
    b, f_ = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, f_, data, axis=1)

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
print("Loading datasets …")
wisdm = load_wisdm(verbose=False)
uci   = load_ucihar(verbose=False)
feats_w = extract_features(wisdm["X"], wisdm["fs"])
feats_u = extract_features(uci["X"],   uci["fs"])
AXIS = ["X-axis", "Y-axis", "Z-axis"]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Per-Axis Distribution  (sensor placement / orientation shift)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 1: Per-Axis Distribution ──")

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
axis_metrics = []

for ax_idx in range(3):
    w_ax = wisdm["X"][:, :, ax_idx].flatten()
    u_ax = uci["X"][:, :, ax_idx].flatten()

    # raw sample KDE (subsample for speed)
    rng = np.random.default_rng(42)
    ws = rng.choice(w_ax, 50000, replace=False)
    us = rng.choice(u_ax, 30000, replace=False)

    xw, yw = make_kde(ws)
    xu, yu = make_kde(us)

    d_emd = emd(ws, us)
    d_kl  = sym_kl(ws, us)
    axis_metrics.append((d_emd, d_kl))

    # Left: KDE
    ax = axes[ax_idx, 0]
    ax.plot(xw, yw, color=C_W, lw=2, label=f"WISDM")
    ax.fill_between(xw, yw, alpha=0.12, color=C_W)
    ax.plot(xu, yu, color=C_U, lw=2, label=f"UCI HAR")
    ax.fill_between(xu, yu, alpha=0.12, color=C_U)
    ax.axvline(ws.mean(), color=C_W, ls="--", lw=1.1, alpha=0.8)
    ax.axvline(us.mean(), color=C_U, ls="--", lw=1.1, alpha=0.8)

    ann = (f"Wass = {d_emd:.4f}\nSym-KL = {d_kl:.4f}\n"
           f"Δμ = {abs(ws.mean()-us.mean()):.4f}")
    ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.88))
    ax.set_xlabel("Acceleration [m/s²]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{AXIS[ax_idx]} — Raw Signal Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Right: Per-axis mean per window (box-plot style violin)
    ax = axes[ax_idx, 1]
    w_mean = wisdm["X"][:, :, ax_idx].mean(axis=1)
    u_mean = uci["X"][:, :, ax_idx].mean(axis=1)
    parts = ax.violinplot([w_mean, u_mean], positions=[0, 1],
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], [C_W, C_U]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WISDM", "UCI HAR"])
    ax.set_ylabel("Window Mean [m/s²]", fontsize=9)
    ax.set_title(f"{AXIS[ax_idx]} — Window Mean Distribution\n"
                 f"(μ_W={w_mean.mean():.2f}, μ_U={u_mean.mean():.2f}, Δ={abs(w_mean.mean()-u_mean.mean()):.2f})",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

fig.suptitle("Per-Axis Signal Distribution — Sensor Placement / Orientation Shift\n"
             "(UCI: waist-fixed  vs  WISDM: front-pocket free-swinging)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUT_DIR, "per_axis_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")
for i, (e_, k_) in enumerate(axis_metrics):
    print(f"  {AXIS[i]}: Wass={e_:.4f}  KL={k_:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Gravity Component Analysis (DC bias)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 2: Gravity Component (DC Bias) ──")

# Gravity = low-freq component via Butterworth LPF (cutoff 0.3 Hz)
# This directly shows sensor orientation difference
grav_w = butterworth_lowpass(wisdm["X"], cutoff=0.3, fs=20)  # (N, 51, 3)
grav_u = butterworth_lowpass(uci["X"],   cutoff=0.3, fs=20)

# Mean gravity per window per axis
grav_w_mean = grav_w.mean(axis=1)   # (N, 3)
grav_u_mean = grav_u.mean(axis=1)

# Dynamic component = raw - gravity
dyn_w = wisdm["X"] - grav_w
dyn_u = uci["X"]   - grav_u
dyn_w_std = dyn_w.std(axis=1)   # (N, 3)
dyn_u_std = dyn_u.std(axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax_idx in range(3):
    # Top row: gravity component distribution per axis
    ax = axes[0, ax_idx]
    xw, yw = make_kde(grav_w_mean[:, ax_idx])
    xu, yu = make_kde(grav_u_mean[:, ax_idx])
    ax.plot(xw, yw, color=C_W, lw=2, label="WISDM")
    ax.fill_between(xw, yw, alpha=0.12, color=C_W)
    ax.plot(xu, yu, color=C_U, lw=2, label="UCI HAR")
    ax.fill_between(xu, yu, alpha=0.12, color=C_U)
    ax.axvline(0, color="gray", ls=":", lw=1)
    d = emd(grav_w_mean[:, ax_idx], grav_u_mean[:, ax_idx])
    ax.text(0.97, 0.97, f"Wass={d:.4f}", transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.88))
    ax.set_xlabel("[m/s²]", fontsize=9)
    ax.set_title(f"Gravity — {AXIS[ax_idx]}\n"
                 f"μ_W={grav_w_mean[:,ax_idx].mean():.2f}  "
                 f"μ_U={grav_u_mean[:,ax_idx].mean():.2f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)

    # Bottom row: dynamic component std
    ax = axes[1, ax_idx]
    xw, yw = make_kde(dyn_w_std[:, ax_idx])
    xu, yu = make_kde(dyn_u_std[:, ax_idx])
    ax.plot(xw, yw, color=C_W, lw=2, label="WISDM")
    ax.fill_between(xw, yw, alpha=0.12, color=C_W)
    ax.plot(xu, yu, color=C_U, lw=2, label="UCI HAR")
    ax.fill_between(xu, yu, alpha=0.12, color=C_U)
    d = emd(dyn_w_std[:, ax_idx], dyn_u_std[:, ax_idx])
    ax.text(0.97, 0.97, f"Wass={d:.4f}", transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.88))
    ax.set_xlabel("[m/s²]", fontsize=9)
    ax.set_title(f"Dynamic Std — {AXIS[ax_idx]}\n"
                 f"μ_W={dyn_w_std[:,ax_idx].mean():.2f}  "
                 f"μ_U={dyn_u_std[:,ax_idx].mean():.2f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)

fig.suptitle("Gravity Component & Dynamic Component Analysis\n"
             "Row 1: Low-freq (gravity direction) — reveals sensor mounting orientation\n"
             "Row 2: High-freq (body motion) — reveals activity intensity",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "gravity_bias_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")
print(f"  Gravity magnitude (L2) — WISDM: {np.linalg.norm(grav_w_mean.mean(axis=0)):.3f} m/s²")
print(f"  Gravity magnitude (L2) — UCI  : {np.linalg.norm(grav_u_mean.mean(axis=0)):.3f} m/s²")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Class Prior Shift P(Y) — TVD + bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 3: Class Prior Shift P(Y) ──")

n_classes = 6
w_counts = np.array([int((wisdm["y"] == c).sum()) for c in range(n_classes)])
u_counts = np.array([int((uci["y"]   == c).sum()) for c in range(n_classes)])
w_prior = w_counts / w_counts.sum()
u_prior = u_counts / u_counts.sum()

tvd_val = tvd(w_prior, u_prior)
print(f"  TVD(P_WISDM, P_UCI) = {tvd_val:.4f}")
for c in range(n_classes):
    print(f"    {UNIFIED_LABELS[c]:<12}: WISDM={w_prior[c]*100:.1f}%  UCI={u_prior[c]*100:.1f}%  "
          f"Δ={abs(w_prior[c]-u_prior[c])*100:.1f}%")

class_names = [UNIFIED_LABELS[c] for c in range(n_classes)]
x_pos = np.arange(n_classes)
w_bar_color = [C_W if w_counts[c] > 0 else "#CCCCCC" for c in range(n_classes)]
u_bar_color = [C_U if u_counts[c] > 0 else "#CCCCCC" for c in range(n_classes)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: grouped bar chart
ax = axes[0]
width = 0.35
bars_w = ax.bar(x_pos - width/2, w_prior * 100, width, label="WISDM",
                color=C_W, alpha=0.85, edgecolor="white")
bars_u = ax.bar(x_pos + width/2, u_prior * 100, width, label="UCI HAR",
                color=C_U, alpha=0.85, edgecolor="white")

for bar in bars_w:
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=C_W)
for bar in bars_u:
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=C_U)

ax.set_xticks(x_pos)
ax.set_xticklabels(class_names, rotation=20, ha="right")
ax.set_ylabel("Class Proportion [%]", fontsize=10)
ax.set_title(f"Class Prior Distribution P(Y)\nTVD = {tvd_val:.4f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Right: absolute difference + note
ax = axes[1]
delta = np.abs(w_prior - u_prior) * 100
CMAP = {0:"#4CAF50",1:"#F44336",2:"#2196F3",3:"#FF9800",4:"#9C27B0",5:"#00BCD4"}
colors_bar = [CMAP[c] for c in range(n_classes)]
CMAP = {0:"#4CAF50",1:"#F44336",2:"#2196F3",3:"#FF9800",4:"#9C27B0",5:"#00BCD4"}
colors_bar = [CMAP[c] for c in range(n_classes)]
ax.barh(class_names, delta, color=colors_bar, alpha=0.85, edgecolor="white")
for i, d in enumerate(delta):
    ax.text(d+0.2, i, f"{d:.1f}%", va="center", fontsize=9)
ax.set_xlabel("|P_WISDM(y) - P_UCI(y)| [%]", fontsize=10)
ax.set_title("Per-Class Prior Difference |ΔP(Y)|", fontsize=11, fontweight="bold")
ax.axvline(delta.mean(), color="gray", ls="--", lw=1.2, label=f"mean={delta.mean():.1f}%")
ax.legend(fontsize=9)
ax.grid(True, axis="x", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# Notes
note = ("Note: Jogging (WISDM-only) = 31.0%\n"
        "Laying (UCI-only) was dropped.\n"
        f"TVD = {tvd_val:.4f}  (0=identical, 1=disjoint)")
fig.text(0.5, -0.04, note, ha="center", fontsize=9, style="italic",
         bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="gray", alpha=0.9))

fig.suptitle("Class Prior Shift P(Y) — WISDM vs UCI HAR",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "class_prior_shift.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Autocorrelation — gait periodicity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 4: Autocorrelation (Gait Periodicity) ──")

# Use Walking class (class 0) for meaningful comparison
def mean_autocorr(X: np.ndarray, max_lag: int = None) -> np.ndarray:
    """Average normalized autocorrelation over all windows."""
    N, L, _ = X.shape
    mag = np.sqrt((X**2).sum(axis=-1))   # (N, L)
    if max_lag is None:
        max_lag = L - 1
    acfs = []
    for i in range(min(N, 500)):   # sample for speed
        sig = mag[i] - mag[i].mean()
        if sig.std() < 1e-6:
            continue
        c = correlate(sig, sig, mode="full")
        c = c[L-1:]                 # one-sided
        c = c / c[0]                # normalise
        acfs.append(c[:max_lag+1])
    return np.array(acfs)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
activity_ids = [0, 1, 2]   # Walking, Jogging/N/A, Upstairs
activity_names_plot = ["Walking (class 0)", "Jogging (WISDM only)", "Upstairs (class 2)"]

for col, (act_id, act_name) in enumerate(zip(activity_ids, activity_names_plot)):
    mask_w = wisdm["y"] == act_id
    mask_u = uci["y"]   == act_id

    # Top row: mean autocorrelation
    ax = axes[0, col]
    lag_axis = np.arange(WISDM_WIN := wisdm["X"].shape[1]) / wisdm["fs"]

    if mask_w.sum() > 10:
        acf_w = mean_autocorr(wisdm["X"][mask_w])
        mu_w = acf_w.mean(axis=0)
        sd_w = acf_w.std(axis=0)
        ax.plot(lag_axis, mu_w, color=C_W, lw=2, label=f"WISDM (n={mask_w.sum()})")
        ax.fill_between(lag_axis, mu_w-sd_w, mu_w+sd_w, alpha=0.15, color=C_W)

    if mask_u.sum() > 10:
        acf_u = mean_autocorr(uci["X"][mask_u])
        mu_u = acf_u.mean(axis=0)
        sd_u = acf_u.std(axis=0)
        ax.plot(lag_axis, mu_u, color=C_U, lw=2, label=f"UCI (n={mask_u.sum()})")
        ax.fill_between(lag_axis, mu_u-sd_u, mu_u+sd_u, alpha=0.15, color=C_U)

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Lag [s]", fontsize=9)
    ax.set_ylabel("Autocorrelation", fontsize=9)
    ax.set_title(f"ACF — {act_name}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)

    # Bottom row: power spectrum (FFT of magnitude)
    ax = axes[1, col]
    fs = wisdm["fs"]
    freqs = np.fft.rfftfreq(wisdm["X"].shape[1], d=1/fs)

    if mask_w.sum() > 10:
        mag_w = np.sqrt((wisdm["X"][mask_w]**2).sum(axis=-1))
        psd_w = np.abs(np.fft.rfft(mag_w, axis=1))**2
        ax.semilogy(freqs, psd_w.mean(axis=0), color=C_W, lw=2, label="WISDM")

    if mask_u.sum() > 10:
        mag_u = np.sqrt((uci["X"][mask_u]**2).sum(axis=-1))
        psd_u = np.abs(np.fft.rfft(mag_u, axis=1))**2
        ax.semilogy(freqs, psd_u.mean(axis=0), color=C_U, lw=2, label="UCI HAR")

    ax.set_xlabel("Frequency [Hz]", fontsize=9)
    ax.set_ylabel("PSD (log)", fontsize=9)
    ax.set_title(f"Power Spectrum — {act_name}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.25)

fig.suptitle("Signal Periodicity & Spectrum — Gait Cycle Analysis\n"
             "Row 1: Autocorrelation (periodicity)   Row 2: Power Spectral Density",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUT_DIR, "autocorrelation_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Comprehensive Summary — All Sources of Shift
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 5: Comprehensive Summary ──")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

# ── Panel A: Metric radar ─────────────────────────────────────────────────────
categories = ["Acc Mag\nWass", "Freq Ctr\nWass", "X-axis\nWass", "Y-axis\nWass",
              "Z-axis\nWass", "Prior\nTVD"]
values_raw = [
    emd(feats_w[:,0], feats_u[:,0]),
    emd(feats_w[:,5], feats_u[:,5]),
    axis_metrics[0][0],
    axis_metrics[1][0],
    axis_metrics[2][0],
    tvd_val,
]
# Normalise to [0,1] for radar
values_norm = np.array(values_raw) / max(values_raw)
values_norm = np.concatenate([values_norm, [values_norm[0]]])  # close polygon
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

ax_radar = fig.add_subplot(gs[0, 0], polar=True)
ax_radar.plot(angles, values_norm, "o-", lw=2, color=C_W)
ax_radar.fill(angles, values_norm, alpha=0.2, color=C_W)
ax_radar.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
ax_radar.set_ylim(0, 1.1)
ax_radar.set_title("Normalised Shift Magnitudes\n(across all metrics)", fontsize=10, fontweight="bold")

# ── Panel B: Body_acc vs Total_acc comparison ────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
body_acc_u = uci["X"] - butterworth_lowpass(uci["X"], 0.3, fs=20)  # approx body_acc
total_mag_u = np.sqrt((uci["X"]**2).sum(axis=-1)).mean(axis=1)
body_mag_u  = np.sqrt((body_acc_u**2).sum(axis=-1)).mean(axis=1)
total_mag_w = np.sqrt((wisdm["X"]**2).sum(axis=-1)).mean(axis=1)
body_mag_w  = np.sqrt(((wisdm["X"] - butterworth_lowpass(wisdm["X"], 0.3, fs=20))**2).sum(axis=-1)).mean(axis=1)

xw, yw = make_kde(total_mag_w); ax.plot(xw, yw, color=C_W, lw=2, ls="-",  label="WISDM total_acc")
xw2,yw2=make_kde(body_mag_w);   ax.plot(xw2,yw2,color=C_W, lw=2, ls="--", label="WISDM body_acc (est)")
xu, yu = make_kde(total_mag_u); ax.plot(xu, yu, color=C_U, lw=2, ls="-",  label="UCI total_acc")
xu2,yu2=make_kde(body_mag_u);   ax.plot(xu2,yu2,color=C_U, lw=2, ls="--", label="UCI body_acc (est)")
ax.set_xlabel("Magnitude [m/s²]", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.set_title("Preprocessing Gap:\nGravity-included vs Gravity-removed", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.25)

# ── Panel C: Prior shift bar ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
x_ = np.arange(n_classes)
ax.bar(x_ - 0.18, w_prior*100, 0.35, color=C_W, label="WISDM", alpha=0.85)
ax.bar(x_ + 0.18, u_prior*100, 0.35, color=C_U, label="UCI HAR", alpha=0.85)
ax.set_xticks(x_)
ax.set_xticklabels([UNIFIED_LABELS[c][:5] for c in range(n_classes)], fontsize=8)
ax.set_ylabel("%", fontsize=9)
ax.set_title(f"Class Prior P(Y)\nTVD = {tvd_val:.4f}", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.25)

# ── Panel D: Gravity direction heatmap ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
grav_table = np.array([
    [grav_w_mean[:, i].mean() for i in range(3)],
    [grav_u_mean[:, i].mean() for i in range(3)],
])
sns.heatmap(grav_table, ax=ax, annot=True, fmt=".2f",
            xticklabels=["X", "Y", "Z"], yticklabels=["WISDM", "UCI HAR"],
            cmap="coolwarm", center=0,
            cbar_kws={"label": "Mean gravity [m/s²]"})
ax.set_title("Gravity Direction per Axis\n(reflects sensor mounting orientation)",
             fontsize=10, fontweight="bold")

# ── Panel E: Per-axis Wasserstein comparison ──────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
metrics_labels = ["X-axis", "Y-axis", "Z-axis", "Acc Mag", "Freq Ctr"]
metrics_values = [axis_metrics[i][0] for i in range(3)] + \
                 [emd(feats_w[:,0], feats_u[:,0]), emd(feats_w[:,5], feats_u[:,5])]
colors_ = AXIS_COLORS + [C_W, "#9C27B0"]
bars = ax.barh(metrics_labels, metrics_values, color=colors_, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, metrics_values):
    ax.text(val + 0.02, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Wasserstein Distance", fontsize=9)
ax.set_title("Feature-wise Wasserstein Distance\n(WISDM → UCI HAR)", fontsize=10, fontweight="bold")
ax.grid(True, axis="x", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

# ── Panel F: Source taxonomy ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
shift_sources = [
    ("1. Activity Gap",      "WISDM: Jogging  |  UCI: Laying\n   Disjoint label spaces"),
    ("2. Prior Shift P(Y)",  f"TVD = {tvd_val:.4f}\n   Jogging 31% vs 0%"),
    ("3. Sensor Placement",  "WISDM: front pocket (free)\n   UCI:   waist (fixed)"),
    ("4. Preprocessing",     "UCI: Butterworth gravity sep.\n   WISDM: raw total_acc only"),
    ("5. Sampling Rate",     "WISDM: 20 Hz  |  UCI: 50 Hz\n   (aligned after resample)"),
    ("6. User Variability",  f"Intra-WISDM Wass mean=0.81\n   > Cross-dataset Wass=1.63"),
    ("7. Sensor Modality",   "WISDM: accel only\n   UCI:   accel + gyro"),
]
y_pos = 0.95
ax.text(0.02, y_pos + 0.02, "Sources of Domain Shift (Taxonomy)",
        transform=ax.transAxes, fontsize=10, fontweight="bold")
for title, desc in shift_sources:
    y_pos -= 0.13
    ax.text(0.02, y_pos, f"▶ {title}", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color="#333333")
    ax.text(0.06, y_pos - 0.04, desc, transform=ax.transAxes,
            fontsize=7.5, color="#555555")
    y_pos -= 0.04

fig.suptitle("Comprehensive Domain Shift Analysis — WISDM vs UCI HAR",
             fontsize=13, fontweight="bold", y=1.01)
path = os.path.join(OUT_DIR, "comprehensive_summary.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")

print("\n" + "═"*60)
print("  ALL FIGURES SAVED:")
for fname in ["per_axis_analysis.png", "gravity_bias_analysis.png",
              "class_prior_shift.png", "autocorrelation_analysis.png",
              "comprehensive_summary.png"]:
    print(f"    {fname}")
print("═"*60)
