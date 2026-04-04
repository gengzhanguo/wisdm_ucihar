"""
noise_analysis.py  —  Noise Characterisation: WISDM vs UCI HAR
===============================================================
Three complementary noise metrics:

  1. Signal Jerk (roughness)
       jerk(t) = Δ²Acc_mag(t)  (second finite difference of magnitude)
       A noisy/unfiltered signal has larger jerk variance.

  2. High-Frequency Energy Ratio
       HF_ratio = Σ P(f)  for f ∈ [5, 10] Hz  /  Σ P(f)  for f ∈ [0, 10] Hz
       UCI HAR was noise-filtered; expect lower HF ratio.

  3. Residual Noise (signal minus low-pass smoothed)
       noise(t) = Acc_mag(t) - LPF(Acc_mag(t))   cutoff = 4 Hz
       Captures unstructured high-frequency fluctuation.

Produces:
  figures/noise_analysis.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import gaussian_kde, wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import load_wisdm, load_ucihar, UNIFIED_LABELS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
C_W, C_U = "#E07B39", "#3A7DC9"
FS = 20   # Hz (both datasets after alignment)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def acc_magnitude(X):
    return np.sqrt((X ** 2).sum(axis=-1))   # (N, win_len)

def butter_lowpass(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, data, axis=1)

def make_kde(data, n=500):
    lo, hi = np.percentile(data, 0.5), np.percentile(data, 99.5)
    x = np.linspace(lo, hi, n)
    kde = gaussian_kde(data, bw_method="scott")
    return x, kde(x)

def emd(a, b): return float(wasserstein_distance(a, b))

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading datasets …")
wisdm = load_wisdm(verbose=False)
uci   = load_ucihar(verbose=False)

mag_w = acc_magnitude(wisdm["X"])   # (N_w, 51)
mag_u = acc_magnitude(uci["X"])     # (N_u, 51)

# ─────────────────────────────────────────────
# Metric 1: Signal Jerk (second difference)
# ─────────────────────────────────────────────
print("Computing jerk …")
jerk_w = np.diff(mag_w, n=2, axis=1)   # (N, 49)
jerk_u = np.diff(mag_u, n=2, axis=1)

jerk_std_w = jerk_w.std(axis=1)   # per-window jerk std
jerk_std_u = jerk_u.std(axis=1)

print(f"  Jerk std — WISDM: {jerk_std_w.mean():.4f}  UCI: {jerk_std_u.mean():.4f}")

# ─────────────────────────────────────────────
# Metric 2: High-Frequency Energy Ratio (5-10 Hz)
# ─────────────────────────────────────────────
print("Computing HF energy ratio …")
freqs = np.fft.rfftfreq(mag_w.shape[1], d=1.0 / FS)   # frequency axis

fft_w = np.abs(np.fft.rfft(mag_w, axis=1)) ** 2
fft_u = np.abs(np.fft.rfft(mag_u, axis=1)) ** 2

hf_mask = (freqs >= 5.0) & (freqs <= 10.0)
total_mask = freqs > 0   # exclude DC

hf_ratio_w = fft_w[:, hf_mask].sum(axis=1) / (fft_w[:, total_mask].sum(axis=1) + 1e-12)
hf_ratio_u = fft_u[:, hf_mask].sum(axis=1) / (fft_u[:, total_mask].sum(axis=1) + 1e-12)

print(f"  HF ratio  — WISDM: {hf_ratio_w.mean():.4f}  UCI: {hf_ratio_u.mean():.4f}")

# ─────────────────────────────────────────────
# Metric 3: Residual Noise (raw - LPF at 4 Hz)
# ─────────────────────────────────────────────
print("Computing residual noise …")
smooth_w = butter_lowpass(mag_w, cutoff=4.0, fs=FS)
smooth_u = butter_lowpass(mag_u, cutoff=4.0, fs=FS)

residual_w = mag_w - smooth_w
residual_u = mag_u - smooth_u

res_std_w = residual_w.std(axis=1)
res_std_u = residual_u.std(axis=1)

print(f"  Residual std — WISDM: {res_std_w.mean():.4f}  UCI: {res_std_u.mean():.4f}")

# Per-activity noise breakdown
print("\nPer-activity residual noise std:")
print(f"  {'Activity':<12} {'WISDM':>8} {'UCI':>8} {'Ratio':>7}")
print(f"  {'-'*38}")
for c in sorted(UNIFIED_LABELS):
    mw = wisdm["y"] == c
    mu = uci["y"]   == c
    if mw.sum() > 0 and mu.sum() > 0:
        rw = res_std_w[mw].mean()
        ru = res_std_u[mu].mean()
        print(f"  {UNIFIED_LABELS[c]:<12} {rw:>8.4f} {ru:>8.4f} {rw/ru:>7.2f}x")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Signal Noise Characterisation — WISDM vs UCI HAR\n"
    "UCI HAR: noise-filtered (median + Butterworth LPF)    "
    "WISDM: raw, unfiltered",
    fontsize=13, fontweight="bold"
)

# ── Row 1: Overall distributions ─────────────────────────────────────────────
titles_r1 = ["Jerk Std per Window\n(Second Difference of Acc Mag)",
             "High-Freq Energy Ratio [5–10 Hz]\n(Proxy for Noise Floor)",
             "Residual Noise Std per Window\n(Raw minus 4 Hz LPF)"]
data_pairs = [(jerk_std_w, jerk_std_u),
              (hf_ratio_w, hf_ratio_u),
              (res_std_w,  res_std_u)]
xlabels = ["Jerk Std [m/s³]", "HF Energy Ratio", "Residual Std [m/s²]"]

for col, (dw, du), title, xlabel in zip(range(3), data_pairs, titles_r1, xlabels):
    ax = axes[0, col]
    xw, yw = make_kde(dw)
    xu, yu = make_kde(du)
    ax.plot(xw, yw, color=C_W, lw=2.2, label=f"WISDM  μ={dw.mean():.3f}")
    ax.fill_between(xw, yw, alpha=0.12, color=C_W)
    ax.plot(xu, yu, color=C_U, lw=2.2, label=f"UCI HAR μ={du.mean():.3f}")
    ax.fill_between(xu, yu, alpha=0.12, color=C_U)
    ax.axvline(dw.mean(), color=C_W, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(du.mean(), color=C_U, ls="--", lw=1.2, alpha=0.8)

    d_emd = emd(dw, du)
    ax.text(0.97, 0.97,
            f"Wass = {d_emd:.4f}\nRatio = {dw.mean()/du.mean():.2f}×",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

# ── Row 2: Per-activity noise breakdown ──────────────────────────────────────
shared_classes = sorted(set(np.unique(wisdm["y"])) & set(np.unique(uci["y"])))

# Col 0: Per-activity jerk std grouped bars
ax = axes[1, 0]
x = np.arange(len(shared_classes))
w = 0.35
jw = [jerk_std_w[wisdm["y"] == c].mean() for c in shared_classes]
ju = [jerk_std_u[uci["y"]   == c].mean() for c in shared_classes]
ax.bar(x - w/2, jw, w, color=C_W, alpha=0.85, label="WISDM")
ax.bar(x + w/2, ju, w, color=C_U, alpha=0.85, label="UCI HAR")
ax.set_xticks(x)
ax.set_xticklabels([UNIFIED_LABELS[c] for c in shared_classes], rotation=20, ha="right")
ax.set_ylabel("Mean Jerk Std [m/s³]", fontsize=10)
ax.set_title("Jerk Std per Activity Class", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.25)

# Col 1: Per-activity residual noise grouped bars
ax = axes[1, 1]
rw = [res_std_w[wisdm["y"] == c].mean() for c in shared_classes]
ru = [res_std_u[uci["y"]   == c].mean() for c in shared_classes]
ax.bar(x - w/2, rw, w, color=C_W, alpha=0.85, label="WISDM")
ax.bar(x + w/2, ru, w, color=C_U, alpha=0.85, label="UCI HAR")
ax.set_xticks(x)
ax.set_xticklabels([UNIFIED_LABELS[c] for c in shared_classes], rotation=20, ha="right")
ax.set_ylabel("Mean Residual Std [m/s²]", fontsize=10)
ax.set_title("Residual Noise per Activity Class", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.25)

# Col 2: Mean PSD comparison (averaged across all windows)
ax = axes[1, 2]
freq_axis = np.fft.rfftfreq(mag_w.shape[1], d=1.0 / FS)
psd_w_mean = fft_w.mean(axis=0)
psd_u_mean = fft_u.mean(axis=0)

ax.semilogy(freq_axis, psd_w_mean, color=C_W, lw=2.2, label="WISDM")
ax.semilogy(freq_axis, psd_u_mean, color=C_U, lw=2.2, label="UCI HAR")
ax.axvspan(5, 10, alpha=0.1, color="red", label="HF noise band [5–10 Hz]")
ax.axvline(4, color="gray", ls=":", lw=1.2, label="LPF cutoff (4 Hz)")
ax.set_xlabel("Frequency [Hz]", fontsize=10)
ax.set_ylabel("Power (log scale)", fontsize=10)
ax.set_title("Mean Power Spectral Density\n(all windows, Acc Magnitude)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "noise_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_path}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "═"*55)
print("  NOISE METRIC SUMMARY")
print("═"*55)
print(f"  {'Metric':<32} {'WISDM':>8} {'UCI HAR':>8} {'Ratio':>7}")
print(f"  {'-'*55}")
print(f"  {'Jerk Std (mean)':<32} {jerk_std_w.mean():>8.4f} {jerk_std_u.mean():>8.4f} {jerk_std_w.mean()/jerk_std_u.mean():>6.2f}x")
print(f"  {'HF Energy Ratio 5-10Hz (mean)':<32} {hf_ratio_w.mean():>8.4f} {hf_ratio_u.mean():>8.4f} {hf_ratio_w.mean()/hf_ratio_u.mean():>6.2f}x")
print(f"  {'Residual Noise Std (mean)':<32} {res_std_w.mean():>8.4f} {res_std_u.mean():>8.4f} {res_std_w.mean()/res_std_u.mean():>6.2f}x")
print(f"  {'Jerk Wasserstein':<32} {emd(jerk_std_w,jerk_std_u):>8.4f}")
print(f"  {'Residual Wass':<32} {emd(res_std_w, res_std_u):>8.4f}")
print("═"*55)
print("\nConclusion: UCI HAR noise floor is suppressed by pre-processing.")
print("WISDM raw signal carries significantly more high-frequency content.")
