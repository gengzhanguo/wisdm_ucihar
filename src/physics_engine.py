"""
physics_engine.py  —  Step 2: Physics Feature Extraction
==========================================================
Extracts two physics-grounded features from each accelerometer window:

  1. Acc Magnitude (合振幅)
       Acc_mag(t) = sqrt(x(t)^2 + y(t)^2 + z(t)^2)
       Scalar summary per window: mean, std, max, min, range

  2. Frequency Centroid (频率重心)
       f_c = Σ [f_i * P(f_i)] / Σ P(f_i)
       where P(f_i) is the one-sided power spectral density via FFT,
       computed on the Acc_mag signal of each window.

Input
-----
  X : np.ndarray  (N, win_len, 3)   float32, unit m/s²

Output
------
  feature_matrix : np.ndarray  (N, 6)  float32
      columns: [mag_mean, mag_std, mag_max, mag_min, mag_range, freq_centroid]
"""

import numpy as np


# ─────────────────────────────────────────────
# Core feature functions
# ─────────────────────────────────────────────

def acc_magnitude(X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Acc magnitude time series.

    Parameters
    ----------
    X : (N, win_len, 3)

    Returns
    -------
    mag : (N, win_len)   Acc_mag(t) = sqrt(x² + y² + z²)
    """
    return np.sqrt((X ** 2).sum(axis=-1))          # (N, win_len)


def freq_centroid(mag: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute frequency centroid of the Acc magnitude via FFT.

    Parameters
    ----------
    mag : (N, win_len)   Acc magnitude time series
    fs  : int            sampling rate in Hz

    Returns
    -------
    fc : (N,)   frequency centroid in Hz
    """
    win_len = mag.shape[1]

    # Real FFT → one-sided spectrum
    fft_coeffs = np.fft.rfft(mag, axis=1)           # (N, win_len//2 + 1)
    power      = (np.abs(fft_coeffs) ** 2)           # power spectrum

    # Frequency axis (Hz)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)    # (win_len//2 + 1,)

    # Weighted mean frequency: Σ(f_i * P_i) / Σ(P_i)
    total_power = power.sum(axis=1, keepdims=True)   # (N, 1)
    # Avoid division by zero (silent windows)
    total_power = np.where(total_power == 0, 1e-12, total_power)

    fc = (power * freqs[np.newaxis, :]).sum(axis=1) / total_power.squeeze()
    return fc.astype(np.float32)                     # (N,)


# ─────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────

def extract_features(X: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract physics feature matrix from raw windows.

    Parameters
    ----------
    X  : (N, win_len, 3)  raw accelerometer windows (m/s²)
    fs : int              sampling rate in Hz

    Returns
    -------
    features : (N, 6)  float32
        [mag_mean, mag_std, mag_max, mag_min, mag_range, freq_centroid]
    """
    mag = acc_magnitude(X)                      # (N, win_len)

    mag_mean  = mag.mean(axis=1)
    mag_std   = mag.std(axis=1)
    mag_max   = mag.max(axis=1)
    mag_min   = mag.min(axis=1)
    mag_range = mag_max - mag_min
    fc        = freq_centroid(mag, fs)

    features = np.stack(
        [mag_mean, mag_std, mag_max, mag_min, mag_range, fc],
        axis=1
    ).astype(np.float32)

    return features


FEATURE_NAMES = [
    "mag_mean",
    "mag_std",
    "mag_max",
    "mag_min",
    "mag_range",
    "freq_centroid",
]


# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_io import load_wisdm, load_ucihar

    wisdm = load_wisdm(verbose=False)
    uci   = load_ucihar(verbose=False)

    for name, ds in [("WISDM", wisdm), ("UCI HAR", uci)]:
        feats = extract_features(ds["X"], ds["fs"])
        print(f"\n{'='*52}")
        print(f"  {name}  →  features shape: {feats.shape}")
        print(f"  {'Feature':<18} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print(f"  {'-'*50}")
        for i, fname in enumerate(FEATURE_NAMES):
            col = feats[:, i]
            print(f"  {fname:<18} {col.mean():>9.3f} {col.std():>9.3f}"
                  f" {col.min():>9.3f} {col.max():>9.3f}")
        print(f"{'='*52}")
