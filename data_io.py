"""
data_io.py  —  Step 1: Data Grounding
========================================
Loads WISDM v1.1 and UCI HAR raw inertial signals, then physically aligns them:
  1. Resample UCI HAR: 50 Hz → 20 Hz  (scipy.signal.resample)
  2. Unit conversion: UCI total_acc (g) × 9.80665 → m/s²
     WISDM is already in m/s² (doc: value of 10 ≈ 1g ≈ 9.81 m/s²)
  3. Label mapping: unified 6-class scheme
  4. Segment WISDM continuous stream into fixed windows (2.56 s @ 20 Hz = 51 samples)

Unified Label Map
-----------------
  0  Walking          (WISDM: Walking,    UCI: WALKING)
  1  Jogging          (WISDM: Jogging,    UCI: — dropped)
  2  Upstairs         (WISDM: Upstairs,   UCI: WALKING_UPSTAIRS)
  3  Downstairs       (WISDM: Downstairs, UCI: WALKING_DOWNSTAIRS)
  4  Sitting          (WISDM: Sitting,    UCI: SITTING)
  5  Standing         (WISDM: Standing,   UCI: STANDING)
  Note: UCI LAYING has no WISDM equivalent → dropped from UCI samples.

Usage
-----
  from data_io import load_wisdm, load_ucihar, AlignedDataset
  wisdm = load_wisdm()
  uci   = load_ucihar()
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample

# ─────────────────────────────────────────────
# Paths (relative to this script's directory)
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

WISDM_RAW_PATH = os.path.join(
    _HERE,
    "WISDM_ar_latest", "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt"
)

UCI_BASE = os.path.join(
    _HERE,
    "human+activity+recognition+using+smartphones", "UCI HAR Dataset"
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
G_TO_MS2 = 9.80665          # standard gravity
WISDM_FS = 20               # Hz
UCI_FS   = 50               # Hz
TARGET_FS = 20              # Hz — common sampling rate after alignment

# UCI window: 128 samples @ 50 Hz = 2.56 s
# After resample to 20 Hz: 128 * 20 / 50 = 51.2 → 51 samples
UCI_WIN_ORIG   = 128
UCI_WIN_TARGET = int(round(UCI_WIN_ORIG * TARGET_FS / UCI_FS))   # 51

# WISDM windowing parameters (match UCI window duration: 2.56 s)
WISDM_WIN_LEN  = UCI_WIN_TARGET   # 51 samples @ 20 Hz ≈ 2.56 s
WISDM_WIN_STEP = int(WISDM_WIN_LEN * 0.5)   # 50% overlap

# ─────────────────────────────────────────────
# Label maps
# ─────────────────────────────────────────────
UNIFIED_LABELS = {
    0: "Walking",
    1: "Jogging",
    2: "Upstairs",
    3: "Downstairs",
    4: "Sitting",
    5: "Standing",
}

WISDM_LABEL_MAP = {
    "Walking":    0,
    "Jogging":    1,
    "Upstairs":   2,
    "Downstairs": 3,
    "Sitting":    4,
    "Standing":   5,
}

UCI_LABEL_MAP = {
    1: 0,   # WALKING        → Walking
    2: 2,   # WALKING_UPSTAIRS → Upstairs
    3: 3,   # WALKING_DOWNSTAIRS → Downstairs
    4: 4,   # SITTING        → Sitting
    5: 5,   # STANDING       → Standing
    6: -1,  # LAYING         → dropped (no WISDM equivalent)
    # UCI Jogging class (1) maps to unified Walking;
    # LAYING is -1 (will be filtered out)
}
# Note: UCI has no Jogging → unified class 1 will only appear in WISDM

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _segment_windows(data: np.ndarray, labels: np.ndarray, subjects: np.ndarray,
                     win_len: int, step: int):
    """Slide a window over continuous time-series and collect (window, label, subject).
    Label is determined by majority vote within the window.
    Windows that span multiple subjects are discarded.
    """
    X, y, s = [], [], []
    n = len(data)
    for start in range(0, n - win_len + 1, step):
        end = start + win_len
        subj_chunk = subjects[start:end]
        # discard windows crossing user boundaries
        if subj_chunk[0] != subj_chunk[-1]:
            continue
        # majority-vote label
        label_chunk = labels[start:end]
        vals, counts = np.unique(label_chunk, return_counts=True)
        majority = vals[np.argmax(counts)]
        X.append(data[start:end])          # shape (win_len, 3)
        y.append(majority)
        s.append(subj_chunk[0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(s, dtype=np.int32)


# ─────────────────────────────────────────────
# WISDM Loader
# ─────────────────────────────────────────────

def load_wisdm(verbose: bool = True) -> dict:
    """
    Load and segment WISDM v1.1 raw accelerometer data.

    Returns
    -------
    dict with keys:
        X        : np.ndarray  (N, WIN_LEN, 3)   float32, unit: m/s²
        y        : np.ndarray  (N,)              int32,   unified label
        subject  : np.ndarray  (N,)              int32,   user id
        fs       : int         20 (Hz)
        win_len  : int
        label_names : dict
    """
    if verbose:
        print(f"[WISDM] Loading from {WISDM_RAW_PATH}")

    # WISDM raw lines end with ';', some have extra whitespace / blank lines
    rows = []
    bad_lines = 0
    with open(WISDM_RAW_PATH, "r") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                bad_lines += 1
                continue
            try:
                user     = int(parts[0].strip())
                activity = parts[1].strip()
                # timestamp: skip
                x = float(parts[3].strip())
                y_ = float(parts[4].strip())
                z = float(parts[5].strip())
                if activity not in WISDM_LABEL_MAP:
                    bad_lines += 1
                    continue
                rows.append((user, WISDM_LABEL_MAP[activity], x, y_, z))
            except ValueError:
                bad_lines += 1
                continue

    if verbose:
        print(f"[WISDM] Parsed {len(rows):,} samples  |  skipped {bad_lines:,} malformed lines")

    arr      = np.array(rows, dtype=np.float64)
    subjects = arr[:, 0].astype(np.int32)
    labels   = arr[:, 1].astype(np.int32)
    accel    = arr[:, 2:5].astype(np.float32)   # (N, 3)  already m/s²

    # Segment into windows
    X, y, s = _segment_windows(accel, labels, subjects, WISDM_WIN_LEN, WISDM_WIN_STEP)

    if verbose:
        print(f"[WISDM] Windows: {X.shape}  |  Subjects: {np.unique(s).size}")
        _print_class_dist(y, "WISDM")

    return {
        "X": X, "y": y, "subject": s,
        "fs": WISDM_FS, "win_len": WISDM_WIN_LEN,
        "label_names": UNIFIED_LABELS,
    }


# ─────────────────────────────────────────────
# UCI HAR Loader
# ─────────────────────────────────────────────

def _load_uci_split(split: str) -> tuple:
    """Load one split ('train' or 'test') of UCI HAR inertial signals."""
    sig_dir = os.path.join(UCI_BASE, split, "Inertial Signals")

    def _read_signal(fname):
        path = os.path.join(sig_dir, fname)
        return np.loadtxt(path, dtype=np.float32)   # (N_windows, 128)

    total_acc_x = _read_signal(f"total_acc_x_{split}.txt")
    total_acc_y = _read_signal(f"total_acc_y_{split}.txt")
    total_acc_z = _read_signal(f"total_acc_z_{split}.txt")

    y_raw = np.loadtxt(os.path.join(UCI_BASE, split, f"y_{split}.txt"), dtype=np.int32)
    subjects = np.loadtxt(os.path.join(UCI_BASE, split, f"subject_{split}.txt"), dtype=np.int32)

    # Stack: (N, 128, 3)
    raw = np.stack([total_acc_x, total_acc_y, total_acc_z], axis=-1)
    return raw, y_raw, subjects


def load_ucihar(verbose: bool = True) -> dict:
    """
    Load UCI HAR inertial signals, resample 50→20 Hz, convert g→m/s², map labels.

    Returns
    -------
    dict with keys:
        X        : np.ndarray  (N, UCI_WIN_TARGET, 3)  float32, unit: m/s²
        y        : np.ndarray  (N,)                   int32,   unified label
        subject  : np.ndarray  (N,)                   int32,   user id
        fs       : int         20 (after resampling)
        win_len  : int         UCI_WIN_TARGET (51)
        label_names : dict
    """
    if verbose:
        print(f"[UCI HAR] Loading from {UCI_BASE}")

    # Load train + test splits
    raw_tr, y_tr, s_tr = _load_uci_split("train")
    raw_te, y_te, s_te = _load_uci_split("test")

    raw = np.concatenate([raw_tr, raw_te], axis=0)     # (N_total, 128, 3)
    y_raw = np.concatenate([y_tr, y_te], axis=0)
    subjects = np.concatenate([s_tr, s_te], axis=0)

    if verbose:
        print(f"[UCI HAR] Raw windows: {raw.shape}  (@ {UCI_FS} Hz, unit: g)")

    # ── Step 1: Resample 50 Hz → 20 Hz ──────────────────────────────────────
    # Each window: (128,) → resample to (UCI_WIN_TARGET,) along axis=1
    raw_resampled = resample(raw, UCI_WIN_TARGET, axis=1).astype(np.float32)
    if verbose:
        print(f"[UCI HAR] After resample ({UCI_FS}→{TARGET_FS} Hz): {raw_resampled.shape}")

    # ── Step 2: Unit conversion g → m/s² ─────────────────────────────────────
    raw_resampled *= G_TO_MS2
    if verbose:
        print(f"[UCI HAR] Unit converted: g × {G_TO_MS2} = m/s²")

    # ── Step 3: Label mapping ─────────────────────────────────────────────────
    unified_labels = np.array([UCI_LABEL_MAP[int(lbl)] for lbl in y_raw], dtype=np.int32)

    # Drop LAYING (unified label == -1)
    keep = unified_labels >= 0
    X        = raw_resampled[keep]
    y_mapped = unified_labels[keep]
    s_mapped = subjects[keep]

    dropped = int((~keep).sum())
    if verbose:
        print(f"[UCI HAR] Dropped {dropped:,} LAYING windows (no WISDM equivalent)")
        print(f"[UCI HAR] Final windows: {X.shape}  |  Subjects: {np.unique(s_mapped).size}")
        _print_class_dist(y_mapped, "UCI HAR")

    return {
        "X": X, "y": y_mapped, "subject": s_mapped,
        "fs": TARGET_FS, "win_len": UCI_WIN_TARGET,
        "label_names": UNIFIED_LABELS,
    }


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def _print_class_dist(y: np.ndarray, name: str):
    total = len(y)
    print(f"  {'Class':<15} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*32}")
    for uid, uname in UNIFIED_LABELS.items():
        cnt = int((y == uid).sum())
        pct = 100 * cnt / total if total > 0 else 0
        if cnt > 0:
            print(f"  {uname:<15} {cnt:>7,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<15} {total:>7,}  100.0%")


def summarize(dataset: dict, name: str):
    """Print a quick summary of a loaded dataset."""
    print(f"\n{'='*50}")
    print(f"  Dataset : {name}")
    print(f"  Shape   : X={dataset['X'].shape}, y={dataset['y'].shape}")
    print(f"  fs      : {dataset['fs']} Hz  |  win_len: {dataset['win_len']} samples")
    print(f"  Subjects: {np.unique(dataset['subject']).tolist()}")
    print(f"  X range : [{dataset['X'].min():.3f}, {dataset['X'].max():.3f}] m/s²")
    print(f"  Class distribution:")
    _print_class_dist(dataset["y"], name)
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# Quick sanity check (run as script)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    wisdm = load_wisdm(verbose=True)
    print()
    uci   = load_ucihar(verbose=True)

    summarize(wisdm, "WISDM v1.1")
    summarize(uci,   "UCI HAR")

    # Verify physical alignment
    print("── Physical alignment check ──")
    print(f"  WISDM  win shape : {wisdm['X'].shape[1:]}  @ {wisdm['fs']} Hz")
    print(f"  UCI    win shape : {uci['X'].shape[1:]}    @ {uci['fs']} Hz")
    assert wisdm['X'].shape[1] == uci['X'].shape[1], "Window length mismatch!"
    assert wisdm['fs'] == uci['fs'], "Sampling rate mismatch!"
    print("  ✓ Window lengths match")
    print("  ✓ Sampling rates match (both 20 Hz)")
    print("  ✓ Units: both m/s²")
    print("  ✓ Label space: unified 6-class scheme")
