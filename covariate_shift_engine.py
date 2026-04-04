"""
covariate_shift_engine.py  —  Phase 1 (v2): Enhanced Covariate Shift Engine
=============================================================================
Five-operator pipeline that transforms UCI HAR windows into statistically
WISDM-like signals, covering all major shift sources quantified in
RQ1_Technical_Report.md.

v2 Changes vs v1
----------------
  Op 2  apply_gravity_attenuation  (NEW)
        Sec 4.3: reduces gravity L2 norm from UCI's 9.69 → WISDM's 7.29 m/s²
        (free-swinging pocket absorbs partial gravity projection).

  Op 4  apply_spectral_boost       (NEW)
        Sec 4.9: boosts 0.8–2.0 Hz gait band +20–50% for locomotion classes,
        simulating WISDM's larger sensor excursion per step in a front pocket.

  Op 5  apply_noise_shift          (REWRITTEN)
        v1 used white Gaussian noise, which produces Jerk Std ≈ 73 m/s³
        (mathematical consequence of FS × σ × √2), far above WISDM's 8.55.
        v2 uses an AR(1) colored noise model with α ≈ 0.987, derived from the
        WISDM Jerk/noise ratio.  This matches both residual noise std AND
        Jerk std simultaneously.  Per-activity noise targets are also applied
        (Walking 2.52 m/s² vs Sitting 0.054 m/s²; 47× range).

Domain Shift Sources Addressed
-------------------------------
Source 1 (Sec 4.2)  — Sensor Mounting Orientation
    Per-axis Wasserstein: X = 9.06, Y = 8.72 m/s²
    UCI  gravity → X-axis  (mean ≈ +8.58 m/s²)
    WISDM gravity → Y-axis (mean ≈ +6.51 m/s²)
    → apply_orientation_shift: 90° Z-rotation + 5–10° random pocket wobble

Source 2a (Sec 4.3) — Gravity Magnitude Attenuation  [NEW]
    UCI gravity L2  = 9.69 m/s²  (rigid waist mount, full gravity projection)
    WISDM gravity L2 = 7.29 m/s²  (free pocket, partial gravity projection)
    Scale factor: 7.29 / 9.69 ≈ 0.753
    → apply_gravity_attenuation: scale gravity DC component by 0.753×

Source 2b (Sec 4.3) — Dynamic Amplitude
    WISDM dynamic std is 3–5× larger than UCI across all axes.
    → apply_amplitude_scaling: scale body_acc 3–5×, preserve gravity

Source 5 (Sec 4.9)  — Gait Periodicity / Spectral Structure  [NEW]
    WISDM front-pocket sensor has larger mechanical excursion per step,
    producing stronger PSD peaks at step frequencies (1–2 Hz).
    UCI HAR PSD is flatter for locomotion activities.
    → apply_spectral_boost: +20–50% gain on 0.8–2.0 Hz band, loco only

Source 8 (Sec 4.10) — Signal Noise Level  [REWRITTEN v2]
    WISDM Residual Noise Std: 2.603 m/s²,  UCI: 0.582 m/s²
    WISDM Jerk Std:           8.549 m/s³,  UCI: 1.838 m/s³
    WISDM noise is time-correlated sensor drift, NOT white noise.
    AR(1) model: x[t] = α·x[t-1] + ε[t],  α = 0.987
    Per-activity noise targets match Table from Sec 4.10.
    → apply_noise_shift: AR(1) colored noise, per-activity sigma

CrossDatasetLoader
------------------
    Source : UCI HAR  → PhysicalDistortion.apply_all() → "pseudo-WISDM"
    Target : WISDM
        Validation : subjects  1–30
        Test       : subjects 31–36
    Shared labels : 5 classes (Jogging dropped, Laying already excluded)
    Label remap   : {Walking:0, Upstairs:1, Downstairs:2, Sitting:3, Standing:4}
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from data_io import load_wisdm, load_ucihar

# ─────────────────────────────────────────────────────────────────────────────
# Report-grounded constants
# ─────────────────────────────────────────────────────────────────────────────

FS = 20.0   # Hz — common sampling rate after UCI resampling

# ── Sec 4.2: per-axis gravity means ─────────────────────────────────────────
UCI_GRAVITY_MEAN   = np.array([ 8.58, -2.20, -1.72], dtype=np.float32)  # m/s²
WISDM_GRAVITY_MEAN = np.array([-0.32,  6.51, -1.12], dtype=np.float32)  # m/s²

# ── Sec 4.3: gravity L2 norms (rigid mount vs free pocket) ──────────────────
UCI_GRAVITY_L2    = 9.69    # m/s²
WISDM_GRAVITY_L2  = 7.29    # m/s²
GRAVITY_ATT_SCALE = WISDM_GRAVITY_L2 / UCI_GRAVITY_L2   # ≈ 0.7523

# ── Sec 4.3: dynamic-component std targets ──────────────────────────────────
# NOTE: report's approximate values ([0.8,1.5,1.5] for UCI; [3.9,5.2,4.0] for
# WISDM) are global means across all activities.  After the 90° Z-rotation
# (old X→new Y) the UCI body_acc std on Y is actually ~2.1 m/s² for locomotion
# classes, not 0.8.  All scale factors below are EMPIRICALLY measured from the
# actual loaded datasets.
WISDM_DYN_STD = np.array([3.9, 5.2, 4.0], dtype=np.float32)  # report reference

# Empirical per-activity per-axis scale factors
# = WISDM_body_acc_std / UCI_body_acc_std_post_rotation
# Measured directly from loaded data (see diagnostic script in sanity check).
# Keys: remapped label IDs (0=Walking,1=Upstairs,2=Downstairs,3=Sitting,4=Standing)
AMP_SCALE_PER_ACTIVITY: dict[int, np.ndarray] = {
    0: np.array([2.58, 2.06, 2.94], dtype=np.float32),  # Walking
    1: np.array([2.13, 1.72, 2.08], dtype=np.float32),  # Upstairs
    2: np.array([1.97, 1.22, 2.24], dtype=np.float32),  # Downstairs
    3: np.array([2.83, 4.82, 2.65], dtype=np.float32),  # Sitting
    4: np.array([3.10, 4.79, 3.19], dtype=np.float32),  # Standing
}
# Global fallback (label unknown): WISDM global / UCI-post-rot global
AMP_SCALE_GLOBAL = np.array([4.24, 2.81, 4.06], dtype=np.float32)

# ── Sec 4.10: global noise statistics ───────────────────────────────────────
WISDM_RESIDUAL_NOISE_STD = 2.603   # m/s²
UCI_RESIDUAL_NOISE_STD   = 0.582   # m/s²
WISDM_JERK_STD           = 8.549   # m/s³
UCI_JERK_STD             = 1.838   # m/s³

# AR(1) coefficient — derived from WISDM Jerk-to-noise ratio
# Model: x[t] = α·x[t-1] + ε[t],  Var(x) = σ²,  Jerk Std = σ·FS·√(2(1−α))
# → α = 1 − (jerk_std / (σ·FS))² / 2
_AR_RATIO = WISDM_JERK_STD / (WISDM_RESIDUAL_NOISE_STD * FS)
AR1_ALPHA = float(1.0 - (_AR_RATIO ** 2) / 2.0)   # ≈ 0.9865
# Expected Jerk Std at steady state: σ·FS·√(2(1−α)) ≈ 8.55 m/s³  ✓

# ── Sec 4.10: per-activity noise statistics ─────────────────────────────────
# Keys: remapped label IDs (0–4 in 5-class scheme)
# Walking:0  Upstairs:1  Downstairs:2  Sitting:3  Standing:4
ACTIVITY_WISDM_NOISE_STD: dict[int, float] = {
    0: 2.52,   # Walking
    1: 2.06,   # Upstairs
    2: 2.29,   # Downstairs
    3: 0.054,  # Sitting
    4: 0.102,  # Standing
}
ACTIVITY_UCI_NOISE_STD: dict[int, float] = {
    0: 1.02,   # Walking
    1: 0.86,   # Upstairs
    2: 1.19,   # Downstairs
    3: 0.027,  # Sitting
    4: 0.035,  # Standing
}

# Locomotion classes eligible for spectral boost
LOCOMOTION_LABELS: frozenset[int] = frozenset({0, 1, 2})

# ── Subject split (WISDM subjects 1–36) ─────────────────────────────────────
WISDM_VAL_SUBJECTS  = list(range(1, 31))   # 1–30
WISDM_TEST_SUBJECTS = list(range(31, 37))  # 31–36

# ── 5-class shared label scheme ─────────────────────────────────────────────
SHARED_UNIFIED_IDS = [0, 2, 3, 4, 5]
SHARED_LABEL_REMAP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4}
SHARED_LABEL_NAMES = {
    0: "Walking",
    1: "Upstairs",
    2: "Downstairs",
    3: "Sitting",
    4: "Standing",
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _euler_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Intrinsic X–Y–Z Euler rotation matrix (float32).
    alpha=rotation around X, beta=around Y, gamma=around Z (all in radians).
    """
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
    cb, sb = float(np.cos(beta)),  float(np.sin(beta))
    cg, sg = float(np.cos(gamma)), float(np.sin(gamma))

    Rx = np.array([[1.,  0.,   0.],
                   [0.,  ca,  -sa],
                   [0.,  sa,   ca]], dtype=np.float32)
    Ry = np.array([[ cb, 0.,  sb],
                   [  0., 1.,  0.],
                   [-sb,  0., cb]], dtype=np.float32)
    Rz = np.array([[cg, -sg, 0.],
                   [sg,  cg, 0.],
                   [ 0., 0., 1.]], dtype=np.float32)
    return Rz @ Ry @ Rx


# ─────────────────────────────────────────────────────────────────────────────
# PhysicalDistortion  —  enhanced 5-operator pipeline (v2)
# ─────────────────────────────────────────────────────────────────────────────

class PhysicalDistortion:
    """
    Programmable augmentation operators that inject the domain shift quantified
    in RQ1_Technical_Report.md into UCI HAR windows, making them statistically
    similar to WISDM signals.

    All operators act on a single window of shape (T, 3) where T = 51 samples
    (2.56 s @ 20 Hz) and the 3 channels are [X, Y, Z] in m/s².

    Pipeline executed by apply_all():
        Op 1  apply_orientation_shift    — gravity axis X→Y + pocket wobble
        Op 2  apply_gravity_attenuation  — gravity L2  9.69 → 7.29 m/s²
        Op 3  apply_amplitude_scaling    — body_acc × 3–5×
        Op 4  apply_spectral_boost       — gait band +20–50% (locomotion only)
        Op 5  apply_noise_shift          — AR(1) colored noise (per-activity)

    Parameters
    ----------
    rng : np.random.Generator, optional
    perturb_deg_range : (float, float)
        Random pocket-wobble range (deg) per axis. Default (5, 10).
    gravity_att_scale : float
        Gravity attenuation factor (WISDM_L2 / UCI_L2 ≈ 0.753). Default auto.
    amp_scale_perturb : (float, float)
        Per-axis random perturbation multiplier range. Default (0.85, 1.15).
        Base scale = AMP_SCALE_PER_AXIS = [2.60, 6.50, 2.67].
    gait_boost_band : (float, float)
        Bandpass range for gait boost (Hz). Default (0.8, 2.0).
    gait_boost_gain : (float, float)
        Uniform additive gain range for gait band. Default (0.2, 0.5).
        Output = window + Uniform[lo,hi] × bandpass_component.
    ar1_alpha : float
        AR(1) noise autocorrelation coefficient. Default ≈ 0.9865.
    gravity_lp_cutoff : float
        Butterworth LPF cutoff for gravity extraction (Hz). Default 0.3.
    gravity_lp_order : int
        LPF filter order. Default 3.
    """

    def __init__(
        self,
        rng:               np.random.Generator | None = None,
        perturb_deg_range: tuple[float, float]        = (5.0, 10.0),
        gravity_att_scale: float                      = GRAVITY_ATT_SCALE,
        amp_scale_perturb: tuple[float, float]        = (0.85, 1.15),
        gait_boost_band:   tuple[float, float]        = (0.8, 2.0),
        gait_boost_gain:   tuple[float, float]        = (0.2, 0.5),
        ar1_alpha:         float                      = AR1_ALPHA,
        gravity_lp_cutoff: float                      = 0.3,
        gravity_lp_order:  int                        = 3,
    ):
        self.rng               = rng or np.random.default_rng()
        self.perturb_deg_range = perturb_deg_range
        self.gravity_att_scale = gravity_att_scale
        self.amp_scale_perturb = amp_scale_perturb
        self.gait_boost_band   = gait_boost_band
        self.gait_boost_gain   = gait_boost_gain
        self.ar1_alpha         = ar1_alpha

        nyq = FS / 2.0
        # Shared Butterworth LPF for gravity extraction (ops 2 & 3)
        self._b_lp, self._a_lp = butter(
            gravity_lp_order, gravity_lp_cutoff / nyq, btype="low"
        )
        # Bandpass for gait frequency boost (op 4)
        self._b_bp, self._a_bp = butter(
            2, [gait_boost_band[0] / nyq, gait_boost_band[1] / nyq], btype="band"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Op 1 — Sensor Mounting Orientation  (Sec 4.2)
    # ─────────────────────────────────────────────────────────────────────

    def apply_orientation_shift(self, window: np.ndarray) -> np.ndarray:
        """
        Rotate UCI HAR's gravity axis from X to Y (WISDM convention), then
        apply a small random perturbation to simulate pocket wobble.

        Physical basis
        --------------
        UCI HAR: rigid waist mount → gravity on +X (mean ≈ +8.58 m/s²).
        WISDM: front trouser pocket, free-swinging → gravity on +Y (≈ +6.51).
        Per-axis Wasserstein ~9.0 m/s², 5× larger than scalar magnitude shift.

        Implementation
        --------------
        Base rotation: 90° around Z
            [ 0 -1  0 ]      maps [1,0,0]→[0,1,0]  (X gravity → Y channel)
            [ 1  0  0 ]
            [ 0  0  1 ]
        Perturbation: ±(5–10°) per axis via Euler rotation, simulating jitter.
        """
        R_base = np.array(
            [[ 0., -1., 0.],
             [ 1.,  0., 0.],
             [ 0.,  0., 1.]], dtype=np.float32
        )
        lo, hi = self.perturb_deg_range
        signs = self.rng.choice([-1.0, 1.0], size=3)
        degs  = self.rng.uniform(lo, hi, size=3) * signs
        R_pert = _euler_rotation_matrix(
            np.deg2rad(degs[0]), np.deg2rad(degs[1]), np.deg2rad(degs[2])
        )
        R = R_pert @ R_base
        return (window @ R.T).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Op 2 — Gravity Magnitude Attenuation  (Sec 4.3)  [NEW in v2]
    # ─────────────────────────────────────────────────────────────────────

    def apply_gravity_attenuation(self, window: np.ndarray) -> np.ndarray:
        """
        Scale down the gravity (DC) component to match WISDM's reduced
        effective gravity magnitude due to free sensor motion in a pocket.

        Physical basis
        --------------
        UCI HAR rigid mount → sensor is fixed relative to body → gravity
        projects fully and consistently onto the accelerometer axes.
        Measured gravity L2 norm ≈ 9.69 m/s² ≈ true g.

        WISDM front-pocket → sensor is free to tilt as the leg moves,
        clothing absorbs some motion, and the pocket geometry varies across
        subjects.  Effective gravity L2 norm ≈ 7.29 m/s² (Fig 3, Sec 4.3).

        Scale factor: 7.29 / 9.69 ≈ 0.753

        Implementation
        --------------
        1. Extract gravity: g = LPF_{0.3 Hz}(window)
        2. Attenuate:       g_att = g × 0.753
        3. Reconstruct:     output = g_att + (window − g)

        The body_acc (dynamic) component is untouched.
        """
        gravity  = filtfilt(self._b_lp, self._a_lp, window.astype(np.float64), axis=0)
        body_acc = window.astype(np.float64) - gravity
        return (gravity * self.gravity_att_scale + body_acc).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Op 3 — Dynamic Amplitude Scaling  (Sec 4.3)
    # ─────────────────────────────────────────────────────────────────────

    def apply_amplitude_scaling(
        self,
        window: np.ndarray,
        label:  int | None = None,
    ) -> np.ndarray:
        """
        Scale body (dynamic) acceleration per-axis and per-activity to match
        WISDM's dynamic range.

        Physical basis
        --------------
        UCI HAR uses a rigid waist mount + Butterworth preprocessing, resulting
        in smaller and more uniform body_acc.  WISDM's phone swings freely in a
        front pocket, producing larger per-step excursions.

        Key finding from empirical measurement
        ---------------------------------------
        The report's approximate global stds (UCI [0.8,1.5,1.5]) are
        misleading because:
          (a) after the 90° Z-rotation, UCI's X axis (vertical bounce at waist,
              which is large: ~2.3 m/s² for walking) maps to the new Y axis;
          (b) the global average conflates locomotion (large std) and static
              activities (near-zero std), giving misleading per-axis estimates.
        Using those approximate values would produce 6.5× on Y — which blows
        up locomotion windows by ~10× relative to target.

        Empirically measured per-activity per-axis scale factors
        (WISDM body_acc std / UCI post-rotation body_acc std):
            Walking:    [2.58, 2.06, 2.94]
            Upstairs:   [2.13, 1.72, 2.08]
            Downstairs: [1.97, 1.22, 2.24]
            Sitting:    [2.83, 4.82, 2.65]
            Standing:   [3.10, 4.79, 3.19]
        (see AMP_SCALE_PER_ACTIVITY)

        A ±15% independent per-axis perturbation simulates inter-subject
        variability in motion amplitude (Sec 4.7).

        Parameters
        ----------
        window : ndarray (T, 3)
        label  : remapped activity label (0–4).  Uses per-activity scale when
                 provided; falls back to AMP_SCALE_GLOBAL otherwise.
        """
        gravity  = filtfilt(self._b_lp, self._a_lp, window.astype(np.float64), axis=0)
        body_acc = window.astype(np.float64) - gravity

        base_scale = (AMP_SCALE_PER_ACTIVITY.get(label, AMP_SCALE_GLOBAL)
                      .astype(np.float64))
        perturb    = self.rng.uniform(*self.amp_scale_perturb, size=3)
        scale      = base_scale * perturb   # (3,)

        return (gravity + body_acc * scale).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Op 4 — Gait Spectral Boost  (Sec 4.9)  [NEW in v2]
    # ─────────────────────────────────────────────────────────────────────

    def apply_spectral_boost(self, window: np.ndarray) -> np.ndarray:
        """
        Boost the gait-frequency band (0.8–2.0 Hz) to simulate WISDM's
        stronger step periodicity for locomotion activities.

        Physical basis
        --------------
        Front-pocket sensor undergoes larger mechanical excursion per step
        than a waist-mounted sensor.  PSD analysis (Fig 9, Sec 4.9) shows:
        - WISDM walking: pronounced power peaks at ~1–2 Hz (step frequency)
        - UCI HAR walking: flatter PSD with less gait periodicity

        This effect is confined to locomotion activities (Walking, Upstairs,
        Downstairs); static activities (Sitting, Standing) are unaffected.

        Implementation
        --------------
        gain ~ Uniform[0.2, 0.5]  (additive, i.e. +20–50%)
        gait_band = BPF_{0.8–2.0 Hz}(window)
        output = window + gain × gait_band

        Note: only called by apply_all() when label ∈ LOCOMOTION_LABELS.
        """
        gait_band = filtfilt(
            self._b_bp, self._a_bp, window.astype(np.float64), axis=0
        )
        gain = self.rng.uniform(*self.gait_boost_gain)
        return (window.astype(np.float64) + gain * gait_band).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Op 5 — AR(1) Colored Noise  (Sec 4.10)  [rewritten in v2]
    # ─────────────────────────────────────────────────────────────────────

    def apply_noise_shift(
        self,
        window: np.ndarray,
        label:  int | None = None,
    ) -> np.ndarray:
        """
        Inject AR(1) colored noise to simultaneously match WISDM's residual
        noise std AND Jerk Std.

        Physical basis
        --------------
        UCI HAR: preprocessed with median + Butterworth filter → low noise.
        WISDM:   entirely raw → sensor drift, clothing vibration, pocket
                 motion add slowly-varying (low-frequency correlated) noise.

        Why v1 (white noise) was wrong:
            White noise std = 2.54 m/s²  →  Jerk Std = σ·FS·√2 ≈ 71.8 m/s³
            WISDM actual Jerk Std = 8.55 m/s³  →  implies effective BW ≈ 0.5 Hz
            WISDM noise is heavily time-correlated, not white.

        AR(1) model: x[t] = α·x[t-1] + ε[t],  ε ~ N(0, σ_ε²)
            Steady-state: Var(x) = σ_ε²/(1−α²)  →  σ_ε = σ·√(1−α²)
            Jerk Std     = σ·FS·√(2(1−α))

        Calibration (from Table 3, Sec 4.10):
            α = 1 − (jerk_std / (σ_noise · FS))² / 2
              = 1 − (8.549 / (2.603 · 20))² / 2
              ≈ 0.9865

        Expected result: noise std ≈ 2.54 m/s², Jerk Std ≈ 8.55 m/s³  ✓

        Per-activity noise targets (Table, Sec 4.10):
            Walking:    σ_WISDM=2.52, σ_UCI=1.02  → σ_add≈2.31 m/s²
            Upstairs:   σ_WISDM=2.06, σ_UCI=0.86  → σ_add≈1.86 m/s²
            Downstairs: σ_WISDM=2.29, σ_UCI=1.19  → σ_add≈1.97 m/s²
            Sitting:    σ_WISDM=0.054,σ_UCI=0.027 → σ_add≈0.046 m/s²
            Standing:   σ_WISDM=0.102,σ_UCI=0.035 → σ_add≈0.096 m/s²

        Parameters
        ----------
        window : ndarray (T, 3)
        label  : remapped activity label (0–4). Uses per-activity sigma when
                 provided; falls back to global target otherwise.
        """
        T, C = window.shape

        # Per-activity noise targets
        if label is not None and label in ACTIVITY_WISDM_NOISE_STD:
            noise_tgt = ACTIVITY_WISDM_NOISE_STD[label]
            noise_src = ACTIVITY_UCI_NOISE_STD[label]
        else:
            noise_tgt = float(WISDM_RESIDUAL_NOISE_STD)
            noise_src = float(UCI_RESIDUAL_NOISE_STD)

        sigma_add = float(np.sqrt(max(noise_tgt**2 - noise_src**2, 0.0)))
        if sigma_add < 1e-9:
            return window.copy()

        colored = self._generate_ar1_noise(T, C, sigma_add, self.ar1_alpha)
        return (window.astype(np.float64) + colored).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helper
    # ─────────────────────────────────────────────────────────────────────

    def _generate_ar1_noise(
        self, T: int, C: int, sigma: float, alpha: float
    ) -> np.ndarray:
        """
        Generate AR(1) colored noise with steady-state std = sigma.

        Process : x[t] = α·x[t-1] + ε[t],  ε ~ N(0, σ_ε²)
        Variance : Var(x) = σ_ε² / (1 − α²)
        → σ_ε = σ · √(1 − α²)

        Expected Jerk Std at steady-state:
            std(Δx · FS) = σ · FS · √(2(1−α))
        """
        sigma_eps = sigma * float(np.sqrt(max(1.0 - alpha ** 2, 1e-12)))
        eps   = self.rng.normal(0.0, sigma_eps, size=(T, C))
        noise = np.empty((T, C), dtype=np.float64)
        # Initialise from the stationary distribution N(0, σ²) so that every
        # sample in the (short) window has the correct steady-state variance.
        # Cold-start with x[0]=ε[0] would give Var(x[t]) ≈ σ²·(1−α^(2t))/(1−α²)
        # which underestimates σ² by ≈30% over T=51 samples at α=0.987.
        noise[0] = self.rng.normal(0.0, sigma, size=C)
        for t in range(1, T):
            noise[t] = alpha * noise[t - 1] + eps[t]
        return noise

    # ─────────────────────────────────────────────────────────────────────
    # Composed pipeline
    # ─────────────────────────────────────────────────────────────────────

    def apply_all(
        self,
        window: np.ndarray,
        label:  int | None = None,
    ) -> np.ndarray:
        """
        Apply all five operators in physically motivated order.

        Pipeline
        --------
        1. orientation_shift    — axis alignment must precede all other ops
        2. gravity_attenuation  — after orientation (gravity is now on Y)
        3. amplitude_scaling    — scale dynamics (before noise)
        4. spectral_boost       — gait enhancement (locomotion only, before noise)
        5. noise_shift          — AR(1) noise injected last (not filtered downstream)

        Parameters
        ----------
        window : ndarray (T, 3)
        label  : remapped activity label (0–4), enables per-activity ops 4 & 5.
        """
        w = self.apply_orientation_shift(window)
        w = self.apply_gravity_attenuation(w)
        w = self.apply_amplitude_scaling(w, label=label)
        if label in LOCOMOTION_LABELS:
            w = self.apply_spectral_boost(w)
        w = self.apply_noise_shift(w, label=label)
        return w

    def transform_batch(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply apply_all independently to every window in a batch.

        Parameters
        ----------
        X : ndarray (N, T, 3)
        y : ndarray (N,) int, optional — enables per-activity ops 4 & 5.

        Returns
        -------
        ndarray (N, T, 3) float32
        """
        if y is not None:
            return np.stack(
                [self.apply_all(X[i], int(y[i])) for i in range(len(X))], axis=0
            )
        return np.stack([self.apply_all(X[i]) for i in range(len(X))], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CrossDatasetLoader
# ─────────────────────────────────────────────────────────────────────────────

class CrossDatasetLoader:
    """
    Produces source-domain (transformed UCI HAR) and target-domain (WISDM)
    datasets for cross-dataset transfer learning.

    Domain assignment
    -----------------
    Source : UCI HAR  → PhysicalDistortion.apply_all() → "pseudo-WISDM"
    Target : WISDM    → subject-based split
        Validation : subjects  1–30
        Test       : subjects 31–36

    Label handling
    --------------
    Only the 5 shared activity classes are retained:
        Walking(0), Upstairs(2), Downstairs(3), Sitting(4), Standing(5)
    Jogging (1) is WISDM-only → dropped.
    Laying  (-1) is UCI-only  → already excluded by load_ucihar().
    Labels remapped to contiguous 0–4 space (see SHARED_LABEL_REMAP).
    """

    def __init__(
        self,
        distortion:       PhysicalDistortion | None = None,
        apply_distortion: bool                      = True,
        verbose:          bool                      = True,
    ):
        self.distortion       = distortion or PhysicalDistortion()
        self.apply_distortion = apply_distortion
        self.verbose          = verbose
        self._uci_raw:   dict | None = None
        self._wisdm_raw: dict | None = None

    def _load_raw(self):
        if self._uci_raw is None:
            self._uci_raw = load_ucihar(verbose=self.verbose)
        if self._wisdm_raw is None:
            self._wisdm_raw = load_wisdm(verbose=self.verbose)

    @staticmethod
    def _filter_shared(
        X: np.ndarray, y: np.ndarray, subj: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask  = np.isin(y, SHARED_UNIFIED_IDS)
        X_f, y_f, s_f = X[mask], y[mask], subj[mask]
        y_out = np.array(
            [SHARED_LABEL_REMAP[int(l)] for l in y_f], dtype=np.int32
        )
        return X_f, y_out, s_f

    def get_source(self) -> dict:
        """
        Source domain: UCI HAR (optionally distorted) with 5-class labels.
        Labels are passed to transform_batch so ops 4 & 5 are activity-aware.
        """
        self._load_raw()
        X, y_raw, subj = (
            self._uci_raw["X"],
            self._uci_raw["y"],
            self._uci_raw["subject"],
        )
        X_f, y_f, s_f = self._filter_shared(X, y_raw, subj)

        if self.apply_distortion:
            if self.verbose:
                print(f"[CrossDatasetLoader] Applying PhysicalDistortion to "
                      f"{len(X_f):,} UCI HAR windows …")
            X_f = self.distortion.transform_batch(X_f, y=y_f)

        return {
            "X": X_f, "y": y_f, "subject": s_f,
            "domain": "source (UCI HAR → pseudo-WISDM)",
            "label_names": SHARED_LABEL_NAMES,
        }

    def get_target_val(self) -> dict:
        """WISDM validation: subjects 1–30, 5 shared classes."""
        return self._get_wisdm_split(WISDM_VAL_SUBJECTS, "target-val (WISDM 1–30)")

    def get_target_test(self) -> dict:
        """WISDM test: subjects 31–36, 5 shared classes."""
        return self._get_wisdm_split(WISDM_TEST_SUBJECTS, "target-test (WISDM 31–36)")

    def get_all(self) -> tuple[dict, dict, dict]:
        """Convenience: returns (source, val, test). Data cached after first call."""
        return self.get_source(), self.get_target_val(), self.get_target_test()

    def _get_wisdm_split(self, subject_ids: list[int], domain_label: str) -> dict:
        self._load_raw()
        X, y_raw, subj = (
            self._wisdm_raw["X"],
            self._wisdm_raw["y"],
            self._wisdm_raw["subject"],
        )
        X_f, y_f, s_f = self._filter_shared(X, y_raw, subj)
        mask = np.isin(s_f, subject_ids)
        return {
            "X": X_f[mask], "y": y_f[mask], "subject": s_f[mask],
            "domain": domain_label,
            "label_names": SHARED_LABEL_NAMES,
        }

    def summary(self):
        """Print shape and class distribution for all three splits."""
        src, val, test = self.get_all()
        for label, ds in [
            ("Source  (UCI HAR → pseudo-WISDM)", src),
            ("Val     (WISDM subjects  1–30)",   val),
            ("Test    (WISDM subjects 31–36)",   test),
        ]:
            print(f"\n{'─'*60}")
            print(f"  {label}")
            print(f"  Shape   : X={ds['X'].shape}  y={ds['y'].shape}")
            print(f"  Subjects: {np.unique(ds['subject']).tolist()}")
            _print_class_dist(ds["y"])


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _print_class_dist(y: np.ndarray):
    total = len(y)
    print(f"  {'Class':<15} {'Count':>7}  {'%':>6}")
    print(f"  {'─'*33}")
    for uid, uname in SHARED_LABEL_NAMES.items():
        cnt = int((y == uid).sum())
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  {uname:<15} {cnt:>7,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<15} {total:>7,}  100.0%")


def compute_signal_stats(X: np.ndarray) -> dict:
    """
    Compute per-axis statistics for distortion fidelity verification.

    Parameters
    ----------
    X : ndarray (N, T, 3)

    Returns
    -------
    dict with: gravity_mean, dynamic_std, residual_noise_std, jerk_std,
               gravity_l2 (scalar)
    """
    flat = X.reshape(-1, 3).astype(np.float64)
    gravity_mean = flat.mean(axis=0)
    dynamic      = flat - gravity_mean
    dynamic_std  = dynamic.std(axis=0)
    residual_std = flat.std()
    gravity_l2   = float(np.linalg.norm(gravity_mean))

    jerk     = np.diff(X.astype(np.float64), axis=1) * FS   # (N, T-1, 3)
    jerk_std = float(jerk.std())

    return {
        "gravity_mean":       gravity_mean,
        "gravity_l2":         gravity_l2,
        "dynamic_std":        dynamic_std,
        "residual_noise_std": residual_std,
        "jerk_std":           jerk_std,
    }


def compute_mmd(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    n_sample: int           = 1500,
    pca_dim:  int           = 2,
    bandwidths: tuple       = (0.1, 1.0, 10.0),
    seed:     int           = 0,
) -> float:
    """
    Multi-kernel MMD² (square-rooted) between source and target windows,
    following the methodology in Sec 4.7 / Fig 7 of RQ1_Technical_Report.md.

    Parameters
    ----------
    X_src, X_tgt : ndarray (N, T, 3)  — window batches
    n_sample : int  — subsample size per domain (for O(n²) kernel cost)
    pca_dim  : int  — PCA dimensionality before kernel computation
    bandwidths : tuple — RBF kernel bandwidths σ (sum of kernels = multi-MMD)
    seed     : int  — RNG seed for reproducible subsampling

    Returns
    -------
    float  — √MMD² (non-negative; 0 = identical distributions)
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel

    rng   = np.random.default_rng(seed)
    n_s   = min(n_sample, len(X_src))
    n_t   = min(n_sample, len(X_tgt))
    idx_s = rng.choice(len(X_src), n_s, replace=False)
    idx_t = rng.choice(len(X_tgt), n_t, replace=False)

    Xs = X_src[idx_s].reshape(n_s, -1).astype(np.float64)
    Xt = X_tgt[idx_t].reshape(n_t, -1).astype(np.float64)

    pca    = PCA(n_components=pca_dim, random_state=seed)
    Z_all  = pca.fit_transform(np.vstack([Xs, Xt]))
    Zs, Zt = Z_all[:n_s], Z_all[n_s:]

    mmd2 = 0.0
    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * sigma ** 2)
        Kss = rbf_kernel(Zs, Zs, gamma)
        Ktt = rbf_kernel(Zt, Zt, gamma)
        Kst = rbf_kernel(Zs, Zt, gamma)
        mmd2 += float(Kss.mean() - 2.0 * Kst.mean() + Ktt.mean())

    return float(np.sqrt(max(mmd2, 0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check  (run as: python covariate_shift_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 66)
    print("  covariate_shift_engine.py v2 — sanity check")
    print("=" * 66)

    rng = np.random.default_rng(42)
    dist = PhysicalDistortion(rng=rng)
    T = 51  # window length

    # ── [0] Constants: verify AR(1) alpha derivation ─────────────────────
    print(f"\n[0] AR(1) constant verification")
    pred_jerk = WISDM_RESIDUAL_NOISE_STD * FS * np.sqrt(2 * (1 - AR1_ALPHA))
    print(f"   AR1_ALPHA       = {AR1_ALPHA:.6f}")
    print(f"   Predicted jerk  = {pred_jerk:.3f} m/s³  (target {WISDM_JERK_STD:.3f})")
    assert abs(pred_jerk - WISDM_JERK_STD) < 0.01, "AR1_ALPHA derivation error"
    print(f"   ✓ AR1 jerk derivation correct")

    # ── [1] Op 1: Orientation shift ──────────────────────────────────────
    print(f"\n[1] apply_orientation_shift")
    w = np.zeros((T, 3), dtype=np.float32)
    w[:, 0] = 8.58  # UCI: gravity on X
    r = dist.apply_orientation_shift(w)
    print(f"   Input : X={w[:,0].mean():+.2f}  Y={w[:,1].mean():+.2f}  Z={w[:,2].mean():+.2f}")
    print(f"   Output: X={r[:,0].mean():+.2f}  Y={r[:,1].mean():+.2f}  Z={r[:,2].mean():+.2f}")
    print(f"   Target: X≈{WISDM_GRAVITY_MEAN[0]:+.2f}  Y≈{WISDM_GRAVITY_MEAN[1]:+.2f}")
    assert r[:, 1].mean() > 5.0, "gravity did not migrate to Y"
    print(f"   ✓ gravity migrated to Y axis")

    # ── [2] Op 2: Gravity attenuation ────────────────────────────────────
    print(f"\n[2] apply_gravity_attenuation")
    # Simulate post-rotation signal: gravity on Y
    w2 = np.zeros((T, 3), dtype=np.float32)
    w2[:, 1] = 9.69  # before attenuation, gravity L2 ≈ 9.69
    a = dist.apply_gravity_attenuation(w2)
    g_l2_before = np.linalg.norm(w2.mean(axis=0))
    g_l2_after  = np.linalg.norm(a.mean(axis=0))
    print(f"   Gravity L2 before: {g_l2_before:.3f} m/s²  (UCI)")
    print(f"   Gravity L2 after : {g_l2_after:.3f} m/s²")
    print(f"   Target           : {WISDM_GRAVITY_L2:.3f} m/s²  (WISDM)")
    assert abs(g_l2_after - WISDM_GRAVITY_L2) < 0.1, "gravity attenuation off"
    print(f"   ✓ gravity magnitude reduced to WISDM level")

    # ── [3] Op 3: Per-activity per-axis amplitude scaling ────────────────
    print(f"\n[3] apply_amplitude_scaling  (per-activity, per-axis)")
    t_ax = np.linspace(0, 2.56, T, dtype=np.float32)
    from scipy.signal import butter as _bt, filtfilt as _ff
    _b, _a = _bt(3, 0.3/10.0, btype='low')

    # Test: Walking (label=0) — empirical post-rot UCI stds ~[1.70, 2.28, 1.35]
    dyn_w = np.column_stack([
        1.70 * np.sin(2 * np.pi * 1.2 * t_ax),
        2.28 * np.sin(2 * np.pi * 1.0 * t_ax),
        1.35 * np.sin(2 * np.pi * 0.8 * t_ax),
    ]).astype(np.float32)
    g_vec = np.array([0., 7.29, 0.], dtype=np.float32)
    w3 = (np.tile(g_vec, (T, 1)) + dyn_w).astype(np.float32)
    s = dist.apply_amplitude_scaling(w3, label=0)
    g_out = _ff(_b, _a, s.astype(np.float64), axis=0)
    dyn_out = (s - g_out).std(axis=0)
    wisdm_walk = np.array([4.37, 4.68, 3.97])
    print(f"   Walking — input  std: {dyn_w.std(axis=0)}")
    print(f"   Walking — output std: {dyn_out}")
    print(f"   Walking — WISDM ref : {wisdm_walk}")
    for ax, name in enumerate('XYZ'):
        ratio = dyn_out[ax] / wisdm_walk[ax]
        flag = "✓" if 0.7 < ratio < 1.3 else "△"
        print(f"     {name}: {dyn_out[ax]:.2f}  target {wisdm_walk[ax]:.2f}  ratio {ratio:.2f}  {flag}")
    assert all(dyn_out > dyn_w.std(axis=0) * 1.0), "scaling didn't increase amplitude"
    print(f"   ✓ per-activity per-axis scaling applied")

    # ── [4] Op 4: Spectral boost (locomotion) ────────────────────────────
    print(f"\n[4] apply_spectral_boost")
    # Walking window with gait signal at 1.2 Hz (reuse dyn_w from [3])
    w4 = dyn_w.copy()
    w4_boosted = dist.apply_spectral_boost(w4)
    # Energy in 0.8–2 Hz band should increase
    from scipy.signal import butter as _bt2, filtfilt as _ff2
    _bb, _ab = _bt2(2, [0.8/10.0, 2.0/10.0], btype='band')
    band_before = _ff2(_bb, _ab, w4.astype(np.float64), axis=0).std()
    band_after  = _ff2(_bb, _ab, w4_boosted.astype(np.float64), axis=0).std()
    print(f"   Gait band std before: {band_before:.4f}")
    print(f"   Gait band std after : {band_after:.4f}")
    print(f"   Band energy increase: {(band_after/band_before - 1)*100:.1f}%  (target 20–50%)")
    assert band_after > band_before * 1.1, "spectral boost too small"
    print(f"   ✓ gait band energy boosted")

    # ── [5] Op 5: AR(1) colored noise ────────────────────────────────────
    print(f"\n[5] apply_noise_shift  (Walking, label=0)")
    N_samples = 2000
    noise_acc = []
    for _ in range(N_samples):
        z = dist.apply_noise_shift(np.zeros((T, 3), dtype=np.float32), label=0)
        noise_acc.append(z)
    noise_arr = np.stack(noise_acc, axis=0)  # (N, T, 3)
    noise_std  = float(noise_arr.std())
    jerk_arr   = np.diff(noise_arr, axis=1) * FS
    jerk_std   = float(jerk_arr.std())
    sigma_add_expected = float(np.sqrt(ACTIVITY_WISDM_NOISE_STD[0]**2 - ACTIVITY_UCI_NOISE_STD[0]**2))
    print(f"   σ_add expected    : {sigma_add_expected:.3f} m/s²")
    print(f"   Noise std (actual): {noise_std:.3f} m/s²")
    pred_jerk_add = sigma_add_expected * FS * np.sqrt(2 * (1 - AR1_ALPHA))
    print(f"   Jerk Std (actual) : {jerk_std:.3f} m/s³  (predicted {pred_jerk_add:.3f})")
    print(f"   v1 white noise would give Jerk Std ≈ {sigma_add_expected * FS * np.sqrt(2):.1f} m/s³")
    assert abs(noise_std - sigma_add_expected) < 0.15, "noise std off"
    assert jerk_std < 20.0, "jerk std too high (colored noise not working)"
    print(f"   ✓ AR(1) colored noise: correct std AND jerk std")

    # ── [5b] Per-activity noise: Sitting (label=3) ───────────────────────
    print(f"\n[5b] apply_noise_shift  (Sitting, label=3)")
    sit_acc = []
    for _ in range(N_samples):
        z = dist.apply_noise_shift(np.zeros((T, 3), dtype=np.float32), label=3)
        sit_acc.append(z)
    sit_arr  = np.stack(sit_acc, axis=0)
    sit_std  = float(sit_arr.std())
    walk_sigma_add = float(np.sqrt(ACTIVITY_WISDM_NOISE_STD[0]**2 - ACTIVITY_UCI_NOISE_STD[0]**2))
    sit_sigma_add  = float(np.sqrt(ACTIVITY_WISDM_NOISE_STD[3]**2 - ACTIVITY_UCI_NOISE_STD[3]**2))
    print(f"   Walking σ_add: {walk_sigma_add:.3f} m/s²  |  Sitting σ_add: {sit_sigma_add:.4f} m/s²")
    print(f"   Sitting noise std (actual): {sit_std:.4f} m/s²")
    assert sit_std < noise_std * 0.1, "sitting noise should be much lower than walking"
    print(f"   ✓ per-activity noise: Sitting is {noise_std/sit_std:.0f}× quieter than Walking")

    # ── [6] Full pipeline: CrossDatasetLoader ────────────────────────────
    print(f"\n[6] CrossDatasetLoader — full pipeline")
    t0 = time.time()
    loader = CrossDatasetLoader(verbose=True)
    loader.summary()
    print(f"\n   Elapsed: {time.time()-t0:.1f}s")

    # ── [7] Statistical alignment: source vs target ───────────────────────
    print(f"\n[7] Signal statistics — source vs WISDM target")
    src, val, _ = loader.get_all()
    ss = compute_signal_stats(src["X"])
    vs = compute_signal_stats(val["X"])

    rows = [
        ("Gravity mean X (m/s²)",    ss["gravity_mean"][0],  vs["gravity_mean"][0],  WISDM_GRAVITY_MEAN[0]),
        ("Gravity mean Y (m/s²)",    ss["gravity_mean"][1],  vs["gravity_mean"][1],  WISDM_GRAVITY_MEAN[1]),
        ("Gravity mean Z (m/s²)",    ss["gravity_mean"][2],  vs["gravity_mean"][2],  WISDM_GRAVITY_MEAN[2]),
        ("Gravity L2 norm (m/s²)",   ss["gravity_l2"],       vs["gravity_l2"],       WISDM_GRAVITY_L2),
        ("Residual noise std",        ss["residual_noise_std"],vs["residual_noise_std"], WISDM_RESIDUAL_NOISE_STD),
        ("Jerk std (m/s³)",           ss["jerk_std"],         vs["jerk_std"],         WISDM_JERK_STD),
        ("Dynamic std X",             ss["dynamic_std"][0],   vs["dynamic_std"][0],   WISDM_DYN_STD[0]),
        ("Dynamic std Y",             ss["dynamic_std"][1],   vs["dynamic_std"][1],   WISDM_DYN_STD[1]),
        ("Dynamic std Z",             ss["dynamic_std"][2],   vs["dynamic_std"][2],   WISDM_DYN_STD[2]),
    ]
    hdr = f"  {'Metric':<28} {'Source (pseudo)':>16}  {'WISDM target':>14}  {'Report ref':>11}"
    print(hdr)
    print(f"  {'─'*74}")
    for name, src_v, tgt_v, ref_v in rows:
        delta_pct = abs(src_v - tgt_v) / (abs(tgt_v) + 1e-9) * 100
        flag = "✓" if delta_pct < 30 else "△"
        print(f"  {name:<28} {src_v:>16.3f}  {tgt_v:>14.3f}  {ref_v:>11.3f}  {flag}")

    # ── [8] MMD: before vs after distortion ─────────────────────────────
    print(f"\n[8] MMD comparison — before vs after PhysicalDistortion")
    print(f"   Loading raw UCI HAR (no distortion) …")
    loader_raw = CrossDatasetLoader(apply_distortion=False, verbose=False)
    src_raw, val_raw, _ = loader_raw.get_all()

    print(f"   Computing MMD (raw UCI vs WISDM val) …")
    mmd_before = compute_mmd(src_raw["X"], val_raw["X"])

    print(f"   Computing MMD (pseudo-WISDM vs WISDM val) …")
    mmd_after = compute_mmd(src["X"], val["X"])

    reduction = (mmd_before - mmd_after) / mmd_before * 100
    print(f"\n   MMD (raw UCI → WISDM)      : {mmd_before:.4f}")
    print(f"   MMD (pseudo-WISDM → WISDM) : {mmd_after:.4f}")
    print(f"   Reduction                  : {reduction:.1f}%")
    if reduction > 0:
        print(f"   ✓ PhysicalDistortion reduces feature-space gap")
    else:
        print(f"   △ MMD increased — distortion may be overcorrecting")

    print("\nDone ✓")
