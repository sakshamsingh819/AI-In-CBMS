"""
Synthetic antigravity data generator.

Generates 2,000 synthetic 1024-sample vibration windows using a modified
Jeffcott rotor model where the gravitational force term is:

    F_g = m * g * α,  α ∈ [0.0, 0.25]  (0–25% of normal gravity loading)

The generator:
  - Modulates the 1× harmonic amplitude inversely with α
  - Elevates sub-synchronous energy by 15–40% as α decreases
  - Injects a non-integer precession component at 0.43× shaft frequency
  - Adds bandlimited white noise at SNR = 25 dB (IIS3DWB noise floor)
  - Saves output as:
      data/synthetic/antigravity_1024.npy   (2000, 1024, 3)
      data/synthetic/antigravity_labels.npy (2000,) all = 6 (antigravity class index)

Usage:
    python backend/scripts/generate_antigravity_data.py

Outputs a summary report on stdout: mean GLI, mean SER, mean HCC.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

# ── Parameters ────────────────────────────────────────────────────────────────

N_SAMPLES: int = 2000        # number of windows to generate
WINDOW_LEN: int = 1024       # samples per window
N_AXES: int = 3              # X, Y, Z
FS: int = 26667              # Sampling rate (IIS3DWB), Hz
SHAFT_RPM: float = 1500.0    # Base shaft speed
FR: float = SHAFT_RPM / 60  # Shaft frequency, Hz = 25 Hz
GRAVITY_MS2: float = 9.81   # m/s²
SNR_DB: float = 25.0         # Noise SNR matching IIS3DWB noise floor
PRECESSION_RATIO: float = 0.43  # Non-integer precession (gyroscopic coupling)
CLASS_LABEL: int = 6         # "antigravity" class index (0-indexed, 7-class problem)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


# ── Jeffcott rotor signal synthesis ───────────────────────────────────────────


def _generate_window(
    rng: np.random.Generator,
    alpha: float,
) -> np.ndarray:
    """
    Generate a single 1024-sample × 3-axis vibration window.

    Args:
        rng:   NumPy random generator (seeded).
        alpha: Gravity loading factor ∈ [0.0, 0.25].

    Returns:
        Signal array of shape (1024, 3) in m/s².
    """
    t = np.arange(WINDOW_LEN) / FS

    # ── 1× harmonic: amplitude inversely proportional to gravity load ────────
    # Under full gravity (α=1): A1 ≈ 1.0 m/s²
    # Under near-zero gravity (α≈0): A1 → 0.1 m/s²
    A1 = 0.1 + 0.9 * alpha / 1.0  # linear interpolation, clamped to [0.1, 0.35]
    phi1 = rng.uniform(0, 2 * np.pi)

    # ── 2×, 3× harmonics ────────────────────────────────────────────────────
    A2 = A1 * rng.uniform(0.15, 0.30)
    A3 = A1 * rng.uniform(0.05, 0.15)
    phi2 = rng.uniform(0, 2 * np.pi)
    phi3 = rng.uniform(0, 2 * np.pi)

    # ── Sub-synchronous energy elevation (0.1× – 0.4×) ───────────────────────
    # Elevation: 15–40% of broadband energy as α decreases
    sub_elevation = rng.uniform(0.15, 0.40) * (1 - alpha / 0.25)
    n_sub = 5  # number of sub-synchronous components
    sub_freqs = rng.uniform(0.1 * FR, 0.4 * FR, size=n_sub)
    sub_amps = rng.uniform(0.05, 0.15, size=n_sub) * (1 + sub_elevation)
    sub_phases = rng.uniform(0, 2 * np.pi, size=n_sub)

    # ── Non-integer precession at 0.43× (gyroscopic coupling) ────────────────
    prec_freq = PRECESSION_RATIO * FR
    A_prec = rng.uniform(0.08, 0.18) * (1 - alpha / 0.25 + 0.1)
    phi_prec = rng.uniform(0, 2 * np.pi)

    # ── Reduced DC offset on Z axis (GLI effect) ─────────────────────────────
    # Under normal gravity, Z-axis sees ~1g DC offset.
    # As α→0, DC offset → 0.
    dc_z = GRAVITY_MS2 * alpha  # m/s²

    # ── Assemble deterministic signal ─────────────────────────────────────────
    # X axis: primarily 1× + 2× + sub-synchronous
    x = (
        A1 * np.sin(2 * np.pi * FR * t + phi1)
        + A2 * np.sin(2 * np.pi * 2 * FR * t + phi2)
        + A3 * np.sin(2 * np.pi * 3 * FR * t + phi3)
        + A_prec * np.sin(2 * np.pi * prec_freq * t + phi_prec)
        + sum(sub_amps[i] * np.sin(2 * np.pi * sub_freqs[i] * t + sub_phases[i])
              for i in range(n_sub))
    )

    # Y axis: phase-shifted version of X (orbital motion)
    phase_offset = np.pi / 2 * (1 - alpha)  # orbit becomes less circular under low gravity
    y = (
        A1 * np.sin(2 * np.pi * FR * t + phi1 + phase_offset)
        + A2 * np.sin(2 * np.pi * 2 * FR * t + phi2 + phase_offset)
        + sum(sub_amps[i] * np.sin(2 * np.pi * sub_freqs[i] * t + sub_phases[i] + 0.3)
              for i in range(n_sub))
    )

    # Z axis: vertical — carries gravity (DC) + axial effects
    z = (
        dc_z
        + A1 * 0.3 * np.sin(2 * np.pi * FR * t + phi1 - np.pi / 3)
        + A_prec * 0.5 * np.sin(2 * np.pi * prec_freq * t + phi_prec + np.pi / 6)
    )

    # ── Add bandlimited white noise at 25 dB SNR ──────────────────────────────
    for axis_sig in [x, y, z]:
        sig_power = np.mean(axis_sig ** 2)
        noise_power = sig_power / (10 ** (SNR_DB / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=WINDOW_LEN)
        # Bandlimit noise to [100 Hz, 12 kHz] (IIS3DWB passband)
        from scipy.signal import butter, filtfilt  # noqa: PLC0415
        b, a = butter(4, [100 / (FS / 2), 12000 / (FS / 2)], btype="band")
        axis_sig += filtfilt(b, a, noise)

    return np.stack([x, y, z], axis=1)  # (1024, 3)


# ── Feature summary helpers ────────────────────────────────────────────────────


def _compute_gli_batch(windows: np.ndarray) -> np.ndarray:
    """Compute GLI for all windows: DC(Z) / 9.81."""
    return np.abs(np.mean(windows[:, :, 2], axis=1)) / GRAVITY_MS2


def _compute_ser_batch(windows: np.ndarray) -> np.ndarray:
    """Approximate SER for all windows."""
    from scipy.signal import welch  # noqa: PLC0415
    sers = []
    for w in windows:
        combined = w.mean(axis=1)
        freqs, psd = welch(combined, fs=FS, nperseg=256)
        total = np.sum(psd) + 1e-12
        sub_mask = (freqs >= 0.1 * FR) & (freqs <= 0.4 * FR)
        sers.append(np.sum(psd[sub_mask]) / total)
    return np.array(sers)


def _compute_hcc_batch(windows: np.ndarray) -> np.ndarray:
    """Approximate HCC for all windows."""
    hccs = []
    for w in windows:
        combined = w.mean(axis=1)
        n = len(combined)
        fft_mag = np.abs(np.fft.rfft(combined)) * 2.0 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / FS)
        bin_1x = int(np.argmin(np.abs(freqs - FR)))
        measured_1x = float(np.max(fft_mag[max(0, bin_1x - 5):bin_1x + 6]))
        hccs.append((1.0 - measured_1x) / 1.0)
    return np.array(hccs)


# ── Main generation routine ────────────────────────────────────────────────────


def generate(seed: int = 42) -> None:
    """Generate all 2000 synthetic antigravity windows and save to disk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    logger.info(f"Generating {N_SAMPLES} synthetic antigravity windows …")
    logger.info(f"  α range : [0.00, 0.25] (0–25% gravity loading)")
    logger.info(f"  SNR     : {SNR_DB} dB")
    logger.info(f"  Window  : {WINDOW_LEN} samples @ {FS} Hz")

    # Sample α values uniformly across [0.0, 0.25]
    alphas = rng.uniform(0.0, 0.25, size=N_SAMPLES)

    windows = np.zeros((N_SAMPLES, WINDOW_LEN, N_AXES), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        windows[i] = _generate_window(rng, alpha)
        if (i + 1) % 500 == 0:
            logger.info(f"  Generated {i + 1}/{N_SAMPLES} windows …")

    labels = np.full(N_SAMPLES, CLASS_LABEL, dtype=np.int32)

    # ── Save ──────────────────────────────────────────────────────────────────
    data_path = OUT_DIR / "antigravity_1024.npy"
    labels_path = OUT_DIR / "antigravity_labels.npy"
    np.save(data_path, windows)
    np.save(labels_path, labels)
    logger.success(f"Saved data  → {data_path}")
    logger.success(f"Saved labels→ {labels_path}")

    # ── Summary report (physical plausibility check) ──────────────────────────
    gli_vals = _compute_gli_batch(windows)
    ser_vals = _compute_ser_batch(windows)
    hcc_vals = _compute_hcc_batch(windows)

    print("\n" + "═" * 55)
    print("  ANTIGRAVITY SYNTHETIC DATA — SUMMARY REPORT")
    print("═" * 55)
    print(f"  Windows generated : {N_SAMPLES}")
    print(f"  Shape             : {windows.shape}")
    print(f"  α (mean ± std)    : {alphas.mean():.3f} ± {alphas.std():.3f}")
    print(f"  GLI  (mean ± std) : {gli_vals.mean():.3f} ± {gli_vals.std():.3f}")
    print(f"  SER  (mean ± std) : {ser_vals.mean():.3f} ± {ser_vals.std():.3f}")
    print(f"  HCC  (mean ± std) : {hcc_vals.mean():.3f} ± {hcc_vals.std():.3f}")
    print("─" * 55)
    print("  Physical plausibility checks:")
    print(f"    GLI < 0.30 (reduced gravity)  : {(gli_vals < 0.30).mean() * 100:.1f}% of windows")
    print(f"    SER > 0.10 (elevated sub-sync) : {(ser_vals > 0.10).mean() * 100:.1f}% of windows")
    print(f"    HCC > 0.50 (1× suppressed)    : {(hcc_vals > 0.50).mean() * 100:.1f}% of windows")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    generate()
