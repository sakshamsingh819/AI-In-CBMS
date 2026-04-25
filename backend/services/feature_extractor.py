"""
Feature extraction pipeline for the Condition Monitoring System.

Extracts a 54-element feature vector from raw 3-axis vibration signals:
  • Features  1–48  : Baseline vibration features (time-domain, frequency-domain,
                      bearing fault harmonics, cross-axis statistics, wavelets)
  • Features 49–54  : Antigravity-sensitive features (GLI, SER, HCC, CPC, BLZA, GPI)

All functions are pure (no side-effects) and operate on NumPy arrays.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import coherence, welch

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE: int = int(os.getenv("SENSOR_SAMPLE_RATE", 26667))
GRAVITY_MS2: float = 9.81  # m/s²

# Bearing geometry constants (SKF 6205 as reference)
# BPFI = (N/2) * (1 + Bd/Pd * cos(α)) * fr
# BPFO = (N/2) * (1 - Bd/Pd * cos(α)) * fr
BPFI_COEFF: float = 5.415   # BPFI / shaft frequency
BPFO_COEFF: float = 3.585   # BPFO / shaft frequency
BSF_COEFF: float = 2.357    # BSF  / shaft frequency
FTF_COEFF: float = 0.3983   # FTF  / shaft frequency

# Antigravity sub-synchronous band (fraction of shaft frequency)
SUB_SYNC_LOW: float = 0.1
SUB_SYNC_HIGH: float = 0.4

# Precession frequency ratio (gyroscopic coupling, see CONTEXT.md Open Questions)
PRECESSION_RATIO: float = float(os.getenv("PRECESSION_RATIO", 0.43))

# Expected 1× harmonic amplitude under normal gravity (m/s², normalised)
EXPECTED_1X_AMP: float = float(os.getenv("EXPECTED_1X_AMPLITUDE", 1.0))

# ── Feature name registry (single source of truth) ────────────────────────────

FEATURE_NAMES: list[str] = [
    # ── Time-domain per axis (15 features, 5 × 3 axes) ────────────────────
    "x_rms", "x_peak", "x_crest_factor", "x_kurtosis", "x_skewness",
    "y_rms", "y_peak", "y_crest_factor", "y_kurtosis", "y_skewness",
    "z_rms", "z_peak", "z_crest_factor", "z_kurtosis", "z_skewness",
    # ── Frequency-domain per axis (9 features, 3 × 3 axes) ────────────────
    "x_spectral_centroid", "x_spectral_spread", "x_spectral_entropy",
    "y_spectral_centroid", "y_spectral_spread", "y_spectral_entropy",
    "z_spectral_centroid", "z_spectral_spread", "z_spectral_entropy",
    # ── Bearing fault harmonics (4 features) ──────────────────────────────
    "bpfi_amplitude", "bpfo_amplitude", "bsf_amplitude", "ftf_amplitude",
    # ── Shaft harmonics on combined signal (4 features) ───────────────────
    "harmonic_1x", "harmonic_2x", "harmonic_3x", "harmonic_4x",
    # ── Envelope analysis (2 features) ───────────────────────────────────
    "envelope_rms", "envelope_kurtosis",
    # ── Cross-axis coherence (3 features) ─────────────────────────────────
    "coherence_xy", "coherence_xz", "coherence_yz",
    # ── Band energy ratios (4 features) ───────────────────────────────────
    "energy_low_band",   # < 1 kHz
    "energy_mid_band",   # 1–5 kHz
    "energy_high_band",  # 5–10 kHz
    "energy_vhigh_band", # > 10 kHz
    # ── Wavelet detail energy (4 features, levels 1–4) ────────────────────
    "wavelet_d1_energy", "wavelet_d2_energy",
    "wavelet_d3_energy", "wavelet_d4_energy",
    # ── Additional statistical features (3 features) ──────────────────────
    "peak_to_peak_combined", "shape_factor_combined", "impulse_factor_combined",
    # ── [49–54] Antigravity-sensitive features ─────────────────────────────
    "gli",   # Gravity Load Index
    "ser",   # Sub-synchronous Energy Ratio
    "hcc",   # Harmonic Cancellation Coefficient
    "cpc",   # Cross-axis Phase Coherence
    "blza",  # Bearing Load Zone Asymmetry
    "gpi",   # Gyroscopic Precession Index
]

assert len(FEATURE_NAMES) == 54, f"Expected 54 features, got {len(FEATURE_NAMES)}"


# ── Helper: spectral amplitude at a target frequency ─────────────────────────

def _spectral_amplitude(
    sig: np.ndarray, target_hz: float, fs: int, window: int = 5
) -> float:
    """Return the peak FFT amplitude within ±window bins of target_hz."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig)) * 2.0 / n
    bin_idx = int(np.argmin(np.abs(freqs - target_hz)))
    lo = max(0, bin_idx - window)
    hi = min(len(fft_mag), bin_idx + window + 1)
    return float(np.max(fft_mag[lo:hi]))


def _shaft_hz(shaft_rpm: float) -> float:
    """Convert shaft RPM to shaft frequency in Hz."""
    return shaft_rpm / 60.0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Time-domain features (per axis)
# ══════════════════════════════════════════════════════════════════════════════


def _time_features(sig: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute (rms, peak, crest_factor, kurtosis, skewness) for one axis."""
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))
    crest = peak / (rms + 1e-12)
    n = len(sig)
    mean = np.mean(sig)
    std = np.std(sig) + 1e-12
    kurt = float(np.mean(((sig - mean) / std) ** 4))
    skew = float(np.mean(((sig - mean) / std) ** 3))
    return rms, peak, crest, kurt, skew


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Frequency-domain features (per axis)
# ══════════════════════════════════════════════════════════════════════════════


def _freq_features(
    sig: np.ndarray, fs: int
) -> tuple[float, float, float]:
    """Compute (spectral_centroid, spectral_spread, spectral_entropy) for one axis."""
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    centroid = float(np.sum(freqs * psd_norm))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm)))
    entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
    return centroid, spread, entropy


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Bearing fault harmonics
# ══════════════════════════════════════════════════════════════════════════════


def _bearing_harmonics(
    sig: np.ndarray, shaft_rpm: float, fs: int
) -> tuple[float, float, float, float]:
    """Compute BPFI, BPFO, BSF, FTF harmonic amplitudes from the combined signal."""
    fr = _shaft_hz(shaft_rpm)
    bpfi = _spectral_amplitude(sig, BPFI_COEFF * fr, fs)
    bpfo = _spectral_amplitude(sig, BPFO_COEFF * fr, fs)
    bsf  = _spectral_amplitude(sig, BSF_COEFF  * fr, fs)
    ftf  = _spectral_amplitude(sig, FTF_COEFF  * fr, fs)
    return bpfi, bpfo, bsf, ftf


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Shaft harmonics
# ══════════════════════════════════════════════════════════════════════════════


def _shaft_harmonics(
    sig: np.ndarray, shaft_rpm: float, fs: int
) -> tuple[float, float, float, float]:
    """Compute 1×, 2×, 3×, 4× shaft harmonic amplitudes."""
    fr = _shaft_hz(shaft_rpm)
    h1 = _spectral_amplitude(sig, 1.0 * fr, fs)
    h2 = _spectral_amplitude(sig, 2.0 * fr, fs)
    h3 = _spectral_amplitude(sig, 3.0 * fr, fs)
    h4 = _spectral_amplitude(sig, 4.0 * fr, fs)
    return h1, h2, h3, h4


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Envelope analysis
# ══════════════════════════════════════════════════════════════════════════════


def _envelope_features(sig: np.ndarray) -> tuple[float, float]:
    """Compute (envelope_rms, envelope_kurtosis) via Hilbert transform."""
    analytic = sp_signal.hilbert(sig)
    env = np.abs(analytic)
    rms = float(np.sqrt(np.mean(env ** 2)))
    mean = np.mean(env)
    std = np.std(env) + 1e-12
    kurt = float(np.mean(((env - mean) / std) ** 4))
    return rms, kurt


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Cross-axis coherence
# ══════════════════════════════════════════════════════════════════════════════


def _coherence_features(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, fs: int
) -> tuple[float, float, float]:
    """Compute mean magnitude-squared coherence between axis pairs."""
    nperseg = min(256, len(x))
    _, cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    _, cxz = coherence(x, z, fs=fs, nperseg=nperseg)
    _, cyz = coherence(y, z, fs=fs, nperseg=nperseg)
    return float(np.mean(cxy)), float(np.mean(cxz)), float(np.mean(cyz))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Band energy ratios
# ══════════════════════════════════════════════════════════════════════════════


def _band_energy_ratios(
    sig: np.ndarray, fs: int
) -> tuple[float, float, float, float]:
    """Compute energy fraction in four frequency bands."""
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    total = np.sum(psd) + 1e-12
    bands = [(0, 1000), (1000, 5000), (5000, 10000), (10000, fs // 2)]
    out = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out.append(float(np.sum(psd[mask]) / total))
    return tuple(out)  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Wavelet energy (manual DWT approximation via Haar-like filter)
# ══════════════════════════════════════════════════════════════════════════════


def _wavelet_energy(sig: np.ndarray) -> tuple[float, float, float, float]:
    """Compute energy in detail coefficient levels 1–4 via iterated downsampling."""
    energies = []
    current = sig.copy()
    for _ in range(4):
        if len(current) < 4:
            energies.append(0.0)
            continue
        # DWT detail coefficients via differencing (Haar)
        detail = (current[::2] - current[1::2]) / np.sqrt(2)
        approx = (current[::2] + current[1::2]) / np.sqrt(2)
        energies.append(float(np.sum(detail ** 2)))
        current = approx
    return tuple(energies)  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Additional combined statistics
# ══════════════════════════════════════════════════════════════════════════════


def _combined_stats(combined: np.ndarray) -> tuple[float, float, float]:
    """Compute peak-to-peak, shape factor, and impulse factor on combined signal."""
    p2p = float(np.max(combined) - np.min(combined))
    rms = float(np.sqrt(np.mean(combined ** 2)))
    mean_abs = float(np.mean(np.abs(combined))) + 1e-12
    peak = float(np.max(np.abs(combined)))
    shape = rms / mean_abs
    impulse = peak / mean_abs
    return p2p, shape, impulse


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — Antigravity-sensitive features (49–54)
# ══════════════════════════════════════════════════════════════════════════════


def compute_gli(z_signal: np.ndarray, fs: int) -> float:
    """
    Gravity Load Index — ratio of DC (mean) component of vertical-axis
    accelerometer to the expected 1g baseline (9.81 m/s²).

    GLI ≈ 1.0 under normal gravity; GLI → 0 indicates gravitational load
    cancellation characteristic of antigravity operating regimes.

    Args:
        z_signal: Vertical-axis (Z) accelerometer time series in m/s².
        fs: Sampling rate in Hz.

    Returns:
        GLI scalar ∈ [0, ∞).
    """
    dc_offset = float(np.abs(np.mean(z_signal)))
    return dc_offset / GRAVITY_MS2


def compute_ser(
    sig: np.ndarray, shaft_rpm: float, fs: int
) -> float:
    """
    Sub-synchronous Energy Ratio — spectral energy in the 0.1×–0.4× shaft
    frequency band normalised by total broadband energy.

    Elevated SER (> 0.15 baseline) is a hallmark of partial gravitational load
    relief; under normal gravity SER remains low due to 1× harmonic dominance.

    Args:
        sig: Combined vibration signal (any axis or RMS combination).
        shaft_rpm: Shaft rotational speed in RPM.
        fs: Sampling rate in Hz.

    Returns:
        SER scalar ∈ [0, 1].
    """
    fr = _shaft_hz(shaft_rpm)
    freqs, psd = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    sub_mask = (freqs >= SUB_SYNC_LOW * fr) & (freqs <= SUB_SYNC_HIGH * fr)
    total_energy = np.sum(psd) + 1e-12
    sub_energy = np.sum(psd[sub_mask])
    return float(sub_energy / total_energy)


def compute_hcc(measured_1x: float) -> float:
    """
    Harmonic Cancellation Coefficient — degree to which the 1× shaft harmonic
    has been suppressed relative to the expected value under normal gravity.

    HCC = (expected_1x - measured_1x) / expected_1x

    HCC → 0 under normal loading; HCC → 1 indicates near-complete 1× suppression,
    consistent with gravitational force cancellation on the rotor.

    Args:
        measured_1x: Measured 1× harmonic amplitude (m/s²).

    Returns:
        HCC scalar, typically ∈ [0, 1] but can be negative if 1× is amplified.
    """
    return float((EXPECTED_1X_AMP - measured_1x) / (EXPECTED_1X_AMP + 1e-12))


def compute_cpc(
    x_signal: np.ndarray,
    z_signal: np.ndarray,
    shaft_rpm: float,
    fs: int,
) -> float:
    """
    Cross-axis Phase Coherence — magnitude-squared coherence between the X and
    Z accelerometer axes in the 0.5×–2× shaft frequency band.

    Under normal gravity the orbital path of the shaft is elliptical and X–Z
    coherence in this band is high (> 0.7). Antigravity regimes alter the
    orbital shape, reducing coherence asymmetrically toward 0.3–0.5.

    Args:
        x_signal: X-axis accelerometer time series.
        z_signal: Z-axis accelerometer time series.
        shaft_rpm: Shaft rotational speed in RPM.
        fs: Sampling rate in Hz.

    Returns:
        Mean MSC scalar ∈ [0, 1] within the 0.5×–2× band.
    """
    fr = _shaft_hz(shaft_rpm)
    nperseg = min(512, len(x_signal))
    freqs, coh = coherence(x_signal, z_signal, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= 0.5 * fr) & (freqs <= 2.0 * fr)
    if not np.any(band_mask):
        return float(np.mean(coh))
    return float(np.mean(coh[band_mask]))


def compute_blza(bpfo_amplitude: float, bpfi_amplitude: float) -> float:
    """
    Bearing Load Zone Asymmetry — ratio of BPFO to BPFI harmonic amplitude.

    Under normal gravity the loaded arc concentrates load on the outer race,
    keeping BPFO/BPFI ≈ 1.2–1.8. Gravitational load reduction shifts the load
    zone, causing BLZA to deviate significantly from this baseline.

    Args:
        bpfo_amplitude: Peak BPFO harmonic amplitude (m/s²).
        bpfi_amplitude: Peak BPFI harmonic amplitude (m/s²).

    Returns:
        BLZA ratio ∈ [0, ∞).
    """
    return float(bpfo_amplitude / (bpfi_amplitude + 1e-12))


def compute_gpi(
    z_signal: np.ndarray,
    shaft_rpm: float,
    fs: int,
) -> float:
    """
    Gyroscopic Precession Index — ratio of the sub-harmonic precession frequency
    (identified from the envelope spectrum) to shaft frequency.

    Under normal gravity GPI ≈ integer values (synchronous components).
    In antigravity-adjacent regimes GPI → PRECESSION_RATIO (default 0.43),
    indicating non-integer gyroscopic coupling.

    Args:
        z_signal: Vertical-axis accelerometer time series.
        shaft_rpm: Shaft rotational speed in RPM.
        fs: Sampling rate in Hz.

    Returns:
        GPI ratio (dimensionless).
    """
    fr = _shaft_hz(shaft_rpm)
    analytic = sp_signal.hilbert(z_signal)
    envelope = np.abs(analytic)
    # Envelope spectrum
    env_fft = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
    env_freqs = np.fft.rfftfreq(len(envelope), d=1.0 / fs)
    # Search for dominant peak in sub-harmonic range: 0.1× – 0.9× shaft freq
    sub_mask = (env_freqs >= 0.1 * fr) & (env_freqs <= 0.9 * fr)
    if not np.any(sub_mask):
        return 0.0
    peak_freq = env_freqs[sub_mask][np.argmax(env_fft[sub_mask])]
    return float(peak_freq / (fr + 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — extract_features
# ══════════════════════════════════════════════════════════════════════════════


class FeatureVector(NamedTuple):
    """Container for the full 54-element feature vector plus metadata."""

    features: np.ndarray        # shape (54,)
    names: list[str]            # FEATURE_NAMES


def extract_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    shaft_rpm: float = 1500.0,
    fs: int = SAMPLE_RATE,
) -> FeatureVector:
    """
    Extract all 54 features from a 3-axis vibration window.

    Args:
        x: X-axis accelerometer samples (m/s²), shape (N,).
        y: Y-axis accelerometer samples (m/s²), shape (N,).
        z: Z-axis accelerometer samples (m/s²), shape (N,).
        shaft_rpm: Current shaft speed in RPM.
        fs: Sensor sampling rate in Hz.

    Returns:
        FeatureVector namedtuple with .features (ndarray, shape 54) and .names.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    combined = (x + y + z) / np.sqrt(3)  # RMS combination

    # ── 1. Time-domain per axis ──────────────────────────────────────────────
    x_td = _time_features(x)
    y_td = _time_features(y)
    z_td = _time_features(z)

    # ── 2. Frequency-domain per axis ─────────────────────────────────────────
    x_fd = _freq_features(x, fs)
    y_fd = _freq_features(y, fs)
    z_fd = _freq_features(z, fs)

    # ── 3. Bearing harmonics ─────────────────────────────────────────────────
    bpfi, bpfo, bsf, ftf = _bearing_harmonics(combined, shaft_rpm, fs)

    # ── 4. Shaft harmonics ───────────────────────────────────────────────────
    h1, h2, h3, h4 = _shaft_harmonics(combined, shaft_rpm, fs)

    # ── 5. Envelope analysis ─────────────────────────────────────────────────
    env_rms, env_kurt = _envelope_features(combined)

    # ── 6. Cross-axis coherence ──────────────────────────────────────────────
    coh_xy, coh_xz, coh_yz = _coherence_features(x, y, z, fs)

    # ── 7. Band energy ratios ────────────────────────────────────────────────
    e_low, e_mid, e_high, e_vhigh = _band_energy_ratios(combined, fs)

    # ── 8. Wavelet energy ────────────────────────────────────────────────────
    w1, w2, w3, w4 = _wavelet_energy(combined)

    # ── 9. Combined stats ────────────────────────────────────────────────────
    p2p, shape_f, impulse_f = _combined_stats(combined)

    # ── 10. Antigravity features (49–54) ─────────────────────────────────────
    gli  = compute_gli(z, fs)
    ser  = compute_ser(combined, shaft_rpm, fs)
    hcc  = compute_hcc(h1)
    cpc  = compute_cpc(x, z, shaft_rpm, fs)
    blza = compute_blza(bpfo, bpfi)
    gpi  = compute_gpi(z, shaft_rpm, fs)

    # ── Assemble feature vector ───────────────────────────────────────────────
    vec = np.array(
        [
            # Time-domain (15)
            *x_td, *y_td, *z_td,
            # Frequency-domain (9)
            *x_fd, *y_fd, *z_fd,
            # Bearing (4)
            bpfi, bpfo, bsf, ftf,
            # Shaft harmonics (4)
            h1, h2, h3, h4,
            # Envelope (2)
            env_rms, env_kurt,
            # Cross-axis coherence (3)
            coh_xy, coh_xz, coh_yz,
            # Band energy (4)
            e_low, e_mid, e_high, e_vhigh,
            # Wavelet (4)
            w1, w2, w3, w4,
            # Combined stats (3)
            p2p, shape_f, impulse_f,
            # Antigravity (6)
            gli, ser, hcc, cpc, blza, gpi,
        ],
        dtype=np.float64,
    )
    assert vec.shape == (54,), f"Feature vector shape mismatch: {vec.shape}"
    return FeatureVector(features=vec, names=FEATURE_NAMES)
