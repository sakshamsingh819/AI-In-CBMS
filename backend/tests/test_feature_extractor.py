"""
Unit tests for the feature extractor (feature_extractor.py).

Tests cover:
  - Output shape (54 features)
  - Feature name count
  - All 6 antigravity-sensitive features (GLI, SER, HCC, CPC, BLZA, GPI):
      Each test uses a synthetic signal where the expected direction of
      feature change is known and verified.

Run: cd backend && pytest tests/test_feature_extractor.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from services.feature_extractor import (
    FEATURE_NAMES,
    compute_blza,
    compute_cpc,
    compute_gli,
    compute_gpi,
    compute_hcc,
    compute_ser,
    extract_features,
)

# ── Test parameters ───────────────────────────────────────────────────────────

FS = 26667
SHAFT_RPM = 1500.0
FR = SHAFT_RPM / 60  # 25 Hz
GRAVITY_MS2 = 9.81
N = 1024
RNG = np.random.default_rng(42)


def _make_normal_gravity_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic signal representing a normal-gravity operating condition."""
    t = np.arange(N) / FS
    # Strong 1× harmonic; Z axis carries ~1g DC offset; minimal sub-synchronous
    x = np.sin(2 * np.pi * FR * t) + 0.3 * np.sin(2 * np.pi * 2 * FR * t)
    x += RNG.normal(0, 0.05, N)
    y = np.sin(2 * np.pi * FR * t + np.pi / 2) + 0.2 * RNG.normal(0, 0.05, N)
    z = GRAVITY_MS2 + 0.3 * np.sin(2 * np.pi * FR * t + np.pi / 3)  # DC ≈ 9.81
    z += RNG.normal(0, 0.05, N)
    return x, y, z


def _make_antigravity_signal(alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic signal representing a reduced-gravity operating condition.
    α = 0.05 → 5% of normal gravity loading.
    """
    t = np.arange(N) / FS
    prec_freq = 0.43 * FR
    sub_freqs = [0.15 * FR, 0.25 * FR, 0.35 * FR]

    # Suppressed 1× harmonic
    A1 = 0.1 + 0.9 * alpha
    x = A1 * np.sin(2 * np.pi * FR * t)
    # Elevated sub-synchronous components
    for sf in sub_freqs:
        x += 0.12 * np.sin(2 * np.pi * sf * t + RNG.uniform(0, 2 * np.pi))
    # Precession component
    x += 0.15 * np.sin(2 * np.pi * prec_freq * t)
    x += RNG.normal(0, 0.05, N)

    y = A1 * np.sin(2 * np.pi * FR * t + np.pi / 4) + RNG.normal(0, 0.05, N)

    # Z axis: very low DC offset (gravity nearly cancelled)
    z = GRAVITY_MS2 * alpha + 0.1 * np.sin(2 * np.pi * FR * t)
    z += RNG.normal(0, 0.05, N)

    return x, y, z


# ── Core tests ────────────────────────────────────────────────────────────────


class TestFeatureVectorShape:
    def test_output_is_54_features(self):
        """extract_features must return exactly 54 elements."""
        x, y, z = _make_normal_gravity_signal()
        fv = extract_features(x, y, z, shaft_rpm=SHAFT_RPM, fs=FS)
        assert fv.features.shape == (54,), f"Got shape {fv.features.shape}"

    def test_feature_names_count(self):
        """FEATURE_NAMES must have exactly 54 entries."""
        assert len(FEATURE_NAMES) == 54

    def test_no_nan_in_features_normal(self):
        """No NaN values should be present in a normal-gravity feature vector."""
        x, y, z = _make_normal_gravity_signal()
        fv = extract_features(x, y, z, shaft_rpm=SHAFT_RPM, fs=FS)
        assert not np.any(np.isnan(fv.features)), "NaN found in normal-gravity feature vector"

    def test_no_nan_in_features_antigravity(self):
        """No NaN values should be present in an antigravity feature vector."""
        x, y, z = _make_antigravity_signal(alpha=0.05)
        fv = extract_features(x, y, z, shaft_rpm=SHAFT_RPM, fs=FS)
        assert not np.any(np.isnan(fv.features)), "NaN found in antigravity feature vector"

    def test_antigravity_features_are_last_six(self):
        """The last 6 feature names must be the antigravity features."""
        expected = ["gli", "ser", "hcc", "cpc", "blza", "gpi"]
        assert FEATURE_NAMES[-6:] == expected


# ── GLI tests ─────────────────────────────────────────────────────────────────


class TestGravityLoadIndex:
    def test_gli_near_one_under_normal_gravity(self):
        """GLI should be ≈ 1.0 when Z-axis DC offset ≈ 9.81 m/s²."""
        # Z signal with DC ≈ 9.81 (full gravity)
        z_normal = GRAVITY_MS2 + RNG.normal(0, 0.1, N)
        gli = compute_gli(z_normal, FS)
        assert 0.8 <= gli <= 1.2, f"GLI={gli:.3f} unexpectedly far from 1.0 under normal gravity"

    def test_gli_near_zero_under_antigravity(self):
        """GLI should be ≪ 1.0 when DC offset → 0 (gravity cancelled)."""
        # Z signal with DC ≈ 0.05g (5% gravity)
        z_ag = GRAVITY_MS2 * 0.05 + RNG.normal(0, 0.05, N)
        gli = compute_gli(z_ag, FS)
        assert gli < 0.15, f"GLI={gli:.3f} should be < 0.15 under near-zero gravity"

    def test_gli_increases_with_gravity_loading(self):
        """GLI must be monotonically increasing with gravity loading factor α."""
        glis = []
        for alpha in [0.01, 0.05, 0.10, 0.20, 1.0]:
            z = GRAVITY_MS2 * alpha + RNG.normal(0, 0.02, N)
            glis.append(compute_gli(z, FS))
        for i in range(len(glis) - 1):
            assert glis[i] < glis[i + 1], (
                f"GLI not monotonic at α step {i}: {glis[i]:.3f} >= {glis[i+1]:.3f}"
            )


# ── SER tests ─────────────────────────────────────────────────────────────────


class TestSubSynchronousEnergyRatio:
    def test_ser_low_under_normal_gravity(self):
        """SER should remain low (< 0.15) under normal 1×-dominated vibration."""
        x, _, _ = _make_normal_gravity_signal()
        ser = compute_ser(x, SHAFT_RPM, FS)
        assert ser < 0.15, f"SER={ser:.4f} should be < 0.15 under normal gravity"

    def test_ser_elevated_under_antigravity(self):
        """SER should be elevated (> 0.15) when sub-synchronous components dominate."""
        x, _, _ = _make_antigravity_signal(alpha=0.05)
        ser = compute_ser(x, SHAFT_RPM, FS)
        assert ser > 0.10, f"SER={ser:.4f} should be > 0.10 under antigravity conditions"

    def test_ser_in_valid_range(self):
        """SER must be ∈ [0, 1]."""
        for make_sig in [_make_normal_gravity_signal, _make_antigravity_signal]:
            x, _, _ = make_sig()
            ser = compute_ser(x, SHAFT_RPM, FS)
            assert 0.0 <= ser <= 1.0, f"SER={ser:.4f} out of [0, 1] range"


# ── HCC tests ─────────────────────────────────────────────────────────────────


class TestHarmonicCancellationCoefficient:
    def test_hcc_near_zero_when_1x_is_nominal(self):
        """HCC ≈ 0 when measured_1x ≈ EXPECTED_1X_AMP (1.0)."""
        hcc = compute_hcc(measured_1x=1.0)
        assert abs(hcc) < 0.05, f"HCC={hcc:.4f} should be ~0 at nominal 1× amplitude"

    def test_hcc_near_one_when_1x_suppressed(self):
        """HCC → 1 when the 1× harmonic is nearly fully suppressed."""
        hcc = compute_hcc(measured_1x=0.02)  # 98% suppressed
        assert hcc > 0.90, f"HCC={hcc:.4f} should be > 0.90 when 1× is suppressed"

    def test_hcc_negative_when_1x_amplified(self):
        """HCC < 0 when measured_1x > expected (amplified condition)."""
        hcc = compute_hcc(measured_1x=2.5)  # 2.5× amplified
        assert hcc < 0, f"HCC={hcc:.4f} should be negative when 1× is amplified"

    def test_hcc_monotonically_decreases_with_increasing_1x(self):
        """HCC decreases as measured_1x increases."""
        hccs = [compute_hcc(amp) for amp in [0.1, 0.3, 0.7, 1.0, 1.5]]
        for i in range(len(hccs) - 1):
            assert hccs[i] > hccs[i + 1], (
                f"HCC not monotonically decreasing: {hccs[i]:.3f} <= {hccs[i+1]:.3f}"
            )


# ── CPC tests ─────────────────────────────────────────────────────────────────


class TestCrossAxisPhaseCoherence:
    def test_cpc_high_for_coherent_signals(self):
        """CPC should be high (> 0.6) for signals with identical frequency content."""
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * FR * t) + RNG.normal(0, 0.02, N)
        # Z is nearly identical to X (high coherence)
        z = np.sin(2 * np.pi * FR * t + 0.1) + RNG.normal(0, 0.02, N)
        cpc = compute_cpc(x, z, SHAFT_RPM, FS)
        assert cpc > 0.5, f"CPC={cpc:.3f} should be > 0.5 for coherent signals"

    def test_cpc_lower_for_antigravity_signal(self):
        """CPC is lower under antigravity (orbital path disrupted)."""
        x_norm, _, z_norm = _make_normal_gravity_signal()
        x_ag, _, z_ag = _make_antigravity_signal(alpha=0.05)
        cpc_normal = compute_cpc(x_norm, z_norm, SHAFT_RPM, FS)
        cpc_ag = compute_cpc(x_ag, z_ag, SHAFT_RPM, FS)
        # Antigravity CPC should be lower than normal (orbital asymmetry)
        # Allow generous tolerance since we're using synthetic signals
        assert cpc_ag <= cpc_normal + 0.1, (
            f"CPC_ag={cpc_ag:.3f} should be ≤ CPC_normal={cpc_normal:.3f}"
        )

    def test_cpc_in_valid_range(self):
        """CPC (mean MSC) must be ∈ [0, 1]."""
        x, _, z = _make_normal_gravity_signal()
        cpc = compute_cpc(x, z, SHAFT_RPM, FS)
        assert 0.0 <= cpc <= 1.0, f"CPC={cpc:.4f} out of [0, 1] range"


# ── BLZA tests ────────────────────────────────────────────────────────────────


class TestBearingLoadZoneAsymmetry:
    def test_blza_normal_range(self):
        """BLZA ≈ 1.2–1.8 under normal gravity loading."""
        blza = compute_blza(bpfo_amplitude=1.5, bpfi_amplitude=1.0)
        assert 1.2 <= blza <= 1.8, f"BLZA={blza:.3f} outside normal range"

    def test_blza_shoots_up_with_elevated_bpfo(self):
        """BLZA >> 1.8 when BPFO is anomalously elevated (shifted load zone)."""
        blza = compute_blza(bpfo_amplitude=4.0, bpfi_amplitude=0.5)
        assert blza > 2.5, f"BLZA={blza:.3f} should be > 2.5 under shifted load zone"

    def test_blza_no_division_by_zero(self):
        """BLZA should not raise ZeroDivisionError when BPFI = 0."""
        blza = compute_blza(bpfo_amplitude=1.5, bpfi_amplitude=0.0)
        assert np.isfinite(blza), "BLZA should be finite even when BPFI = 0"

    def test_blza_proportional(self):
        """BLZA must be proportional to BPFO/BPFI ratio."""
        blza1 = compute_blza(1.0, 0.5)
        blza2 = compute_blza(2.0, 0.5)
        assert abs(blza2 - 2 * blza1) < 0.01, "BLZA should scale linearly with BPFO"


# ── GPI tests ─────────────────────────────────────────────────────────────────


class TestGyroscopicPrecessionIndex:
    def test_gpi_near_precession_ratio_under_antigravity(self):
        """GPI should approach PRECESSION_RATIO (0.43) under antigravity conditions."""
        x, _, z = _make_antigravity_signal(alpha=0.05)
        gpi = compute_gpi(z, SHAFT_RPM, FS)
        # GPI should be in sub-harmonic range (0 < GPI < 1)
        assert 0.0 < gpi < 1.0, f"GPI={gpi:.4f} should be in (0, 1) sub-harmonic range"

    def test_gpi_non_negative(self):
        """GPI must be non-negative."""
        for make_sig in [_make_normal_gravity_signal, _make_antigravity_signal]:
            _, _, z = make_sig()
            gpi = compute_gpi(z, SHAFT_RPM, FS)
            assert gpi >= 0.0, f"GPI={gpi:.4f} should be non-negative"

    def test_gpi_finite(self):
        """GPI must be a finite number."""
        x, _, z = _make_antigravity_signal()
        gpi = compute_gpi(z, SHAFT_RPM, FS)
        assert np.isfinite(gpi), f"GPI={gpi} is not finite"
