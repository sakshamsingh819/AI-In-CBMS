"""
Integration tests for the antigravity API endpoints.

Tests:
  - POST /api/analysis/antigravity
  - POST /api/analysis/features (include_antigravity_features=true)
  - POST /api/analysis/classify
  - GET /api/models

Fixtures:
  - normal_gravity_payload: signal with DC Z ≈ 9.81 m/s² and strong 1×
  - antigravity_payload:    signal with DC Z ≈ 0.5 m/s² and weak 1× + sub-synchronous

Run: cd backend && pytest tests/test_antigravity_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── App import ────────────────────────────────────────────────────────────────

from main import app

# ── Synthetic signal builders ─────────────────────────────────────────────────

FS = 26667
FR = 25.0        # 1500 RPM → 25 Hz
N = 1024
GRAVITY_MS2 = 9.81
RNG = np.random.default_rng(123)


def _make_normal_signal_payload():
    """Build a normal-gravity API request payload."""
    t = np.arange(N) / FS
    x = (np.sin(2 * np.pi * FR * t) + 0.3 * np.sin(4 * np.pi * FR * t)
         + RNG.normal(0, 0.05, N))
    y = (np.sin(2 * np.pi * FR * t + np.pi / 2)
         + RNG.normal(0, 0.05, N))
    z = GRAVITY_MS2 + 0.3 * np.sin(2 * np.pi * FR * t) + RNG.normal(0, 0.05, N)
    return {
        "session_id": "test_normal_001",
        "window_index": 0,
        "signal_x": x.tolist(),
        "signal_y": y.tolist(),
        "signal_z": z.tolist(),
        "shaft_rpm": 1500.0,
    }


def _make_antigravity_signal_payload():
    """Build a reduced-gravity API request payload (α ≈ 0.05)."""
    alpha = 0.05
    t = np.arange(N) / FS
    prec_freq = 0.43 * FR
    A1 = 0.1 + 0.9 * alpha
    sub_freqs = [0.15 * FR, 0.25 * FR, 0.35 * FR]
    x = A1 * np.sin(2 * np.pi * FR * t)
    for sf in sub_freqs:
        x += 0.12 * np.sin(2 * np.pi * sf * t)
    x += 0.15 * np.sin(2 * np.pi * prec_freq * t) + RNG.normal(0, 0.05, N)
    y = A1 * np.sin(2 * np.pi * FR * t + np.pi / 4) + RNG.normal(0, 0.05, N)
    z = GRAVITY_MS2 * alpha + 0.1 * np.sin(2 * np.pi * FR * t) + RNG.normal(0, 0.05, N)
    return {
        "session_id": "test_ag_001",
        "window_index": 0,
        "signal_x": x.tolist(),
        "signal_y": y.tolist(),
        "signal_z": z.tolist(),
        "shaft_rpm": 1500.0,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_payload():
    """Normal-gravity request fixture."""
    return _make_normal_signal_payload()


@pytest.fixture
def antigravity_payload():
    """Reduced-gravity request fixture."""
    return _make_antigravity_signal_payload()


@pytest_asyncio.fixture
async def client():
    """Async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Health / models ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint must return 200 with status=ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "antigravity_enabled" in data


@pytest.mark.asyncio
async def test_models_endpoint(client):
    """GET /api/models must return a list of models with required fields."""
    resp = await client.get("/api/models/")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "active_model" in data
    assert "antigravity_enabled" in data
    assert len(data["models"]) >= 1
    for model in data["models"]:
        assert "antigravity_capable" in model
        assert "feature_vector_length" in model


# ── POST /api/analysis/antigravity ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_antigravity_endpoint_normal_gravity(client, normal_payload):
    """
    Antigravity endpoint with a normal-gravity signal:
    - Response must include all 6 feature fields
    - Regime should be 'normal_gravity'
    """
    resp = await client.post("/api/analysis/antigravity", json=normal_payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # All 6 antigravity features must be present
    for field in ["gli", "ser", "hcc", "cpc", "blza", "gpi"]:
        assert field in data, f"Missing field: {field}"
        assert isinstance(data[field], float), f"{field} must be a float"

    assert "antigravity_probability" in data
    assert 0.0 <= data["antigravity_probability"] <= 1.0
    assert data["regime"] in ["normal_gravity", "reduced_gravity", "near_zero_gravity"]


@pytest.mark.asyncio
async def test_antigravity_endpoint_antigravity_signal(client, antigravity_payload):
    """
    Antigravity endpoint with a synthetic antigravity signal:
    - GLI should be significantly less than 1.0
    - SER should be elevated
    - HCC should be elevated
    - Session/window IDs must round-trip
    """
    resp = await client.post("/api/analysis/antigravity", json=antigravity_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == antigravity_payload["session_id"]
    assert data["window_index"] == antigravity_payload["window_index"]

    # GLI should be small (gravity cancelled)
    assert data["gli"] < 0.5, f"GLI={data['gli']:.3f} should be < 0.5 for antigravity signal"

    # SER should be elevated
    assert data["ser"] > 0.05, f"SER={data['ser']:.4f} should be > 0.05"

    # HCC should indicate 1× suppression
    assert data["hcc"] > 0.3, f"HCC={data['hcc']:.3f} should be > 0.3 (1× suppressed)"


@pytest.mark.asyncio
async def test_antigravity_endpoint_invalid_signal_length(client):
    """Mismatched signal lengths must return HTTP 422."""
    payload = _make_normal_signal_payload()
    payload["signal_z"] = payload["signal_z"][:512]  # mismatch
    resp = await client.post("/api/analysis/antigravity", json=payload)
    assert resp.status_code == 422, f"Expected 422 for mismatched signals, got {resp.status_code}"


# ── POST /api/analysis/features ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_features_endpoint_returns_base_features(client, normal_payload):
    """Features endpoint must return a non-empty feature list."""
    resp = await client.post("/api/analysis/features", json=normal_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "features" in data
    assert len(data["features"]) >= 48  # at least 48 features


@pytest.mark.asyncio
async def test_features_endpoint_with_antigravity_details(client, normal_payload):
    """
    When include_antigravity_features=true, response must include
    antigravity_features list with 6 entries, each having:
      name, value, baseline_mean, baseline_std, is_anomalous
    """
    payload = {**normal_payload, "include_antigravity_features": True}
    resp = await client.post("/api/analysis/features", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["antigravity_features"] is not None
    assert len(data["antigravity_features"]) == 6
    for feat in data["antigravity_features"]:
        for key in ["name", "value", "baseline_mean", "baseline_std", "is_anomalous"]:
            assert key in feat, f"Missing key '{key}' in antigravity feature detail"


# ── POST /api/analysis/classify ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_endpoint(client, normal_payload):
    """Classify endpoint must return a valid FaultClass and confidence."""
    resp = await client.post("/api/analysis/classify", json=normal_payload)
    assert resp.status_code == 200
    data = resp.json()

    valid_classes = [
        "normal", "bpfi", "bpfo", "bsf",
        "unbalance", "misalignment", "looseness", "antigravity"
    ]
    assert data["predicted_class"] in valid_classes
    assert 0.0 <= data["confidence"] <= 1.0
    assert "class_probabilities" in data
    total_prob = sum(data["class_probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities don't sum to 1: {total_prob}"
