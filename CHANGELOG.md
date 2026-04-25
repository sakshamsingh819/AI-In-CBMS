# CHANGELOG

All notable changes to this project are documented here.  
Format: [Semantic Versioning](https://semver.org/).

---

## [1.1.0-antigravity] — 2026-04-25

### ⚛ Antigravity Detection (Experimental)

This release adds a first-class Antigravity detection layer to the Condition
Monitoring System. All changes are behind the `ANTIGRAVITY_ENABLED` feature flag
and are **non-breaking** for existing 6-class users.

#### Breaking Changes
- **None.** The 6-class models and API remain unchanged when `ANTIGRAVITY_ENABLED=false`.

#### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analysis/antigravity` | Dedicated antigravity regime analysis (GLI, SER, HCC, CPC, BLZA, GPI + probability) |

#### Modified Endpoints

| Method | Path | Change |
|--------|------|--------|
| `POST` | `/api/analysis/features` | Added `include_antigravity_features: bool` query param |
| `POST` | `/api/analysis/classify` | Added `antigravity_probability: float | null` to response |
| `GET`  | `/api/models` | Added `antigravity_capable: bool` and `antigravity_f1: float | null` per model |

#### New Features
- **54-feature vector** (extended from 48): adds GLI, SER, HCC, CPC, BLZA, GPI
- **7th fault class**: `antigravity` added to `FaultClass` Literal type in `schemas.py`
- **ANTIGRAVITY_ENABLED feature flag** in `.env` (default: `false`)
- **3 operating regimes**: `normal_gravity`, `reduced_gravity`, `near_zero_gravity`
- **AntigravityPanel** React component: 6 horizontal gauge bars with ±2σ reference bands
- **AntigravityProbabilityCard**: SVG arc gauge 0–100% with threshold lines at 30%/60%/85%
- **FaultCard amber badge**: "⚠ Reduced Gravity Regime Detected" when probability > 0.30
- **ModelSelector** tooltip: shows `antigravity_f1` when ANTIGRAVITY_ENABLED=true
- **SeverityGauge**: deep purple (#4B0082) Antigravity entry in legend
- **Zustand store**: `antigravityFeatures`, `antigravityProbability`, `antigravityRegime` state slices

#### New ML Scripts

| Script | Purpose |
|--------|---------|
| `backend/scripts/generate_antigravity_data.py` | Synthetic Jeffcott rotor data (2000 windows, α ∈ [0,0.25]) |
| `backend/scripts/retrain_rf.py` | 7-class RF (500 estimators, class_weight=balanced) |
| `backend/scripts/retrain_lstm.py` | LSTM fine-tuning (frozen encoder, 30 epochs) |
| `backend/scripts/retrain_cgan.py` | cGAN-Hybrid discriminator extension (detailed stub) |

#### New Documentation

| File | Description |
|------|-------------|
| `docs/ANTIGRAVITY_THEORY.md` | Rotordynamic theory, feature derivations, validation procedure |
| `CONTEXT.md` | Living session context document |
| `CHANGELOG.md` | This file |

#### Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `scipy` | ≥1.13.0 | Signal processing (coherence, Welch PSD, Hilbert transform) |
| `joblib` | ≥1.4.0 | Model serialisation |
| `loguru` | ≥0.7.0 | Structured logging |
| `pytest-asyncio` | ≥0.23.0 | Async test support |
| `httpx` | ≥0.27.0 | Test HTTP client |
| `zustand` | ^4.5.4 | Frontend state management |
| `recharts` | ^2.12.7 | Chart components |
| `@radix-ui/react-tooltip` | ^1.1.2 | Tooltip primitives |

#### Known Limitations
- Antigravity class trained exclusively on synthetic data. Real-world F1 may differ from the 0.83 synthetic benchmark.
- GLI requires vertical Z-axis sensor alignment. Tilt errors ≥15° cause false negatives.
- GPI precession ratio (0.43×) is theoretically derived and requires hardware calibration.
- cGAN-Hybrid retraining is a stub — full training loop implementation is pending hardware data.

---

## [1.0.0] — (initial build)

- Initial 6-class fault classifier: normal, bpfi, bpfo, bsf, unbalance, misalignment, looseness
- 48-feature extraction pipeline
- FastAPI backend + React 18 dashboard
- Random Forest + LSTM inference models
- STWIN.box IIS3DWB sensor integration
