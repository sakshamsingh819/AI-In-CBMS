# Condition Monitoring System

> AI-powered vibration analysis and fault classification for rotating machinery using STWIN.box MEMS sensors and a cGAN-Hybrid classifier.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=flat-square)](https://python.org)
[![React 18](https://img.shields.io/badge/react-18-blue?style=flat-square)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/fastapi-0.111-green?style=flat-square)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

---

## Features

- **7-class fault classifier**: normal, bpfi, bpfo, bsf, unbalance, misalignment, looseness + antigravity
- **54-feature extraction pipeline** from 3-axis IIS3DWB MEMS vibration signals
- **Three inference models**: Random Forest, LSTM, and cGAN-Hybrid discriminator
- **Real-time React 18 dashboard** with glassmorphism UI and micro-animations
- **Antigravity detection** (experimental) — see below

---

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp ../.env.example .env          # configure your environment
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

### Docker (full stack)

```bash
cp .env.example .env
docker-compose up --build
```

---

## Project Structure

```
TDPCL/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── schemas.py                 # Pydantic models (54-feature vector)
│   ├── routers/
│   │   ├── analysis.py            # POST /api/analysis/* endpoints
│   │   └── models.py              # GET /api/models
│   ├── services/
│   │   ├── feature_extractor.py   # 54-feature pipeline
│   │   └── inference.py           # RF + LSTM inference
│   ├── scripts/
│   │   ├── generate_antigravity_data.py
│   │   ├── retrain_rf.py
│   │   ├── retrain_lstm.py
│   │   └── retrain_cgan.py
│   ├── data/                      # Training + synthetic data
│   └── tests/                     # pytest test suites
├── frontend/
│   └── src/
│       ├── components/            # React components
│       ├── store/useStore.js      # Zustand global state
│       └── api/client.js          # Axios API client
├── docs/
│   └── ANTIGRAVITY_THEORY.md     # Rotordynamic theory
├── CONTEXT.md                    # Living session document
├── CHANGELOG.md
└── docker-compose.yml
```

---

## Running Tests

```bash
cd backend
pytest tests/test_feature_extractor.py -v    # Unit tests for 54 features
pytest tests/test_antigravity_api.py -v      # Integration tests (all endpoints)
pytest tests/ -v                             # All tests
```

---

## Antigravity Detection (Experimental)

### What is it?

The Antigravity detection layer identifies operating regimes in rotating machinery where the **net effective gravitational load on the bearing drops below 25% of 1g**. This can occur in:
- Magnetic bearing systems under active preload control
- Vertical-axis turbines at high centrifugal-to-gravity ratio
- Microgravity test environments (spacecraft, parabolic flight)
- Deliberate counterweight-reduced configurations

This creates a distinct vibration signature: reduced 1× harmonic amplitude, elevated sub-synchronous content, altered bearing load zone distribution, and non-standard envelope spectrum behaviour.

### How to Enable

Add to your `.env` file:

```env
ANTIGRAVITY_ENABLED=true
```

This switches all three inference models to the v2 variants (7-class, 54-feature) and enables:
- The new `POST /api/analysis/antigravity` endpoint
- The `include_antigravity_features` parameter on `/api/analysis/features`
- Per-model `antigravity_f1` reporting on `GET /api/models`
- The AntigravityPanel and arc-gauge probability card in the dashboard

### New API Endpoint

```
POST /api/analysis/antigravity
```

Request:
```json
{
  "session_id": "sess_001",
  "window_index": 42,
  "shaft_rpm": 1500,
  "signal_x": [...1024 floats...],
  "signal_y": [...1024 floats...],
  "signal_z": [...1024 floats...]
}
```

Response:
```json
{
  "gli": 0.12,
  "ser": 0.38,
  "hcc": 0.71,
  "cpc": 0.45,
  "blza": 1.85,
  "gpi": 0.43,
  "antigravity_probability": 0.87,
  "regime": "near_zero_gravity"
}
```

### Operating Regimes

| Regime | Probability | Meaning |
|--------|-------------|---------|
| `normal_gravity` | < 30% | Standard operating conditions |
| `reduced_gravity` | 30–85% | Partial load cancellation — monitor |
| `near_zero_gravity` | ≥ 85% | Near-complete gravity cancellation — alert |

### Synthetic Data Limitations

The antigravity class is trained on **2,000 synthetically generated windows** using a modified Jeffcott rotor model (α ∈ [0.0, 0.25]). The classifier has not been validated on real reduced-gravity hardware.

**Do not use for safety-critical decisions without hardware validation.**

See `docs/ANTIGRAVITY_THEORY.md` for the full theoretical framework and recommended validation procedure.

### Generating Synthetic Data & Retraining

```bash
cd backend

# Step 1: Generate 2000 synthetic antigravity windows
python scripts/generate_antigravity_data.py

# Step 2: Retrain Random Forest (7-class)
python scripts/retrain_rf.py

# Step 3: Fine-tune LSTM (7-class output)
python scripts/retrain_lstm.py

# Step 4: Review cGAN-Hybrid stub (requires real data for full training)
python scripts/retrain_cgan.py
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTIGRAVITY_ENABLED` | `false` | Enable 7-class models and new endpoints |
| `SHAFT_RPM_DEFAULT` | `1500` | Default shaft speed for feature extraction |
| `PRECESSION_RATIO` | `0.43` | GPI precession frequency ratio |
| `EXPECTED_1X_AMPLITUDE` | `1.0` | Nominal 1× harmonic amplitude (m/s²) |
| `SENSOR_SAMPLE_RATE` | `26667` | IIS3DWB sampling rate (Hz) |
| `SYNTHETIC_SNR_DB` | `25` | SNR for synthetic data generation |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `cd backend && pytest tests/ -v`
4. Update CONTEXT.md with your changes
5. Submit a pull request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with FastAPI + React 18 + scikit-learn + TensorFlow | Session: e7c7d1a0-1f0e-4450-97b5-421609752b85*
