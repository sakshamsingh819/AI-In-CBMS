"""
Analysis router — vibration signal processing and fault classification endpoints.

Endpoints:
  POST /api/analysis/classify       — Full fault classification (6 or 7 classes)
  POST /api/analysis/features       — Feature extraction (48 or 54 features)
  POST /api/analysis/antigravity    — Dedicated antigravity regime analysis
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from schemas import (
    AnalysisRequest,
    AntigravityAnalysisRequest,
    AntigravityAnalysisResponse,
    AntigravityFeatureDetail,
    ClassificationResponse,
    FeaturesRequest,
    FeaturesResponse,
)
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
from services.inference import InferenceService

router = APIRouter()

# ── Normal-gravity baselines (mean ± std) for antigravity features ────────────
# These are empirically defined from normal-gravity training data.
# Update after running the RF retraining script with real hardware data.
ANTIGRAVITY_BASELINES: dict[str, tuple[float, float]] = {
    "gli":  (0.98, 0.04),   # mean, std
    "ser":  (0.06, 0.02),
    "hcc":  (0.05, 0.03),
    "cpc":  (0.75, 0.08),
    "blza": (1.50, 0.25),
    "gpi":  (0.98, 0.12),   # near-integer under normal gravity
}


def _get_inference(request: Request) -> InferenceService:
    """Retrieve the shared InferenceService from application state."""
    return request.app.state.inference


def _validate_signal_lengths(*signals: list[float]) -> None:
    """Ensure all signals have equal length."""
    lengths = [len(s) for s in signals]
    if len(set(lengths)) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"All signal axes must have equal length. Got: {lengths}",
        )


# ── Endpoint 1: Full fault classification ─────────────────────────────────────


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify fault from a 3-axis vibration window",
    description=(
        "Extracts a 48 or 54-element feature vector (depending on ANTIGRAVITY_ENABLED) "
        "from the provided X/Y/Z accelerometer signal window and returns the predicted "
        "fault class along with per-class softmax probabilities."
    ),
)
async def classify_fault(
    payload: AnalysisRequest,
    request: Request,
) -> ClassificationResponse:
    """Run full fault classification on a 3-axis vibration window."""
    _validate_signal_lengths(payload.signal_x, payload.signal_y, payload.signal_z)

    inference = _get_inference(request)
    x = np.array(payload.signal_x)
    y = np.array(payload.signal_y)
    z = np.array(payload.signal_z)

    fv = extract_features(x, y, z, shaft_rpm=payload.shaft_rpm)
    features = fv.features

    predicted, confidence, prob_dict = inference.predict_rf(features)
    ag_prob = inference.get_antigravity_probability(prob_dict)

    return ClassificationResponse(
        session_id=payload.session_id,
        window_index=payload.window_index,
        predicted_class=predicted,
        confidence=confidence,
        class_probabilities=prob_dict,
        model_used="random_forest_v2" if inference.antigravity_enabled else "random_forest_v1",
        feature_vector_length=len(features),
        antigravity_probability=ag_prob,
    )


# ── Endpoint 2: Feature extraction ───────────────────────────────────────────


@router.post(
    "/features",
    response_model=FeaturesResponse,
    summary="Extract feature vector from a 3-axis vibration window",
    description=(
        "Returns the 48-element baseline feature vector, or 54 elements when "
        "ANTIGRAVITY_ENABLED=true. Set include_antigravity_features=true to also "
        "receive individual antigravity feature details with normal-gravity baseline refs."
    ),
)
async def extract_features_endpoint(
    payload: FeaturesRequest,
    request: Request,
) -> FeaturesResponse:
    """Extract the vibration feature vector, optionally including antigravity features."""
    _validate_signal_lengths(payload.signal_x, payload.signal_y, payload.signal_z)

    inference = _get_inference(request)
    x = np.array(payload.signal_x)
    y = np.array(payload.signal_y)
    z = np.array(payload.signal_z)

    fv = extract_features(x, y, z, shaft_rpm=payload.shaft_rpm)
    features_list = fv.features.tolist()
    names = fv.names

    # If caller does not want antigravity features and system is not AG-enabled,
    # return only the first 48 baseline features.
    if not inference.antigravity_enabled and not payload.include_antigravity_features:
        features_list = features_list[:48]
        names = names[:48]

    ag_details: list[AntigravityFeatureDetail] | None = None
    if payload.include_antigravity_features:
        ag_details = []
        ag_feature_values = {
            "gli":  float(fv.features[48]),
            "ser":  float(fv.features[49]),
            "hcc":  float(fv.features[50]),
            "cpc":  float(fv.features[51]),
            "blza": float(fv.features[52]),
            "gpi":  float(fv.features[53]),
        }
        for feat_name, val in ag_feature_values.items():
            mean, std = ANTIGRAVITY_BASELINES[feat_name]
            ag_details.append(
                AntigravityFeatureDetail(
                    name=feat_name.upper(),
                    value=val,
                    baseline_mean=mean,
                    baseline_std=std,
                    is_anomalous=abs(val - mean) > 2.0 * std,
                )
            )

    return FeaturesResponse(
        session_id=payload.session_id,
        window_index=payload.window_index,
        features=features_list,
        feature_names=names,
        antigravity_features=ag_details,
    )


# ── Endpoint 3: Dedicated antigravity analysis ────────────────────────────────


@router.post(
    "/antigravity",
    response_model=AntigravityAnalysisResponse,
    summary="Dedicated antigravity regime analysis",
    description=(
        "Computes all six antigravity-sensitive features (GLI, SER, HCC, CPC, BLZA, GPI) "
        "from the provided vibration window and returns the antigravity softmax probability "
        "alongside the classified operating regime.\n\n"
        "**Regimes:**\n"
        "- `normal_gravity`: probability < 0.30\n"
        "- `reduced_gravity`: 0.30 ≤ probability < 0.85\n"
        "- `near_zero_gravity`: probability ≥ 0.85\n\n"
        "**Example response:**\n"
        "```json\n"
        '{"gli": 0.12, "ser": 0.38, "hcc": 0.71, "cpc": 0.45, "blza": 1.85, '
        '"gpi": 0.43, "antigravity_probability": 0.87, "regime": "near_zero_gravity"}\n'
        "```"
    ),
    responses={
        200: {
            "description": "Successful antigravity analysis",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_001",
                        "window_index": 42,
                        "gli": 0.12,
                        "ser": 0.38,
                        "hcc": 0.71,
                        "cpc": 0.45,
                        "blza": 1.85,
                        "gpi": 0.43,
                        "antigravity_probability": 0.87,
                        "regime": "near_zero_gravity",
                    }
                }
            },
        }
    },
)
async def antigravity_analysis(
    payload: AntigravityAnalysisRequest,
    request: Request,
) -> AntigravityAnalysisResponse:
    """Compute all six antigravity features and the antigravity regime probability."""
    _validate_signal_lengths(payload.signal_x, payload.signal_y, payload.signal_z)

    inference = _get_inference(request)
    x = np.array(payload.signal_x)
    y = np.array(payload.signal_y)
    z = np.array(payload.signal_z)
    combined = (x + y + z) / np.sqrt(3)

    fv = extract_features(x, y, z, shaft_rpm=payload.shaft_rpm)
    features = fv.features

    # Compute antigravity features directly for explicit response fields
    from services.feature_extractor import SAMPLE_RATE  # noqa: PLC0415

    gli  = compute_gli(z, SAMPLE_RATE)
    ser  = compute_ser(combined, payload.shaft_rpm, SAMPLE_RATE)
    hcc  = compute_hcc(float(features[28]))  # harmonic_1x index
    cpc  = compute_cpc(x, z, payload.shaft_rpm, SAMPLE_RATE)
    bpfo = float(features[25])
    bpfi = float(features[24])
    blza = compute_blza(bpfo, bpfi)
    gpi  = compute_gpi(z, payload.shaft_rpm, SAMPLE_RATE)

    _, _, prob_dict = inference.predict_rf(features)
    ag_prob = inference.get_antigravity_probability(prob_dict) or 0.0
    regime = inference.classify_regime(ag_prob)

    return AntigravityAnalysisResponse(
        session_id=payload.session_id,
        window_index=payload.window_index,
        gli=gli,
        ser=ser,
        hcc=hcc,
        cpc=cpc,
        blza=blza,
        gpi=gpi,
        antigravity_probability=ag_prob,
        regime=regime,
    )
