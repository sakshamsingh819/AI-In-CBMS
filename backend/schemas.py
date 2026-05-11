"""
Pydantic schemas for the Condition Monitoring System API.
All request/response models are defined here.
Feature vector length: 54 (48 baseline + 6 antigravity-sensitive features).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Allow fields starting with 'model_'
BaseModel.model_config = ConfigDict(protected_namespaces=())

# ── Class labels ──────────────────────────────────────────────────────────────

FaultClass = Literal[
    "normal",
    "bpfi",       # Ball Pass Frequency Inner race
    "bpfo",       # Ball Pass Frequency Outer race
    "bsf",        # Ball Spin Frequency
    "unbalance",
    "misalignment",
    "looseness",
    "antigravity",  # Reduced-gravity operating regime (experimental)
]

AntigravityRegime = Literal["normal_gravity", "reduced_gravity", "near_zero_gravity"]

# ── Feature names (54 total) ──────────────────────────────────────────────────
# Imported from feature_extractor to maintain single source of truth.

# ── Request models ────────────────────────────────────────────────────────────


class AnalysisRequest(BaseModel):
    """Request payload for the main fault-classification endpoint."""

    session_id: str = Field(..., description="Unique analysis session identifier")
    window_index: int = Field(..., ge=0, description="Index of the vibration window")
    signal_x: list[float] = Field(
        ..., min_length=256, max_length=8192, description="X-axis accelerometer samples"
    )
    signal_y: list[float] = Field(
        ..., min_length=256, max_length=8192, description="Y-axis accelerometer samples"
    )
    signal_z: list[float] = Field(
        ..., min_length=256, max_length=8192, description="Z-axis accelerometer samples"
    )
    shaft_rpm: float = Field(
        1500.0, gt=0, description="Current shaft rotational speed in RPM"
    )


class FeaturesRequest(BaseModel):
    """Request payload for the feature extraction endpoint."""

    session_id: str = Field(..., description="Unique analysis session identifier")
    window_index: int = Field(..., ge=0, description="Index of the vibration window")
    signal_x: list[float] = Field(..., min_length=256, max_length=8192)
    signal_y: list[float] = Field(..., min_length=256, max_length=8192)
    signal_z: list[float] = Field(..., min_length=256, max_length=8192)
    shaft_rpm: float = Field(1500.0, gt=0)
    include_antigravity_features: bool = Field(
        False,
        description=(
            "When true, appends all 6 antigravity-sensitive features to the "
            "response alongside their computed values and normal-gravity baseline refs"
        ),
    )


class AntigravityAnalysisRequest(BaseModel):
    """Request payload for the dedicated antigravity analysis endpoint."""

    session_id: str = Field(..., description="Unique analysis session identifier")
    window_index: int = Field(..., ge=0, description="Index of the vibration window")
    shaft_rpm: float = Field(1500.0, gt=0, description="Shaft speed in RPM")
    signal_x: list[float] = Field(..., min_length=256, max_length=8192)
    signal_y: list[float] = Field(..., min_length=256, max_length=8192)
    signal_z: list[float] = Field(..., min_length=256, max_length=8192)


# ── Response models ───────────────────────────────────────────────────────────


class ClassificationResponse(BaseModel):
    """Fault classification result from any of the three inference models."""

    session_id: str
    window_index: int
    predicted_class: FaultClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: dict[FaultClass, float]
    model_used: str
    feature_vector_length: int
    antigravity_probability: float | None = Field(
        None,
        description="Softmax probability of the antigravity class (null when ANTIGRAVITY_ENABLED=false)",
    )


class AntigravityFeatureDetail(BaseModel):
    """Single antigravity feature with its value and baseline reference."""

    name: str
    value: float
    baseline_mean: float
    baseline_std: float
    is_anomalous: bool = Field(
        description="True when value is outside baseline ± 2σ"
    )


class FeaturesResponse(BaseModel):
    """Extracted feature vector response."""

    session_id: str
    window_index: int
    features: Annotated[list[float], Field(min_length=48, max_length=54)]
    feature_names: list[str]
    antigravity_features: list[AntigravityFeatureDetail] | None = None


class AntigravityAnalysisResponse(BaseModel):
    """
    Dedicated antigravity analysis response.

    Example response:
    {
        "session_id": "sess_001",
        "window_index": 42,
        "gli": 0.12,
        "ser": 0.38,
        "hcc": 0.71,
        "cpc": 0.45,
        "blza": 1.85,
        "gpi": 0.43,
        "antigravity_probability": 0.87,
        "regime": "near_zero_gravity"
    }
    """

    session_id: str
    window_index: int
    gli: float = Field(..., description="Gravity Load Index (0 = full cancellation)")
    ser: float = Field(..., description="Sub-synchronous Energy Ratio")
    hcc: float = Field(..., description="Harmonic Cancellation Coefficient")
    cpc: float = Field(..., description="Cross-axis Phase Coherence")
    blza: float = Field(..., description="Bearing Load Zone Asymmetry")
    gpi: float = Field(..., description="Gyroscopic Precession Index")
    antigravity_probability: float = Field(..., ge=0.0, le=1.0)
    regime: AntigravityRegime


class ModelInfo(BaseModel):
    """Metadata for a single inference model."""

    name: str
    version: str
    fault_classes: list[str]
    feature_vector_length: int
    antigravity_capable: bool = Field(
        description="True when model was trained with the 7th antigravity class"
    )
    antigravity_f1: float | None = Field(
        None,
        description="Per-class F1-score for antigravity on the synthetic test set",
    )
    loaded: bool


class ModelsResponse(BaseModel):
    """Response for GET /api/models."""

    models: list[ModelInfo]
    active_model: str
    antigravity_enabled: bool
