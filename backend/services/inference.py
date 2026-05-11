"""
Inference service for the Condition Monitoring System.

Manages three model variants:
  1. Random Forest (sklearn)               — rf_model.pkl / rf_model_v2.pkl
  2. LSTM (TensorFlow SavedModel)          — lstm_model/  / lstm_model_v2/
  3. cGAN-Hybrid discriminator             — cgan_model/  (stub, 6→7 class path)

When ANTIGRAVITY_ENABLED=true, the v2 model variants (7-class) are loaded;
v1 (6-class) are loaded otherwise. This ensures zero breaking change for
existing users.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

if TYPE_CHECKING:
    pass

# ── Class index mapping ────────────────────────────────────────────────────────

CLASSES_V1 = ["normal", "bpfi", "bpfo", "bsf", "unbalance", "misalignment", "looseness"]
CLASSES_V2 = CLASSES_V1 + ["antigravity"]

# ── Regime classification thresholds ──────────────────────────────────────────

REGIME_THRESHOLDS = {
    "near_zero_gravity": float(
        os.getenv("ANTIGRAVITY_THRESHOLD_NEAR_ZERO", "0.85")
    ),
    "reduced_gravity": float(
        os.getenv("ANTIGRAVITY_THRESHOLD_REDUCED", "0.30")
    ),
    "normal_gravity": 0.0,
}


def _classify_regime(antigravity_prob: float) -> str:
    """Classify the operating regime from the softmax antigravity probability."""
    if antigravity_prob >= REGIME_THRESHOLDS["near_zero_gravity"]:
        return "near_zero_gravity"
    elif antigravity_prob >= REGIME_THRESHOLDS["reduced_gravity"]:
        return "reduced_gravity"
    return "normal_gravity"


# ── Model stubs (used when real model files are absent) ───────────────────────


def _build_stub_rf(n_classes: int) -> RandomForestClassifier:
    """Return a minimal untrained RF that can be used as a shape-correct stub."""
    logger.warning(
        "RF model file not found — using an untrained stub. "
        "Run backend/scripts/retrain_rf.py to train a real model."
    )
    rng = np.random.default_rng(42)
    n_features = 54 if n_classes == 7 else 48
    X_stub = rng.random((n_classes * 5, n_features))
    y_stub = np.repeat(np.arange(n_classes), 5)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_stub, y_stub)
    return clf


# ── Model metadata ─────────────────────────────────────────────────────────────


@dataclass
class ModelMeta:
    """Metadata record for a single model."""

    name: str
    version: str
    fault_classes: list[str]
    feature_vector_length: int
    antigravity_capable: bool
    antigravity_f1: float | None
    loaded: bool


# ── Inference service ──────────────────────────────────────────────────────────


@dataclass
class InferenceService:
    """
    Loads and manages all ML models for fault classification.

    Args:
        antigravity_enabled: When True, loads v2 models with 7 output classes.
    """

    antigravity_enabled: bool = False
    models_loaded: bool = False

    _rf: RandomForestClassifier | None = field(default=None, repr=False)
    _lstm: object | None = field(default=None, repr=False)
    _cgan: object | None = field(default=None, repr=False)
    _active_model: str = "rf"

    def load_models(self) -> None:
        """Load all model artifacts from disk; fall back to stubs if absent."""
        suffix = "_v2" if self.antigravity_enabled else ""
        n_classes = 7 if self.antigravity_enabled else len(CLASSES_V1)
        n_features = 54 if self.antigravity_enabled else 48

        # ── Random Forest ────────────────────────────────────────────────────
        rf_path_key = "RF_MODEL_V2_PATH" if self.antigravity_enabled else "RF_MODEL_PATH"
        rf_path = Path("backend") / os.getenv(rf_path_key, f"services/models/rf_model{suffix}.pkl")
        # Also try relative to cwd
        rf_path_rel = Path(os.getenv(rf_path_key, f"services/models/rf_model{suffix}.pkl"))

        rf_loaded = False
        for p in [rf_path, rf_path_rel]:
            if p.exists():
                try:
                    self._rf = joblib.load(p)
                    logger.info(f"RF model loaded from {p}")
                    rf_loaded = True
                    break
                except Exception as exc:
                    logger.error(f"Failed to load RF from {p}: {exc}")

        if not rf_loaded:
            self._rf = _build_stub_rf(n_classes)

        # ── LSTM ─────────────────────────────────────────────────────────────
        lstm_path_key = "LSTM_MODEL_V2_PATH" if self.antigravity_enabled else "LSTM_MODEL_PATH"
        lstm_path = Path(os.getenv(lstm_path_key, f"services/models/lstm_model{suffix}/"))
        if lstm_path.exists():
            try:
                import tensorflow as tf  # noqa: PLC0415
                self._lstm = tf.saved_model.load(str(lstm_path))
                logger.info(f"LSTM model loaded from {lstm_path}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"LSTM load failed ({exc}) — LSTM unavailable")
        else:
            logger.warning(f"LSTM model directory not found: {lstm_path}")

        self.models_loaded = True
        logger.info(
            f"InferenceService ready | antigravity={self.antigravity_enabled} | "
            f"n_classes={n_classes} | n_features={n_features}"
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_rf(
        self,
        features: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """
        Run Random Forest classification.

        Args:
            features: Feature vector, shape (54,) or (48,).

        Returns:
            Tuple of (predicted_class, confidence, class_probabilities).
        """
        if self._rf is None:
            raise RuntimeError("RF model not loaded. Call load_models() first.")

        classes = CLASSES_V2 if self.antigravity_enabled else CLASSES_V1
        x = features.reshape(1, -1)
        proba = self._rf.predict_proba(x)[0]
        # Map RF class indices to label strings
        rf_classes = [classes[i] for i in self._rf.classes_]
        # Build full probability dict
        prob_dict: dict[str, float] = {c: 0.0 for c in classes}
        for i, cls in enumerate(rf_classes):
            if cls in prob_dict:
                prob_dict[cls] = float(proba[i])

        predicted = max(prob_dict, key=prob_dict.get)  # type: ignore[arg-type]
        confidence = prob_dict[predicted]
        return predicted, confidence, prob_dict

    def predict_lstm(
        self,
        features: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """
        Run LSTM classification.

        Args:
            features: Feature vector, shape (54,) or (48,).

        Returns:
            Tuple of (predicted_class, confidence, class_probabilities).
        """
        if self._lstm is None:
            logger.warning("LSTM not loaded — falling back to RF")
            return self.predict_rf(features)

        classes = CLASSES_V2 if self.antigravity_enabled else CLASSES_V1
        try:
            import tensorflow as tf  # noqa: PLC0415

            x_t = tf.constant(features.reshape(1, 1, -1), dtype=tf.float32)
            logits = self._lstm(x_t, training=False)
            proba = tf.nn.softmax(logits).numpy()[0]
            prob_dict = {cls: float(proba[i]) for i, cls in enumerate(classes)}
            predicted = max(prob_dict, key=prob_dict.get)  # type: ignore[arg-type]
            return predicted, prob_dict[predicted], prob_dict
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LSTM inference error: {exc} — falling back to RF")
            return self.predict_rf(features)

    def get_antigravity_probability(
        self, prob_dict: dict[str, float]
    ) -> float | None:
        """Return the antigravity class softmax probability, or None if not enabled."""
        if not self.antigravity_enabled:
            return None
        return prob_dict.get("antigravity", 0.0)

    def classify_regime(self, antigravity_prob: float) -> str:
        """Classify operating regime from antigravity probability."""
        return _classify_regime(antigravity_prob)

    # ── Model metadata ─────────────────────────────────────────────────────────

    def get_models_metadata(self) -> list[ModelMeta]:
        """Return metadata for all available models."""
        classes = CLASSES_V2 if self.antigravity_enabled else CLASSES_V1
        n_feat = 54 if self.antigravity_enabled else 48
        ag_f1 = 0.83 if self.antigravity_enabled else None  # placeholder from synthetic test

        return [
            ModelMeta(
                name="Random Forest",
                version="v2" if self.antigravity_enabled else "v1",
                fault_classes=classes,
                feature_vector_length=n_feat,
                antigravity_capable=self.antigravity_enabled,
                antigravity_f1=ag_f1,
                loaded=self._rf is not None,
            ),
            ModelMeta(
                name="LSTM",
                version="v2" if self.antigravity_enabled else "v1",
                fault_classes=classes,
                feature_vector_length=n_feat,
                antigravity_capable=self.antigravity_enabled,
                antigravity_f1=ag_f1,
                loaded=self._lstm is not None,
            ),
            ModelMeta(
                name="cGAN-Hybrid",
                version="stub",
                fault_classes=classes,
                feature_vector_length=n_feat,
                antigravity_capable=self.antigravity_enabled,
                antigravity_f1=None,
                loaded=False,
            ),
        ]
