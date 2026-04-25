"""
Random Forest retraining script — 7-class (adds antigravity class).

Workflow:
  1. Load existing 6-class training data (features_train.npy / labels_train.npy)
     OR generate synthetic baseline data if files are absent.
  2. Extract 54-feature vectors from synthetic antigravity windows.
  3. Augment: stack antigravity samples with class label 6.
  4. Train RandomForestClassifier (500 estimators, class_weight='balanced').
  5. Evaluate on held-out test set — print full classification report.
  6. Save model to services/models/rf_model_v2.pkl.

Usage:
    cd backend
    python scripts/retrain_rf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from services.feature_extractor import FEATURE_NAMES, extract_features

# ── Paths ─────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parents[1]
FEAT_TRAIN_PATH = BACKEND_DIR / "data" / "processed" / "features_train.npy"
LABELS_TRAIN_PATH = BACKEND_DIR / "data" / "processed" / "labels_train.npy"
AG_DATA_PATH = BACKEND_DIR / "data" / "synthetic" / "antigravity_1024.npy"
AG_LABELS_PATH = BACKEND_DIR / "data" / "synthetic" / "antigravity_labels.npy"
MODEL_OUT_PATH = BACKEND_DIR / "services" / "models" / "rf_model_v2.pkl"

CLASS_NAMES = [
    "normal", "bpfi", "bpfo", "bsf",
    "unbalance", "misalignment", "looseness", "antigravity",
]

SHAFT_RPM = 1500.0


def _generate_synthetic_baseline() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic baseline 6-class features when no training data exists.
    Each class gets 400 samples (54 features) for a total of 2400 samples.
    """
    logger.warning("features_train.npy not found — generating synthetic baseline features.")
    rng = np.random.default_rng(42)
    n_per_class = 400
    n_classes_base = 7  # normal + 6 fault classes (excluding antigravity)
    all_X, all_y = [], []

    class_signatures = {
        0: {"harmonic_1x_boost": 1.0, "kurtosis_boost": 1.0},   # normal
        1: {"bpfi_boost": 3.0, "kurtosis_boost": 4.5},           # bpfi
        2: {"bpfo_boost": 2.5, "kurtosis_boost": 3.8},           # bpfo
        3: {"bsf_boost": 2.0, "kurtosis_boost": 2.5},            # bsf
        4: {"harmonic_1x_boost": 3.0, "harmonic_2x_boost": 1.5}, # unbalance
        5: {"harmonic_2x_boost": 2.8, "harmonic_3x_boost": 1.8}, # misalignment
        6: {"kurtosis_boost": 1.5, "broadband_boost": 2.0},      # looseness
    }

    for cls_idx in range(n_classes_base):
        features = rng.normal(loc=0.5, scale=0.15, size=(n_per_class, 54))
        features = np.clip(features, 0.0, 5.0)
        sig = class_signatures.get(cls_idx, {})
        if "bpfi_boost" in sig:   features[:, 24] *= sig["bpfi_boost"]
        if "bpfo_boost" in sig:   features[:, 25] *= sig["bpfo_boost"]
        if "bsf_boost" in sig:    features[:, 26] *= sig["bsf_boost"]
        if "kurtosis_boost" in sig: features[:, 3] *= sig["kurtosis_boost"]
        if "harmonic_1x_boost" in sig: features[:, 28] *= sig["harmonic_1x_boost"]
        if "harmonic_2x_boost" in sig: features[:, 29] *= sig["harmonic_2x_boost"]
        if "harmonic_3x_boost" in sig: features[:, 30] *= sig["harmonic_3x_boost"]
        all_X.append(features)
        all_y.append(np.full(n_per_class, cls_idx, dtype=np.int32))

    return np.vstack(all_X), np.concatenate(all_y)


def _extract_ag_features() -> tuple[np.ndarray, np.ndarray]:
    """Extract 54-feature vectors from the synthetic antigravity windows."""
    if not AG_DATA_PATH.exists():
        logger.error(f"Antigravity data not found at {AG_DATA_PATH}.")
        logger.error("Run: python backend/scripts/generate_antigravity_data.py first.")
        sys.exit(1)

    logger.info(f"Loading antigravity windows from {AG_DATA_PATH} …")
    windows = np.load(AG_DATA_PATH)  # (2000, 1024, 3)
    labels = np.load(AG_LABELS_PATH)

    logger.info(f"Extracting features from {len(windows)} antigravity windows …")
    features_list = []
    for i, w in enumerate(windows):
        x, y, z = w[:, 0], w[:, 1], w[:, 2]
        fv = extract_features(x, y, z, shaft_rpm=SHAFT_RPM)
        features_list.append(fv.features)
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i + 1}/{len(windows)} windows …")

    return np.array(features_list), labels


def retrain() -> None:
    """End-to-end RF retraining pipeline."""
    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load or generate baseline features ────────────────────────────────────
    if FEAT_TRAIN_PATH.exists() and LABELS_TRAIN_PATH.exists():
        logger.info(f"Loading existing training data from {FEAT_TRAIN_PATH} …")
        X_base = np.load(FEAT_TRAIN_PATH)
        y_base = np.load(LABELS_TRAIN_PATH)
        # Pad to 54 features if still 48-feature legacy data
        if X_base.shape[1] == 48:
            logger.info("Padding 48-feature legacy data to 54 features (zeros for new features).")
            X_base = np.hstack([X_base, np.zeros((len(X_base), 6))])
    else:
        X_base, y_base = _generate_synthetic_baseline()

    logger.info(f"Baseline data: {X_base.shape}, classes: {np.unique(y_base)}")

    # ── Extract antigravity features ──────────────────────────────────────────
    X_ag, y_ag = _extract_ag_features()
    logger.info(f"Antigravity data: {X_ag.shape}, class label: {np.unique(y_ag)}")

    # ── Augment dataset ───────────────────────────────────────────────────────
    X_full = np.vstack([X_base, X_ag])
    y_full = np.concatenate([y_base, y_ag])
    logger.info(f"Augmented dataset: {X_full.shape}, classes: {np.unique(y_full)}")

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Train Random Forest ───────────────────────────────────────────────────
    logger.info("Training RandomForestClassifier (500 estimators, class_weight='balanced') …")
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_split=2,
    )
    clf.fit(X_train, y_train)
    logger.success("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_class_names = [CLASS_NAMES[i] for i in present_labels]
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_class_names,
    )
    print("\n" + "═" * 60)
    print("  RANDOM FOREST v2 — CLASSIFICATION REPORT (7-class)")
    print("═" * 60)
    print(report)
    print("═" * 60 + "\n")

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_OUT_PATH)
    logger.success(f"Model saved → {MODEL_OUT_PATH}")
    logger.info(f"Feature names (54): {FEATURE_NAMES}")


if __name__ == "__main__":
    retrain()
