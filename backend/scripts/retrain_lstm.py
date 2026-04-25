"""
LSTM retraining script — extends 6-class model to 7-class with antigravity.

Strategy:
  - Load existing SavedModel (or build a new one if absent).
  - Extend the final dense layer from 6 → 7 output units.
  - Freeze all layers except the last two (encoder frozen, head trainable).
  - Fine-tune for 30 epochs with class_weight compensation.
  - Save updated SavedModel to services/models/lstm_model_v2/.

Usage:
    cd backend
    python scripts/retrain_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

BACKEND_DIR = Path(__file__).resolve().parents[1]
FEAT_TRAIN_PATH = BACKEND_DIR / "data" / "processed" / "features_train.npy"
LABELS_TRAIN_PATH = BACKEND_DIR / "data" / "processed" / "labels_train.npy"
AG_FEATURES_CACHE = BACKEND_DIR / "data" / "synthetic" / "ag_features_54.npy"
AG_LABELS_PATH = BACKEND_DIR / "data" / "synthetic" / "antigravity_labels.npy"
LSTM_V1_PATH = BACKEND_DIR / "services" / "models" / "lstm_model"
LSTM_V2_PATH = BACKEND_DIR / "services" / "models" / "lstm_model_v2"

N_CLASSES_V2 = 8  # 0-6 original + 7 antigravity (index 7 for LSTM)
N_FEATURES = 54
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "normal", "bpfi", "bpfo", "bsf",
    "unbalance", "misalignment", "looseness", "antigravity",
]


def _build_lstm_model(n_features: int, n_classes: int):
    """Build a fresh 1D LSTM classifier if no pretrained model is available."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        logger.error("TensorFlow is not installed. Run: pip install tensorflow>=2.16.0")
        sys.exit(1)

    inputs = tf.keras.Input(shape=(1, n_features), name="feature_input")
    x = tf.keras.layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM_CMS_v2")
    return model


def _extend_lstm_head(model, n_classes_new: int):
    """
    Extend the final dense output layer of an existing model from N → N+1 classes.

    Architecture:
      - Keep all layers except the last Dense (output) layer.
      - Add new Dense(n_classes_new, activation='softmax') as the new head.
      - Freeze all layers except the last two (trainable encoder layers remain frozen).
    """
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        logger.error("TensorFlow is not installed.")
        sys.exit(1)

    # Rebuild model discarding the last output layer
    penultimate_output = model.layers[-2].output
    new_output = tf.keras.layers.Dense(
        n_classes_new, activation="softmax", name="output_v2"
    )(penultimate_output)
    new_model = tf.keras.Model(inputs=model.inputs, outputs=new_output)

    # Freeze all layers except the last two
    for layer in new_model.layers[:-2]:
        layer.trainable = False

    trainable_names = [l.name for l in new_model.layers if l.trainable]
    frozen_names = [l.name for l in new_model.layers if not l.trainable]
    logger.info(f"Trainable layers : {trainable_names}")
    logger.info(f"Frozen layers    : {frozen_names}")

    return new_model


def retrain() -> None:
    """End-to-end LSTM fine-tuning pipeline."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        logger.error("TensorFlow is not installed. Run: pip install tensorflow>=2.16.0")
        sys.exit(1)

    LSTM_V2_PATH.mkdir(parents=True, exist_ok=True)

    # ── Load or build base LSTM ───────────────────────────────────────────────
    if LSTM_V1_PATH.exists():
        logger.info(f"Loading existing LSTM from {LSTM_V1_PATH} …")
        base_model = tf.saved_model.load(str(LSTM_V1_PATH))
        # Wrap as Keras if possible; otherwise build fresh
        try:
            model = tf.keras.models.load_model(str(LSTM_V1_PATH))
            logger.info("Loaded as Keras model — extending output head.")
            model = _extend_lstm_head(model, N_CLASSES_V2)
        except Exception:
            logger.warning("Could not reload as Keras model; building fresh v2 model.")
            model = _build_lstm_model(N_FEATURES, N_CLASSES_V2)
    else:
        logger.warning(f"No v1 LSTM found at {LSTM_V1_PATH} — building fresh v2 model.")
        model = _build_lstm_model(N_FEATURES, N_CLASSES_V2)

    # ── Load training data ────────────────────────────────────────────────────
    from scripts.retrain_rf import _generate_synthetic_baseline  # noqa: PLC0415

    if FEAT_TRAIN_PATH.exists():
        X_base = np.load(FEAT_TRAIN_PATH)
        y_base = np.load(LABELS_TRAIN_PATH)
        if X_base.shape[1] == 48:
            X_base = np.hstack([X_base, np.zeros((len(X_base), 6))])
    else:
        X_base, y_base = _generate_synthetic_baseline()

    # Load or compute antigravity features
    if AG_FEATURES_CACHE.exists():
        X_ag = np.load(AG_FEATURES_CACHE)
        y_ag = np.load(AG_LABELS_PATH)
    else:
        logger.warning("AG feature cache not found. Run retrain_rf.py first, or generate data.")
        logger.warning("Using small synthetic antigravity feature set as fallback.")
        rng = np.random.default_rng(99)
        X_ag = rng.random((500, N_FEATURES)).astype(np.float32)
        # Mark as antigravity class (7 for LSTM, offset by 1 from RF)
        y_ag = np.full(500, 7, dtype=np.int32)

    # Remap antigravity label from RF index 6 → LSTM index 7 if needed
    y_ag_lstm = np.where(y_ag == 6, 7, y_ag)

    X_full = np.vstack([X_base, X_ag]).astype(np.float32)
    y_full = np.concatenate([y_base, y_ag_lstm]).astype(np.int32)

    # ── Class weights ─────────────────────────────────────────────────────────
    from sklearn.utils.class_weight import compute_class_weight  # noqa: PLC0415

    unique_classes = np.unique(y_full)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_full)
    class_weight_dict = dict(zip(unique_classes.tolist(), weights.tolist()))
    logger.info(f"Class weights: {class_weight_dict}")

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Reshape for LSTM (samples, timesteps=1, features) ────────────────────
    X_lstm = X_full[:, np.newaxis, :]  # (N, 1, 54)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    logger.info(f"Fine-tuning LSTM for up to {EPOCHS} epochs …")
    history = model.fit(
        X_lstm, y_full,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    final_acc = history.history["val_accuracy"][-1]
    logger.success(f"Training complete. Final val accuracy: {final_acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(str(LSTM_V2_PATH))
    logger.success(f"LSTM v2 SavedModel saved → {LSTM_V2_PATH}")


if __name__ == "__main__":
    retrain()
