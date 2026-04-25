"""
CMS Demo Results Script
=======================
Generates all scores, graphs and results for the Condition Monitoring System.

Run from the backend/ directory:
    python scripts/demo_results.py

Outputs (saved to backend/results/):
    1. classification_report.txt    -- precision / recall / F1 per class
    2. confusion_matrix.png         -- colour heatmap
    3. feature_importance.png       -- top-20 RF feature importances
    4. roc_curves.png               -- per-class ROC curves (OvR)
    5. antigravity_feature_dist.png -- distribution of 6 AG features in each class

Requires only: numpy, scipy, scikit-learn, matplotlib
               (no FastAPI, no TensorFlow, no database)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from backend/ or from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from services.feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FS         = 26667
SHAFT_RPM  = 1500.0
FR         = SHAFT_RPM / 60   # 25 Hz
G          = 9.81
N_WINDOW   = 1024
RNG        = np.random.default_rng(42)

CLASS_NAMES = [
    "normal", "bpfi", "bpfo", "bsf",
    "unbalance", "misalignment", "looseness", "antigravity",
]

COLOURS = [
    "#22c55e", "#3b82f6", "#06b6d4", "#8b5cf6",
    "#f59e0b", "#f97316", "#ec4899", "#7c3aed",
]

N_PER_CLASS    = 300   # samples per class
N_AG_SAMPLES   = 300   # antigravity samples


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Synthetic signal generators (one per fault class)
# ══════════════════════════════════════════════════════════════════════════════

def _noise(n: int, snr_db: float = 25) -> np.ndarray:
    """White noise scaled to a given SNR (relative to unit-power signal)."""
    noise_power = 1.0 / (10 ** (snr_db / 10))
    return RNG.normal(0, np.sqrt(noise_power), n)

def _t() -> np.ndarray:
    return np.arange(N_WINDOW) / FS

T = _t()

# Bearing coefficients (SKF 6205)
BPFI = 5.415 * FR
BPFO = 3.585 * FR
BSF  = 2.357 * FR
FTF  = 0.3983 * FR


def gen_normal(_=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = 0.8 * np.sin(2*np.pi*FR*T) + 0.2*np.sin(2*np.pi*2*FR*T) + _noise(N_WINDOW)
    y = 0.7 * np.sin(2*np.pi*FR*T + np.pi/2) + _noise(N_WINDOW)
    z = G   + 0.1 * np.sin(2*np.pi*FR*T)    + _noise(N_WINDOW)
    return x, y, z

def gen_bpfi(_=None):
    x = (0.6*np.sin(2*np.pi*FR*T)
         + 1.8*np.sin(2*np.pi*BPFI*T)
         + 0.4*np.sin(2*np.pi*2*BPFI*T)
         + _noise(N_WINDOW, snr_db=22))
    y = 0.5*np.sin(2*np.pi*FR*T+np.pi/2) + _noise(N_WINDOW)
    z = G + 0.3*np.sin(2*np.pi*BPFI*T) + _noise(N_WINDOW)
    return x, y, z

def gen_bpfo(_=None):
    x = (0.6*np.sin(2*np.pi*FR*T)
         + 1.5*np.sin(2*np.pi*BPFO*T)
         + 0.5*np.sin(2*np.pi*2*BPFO*T)
         + _noise(N_WINDOW, snr_db=22))
    y = 0.5*np.sin(2*np.pi*FR*T+np.pi/2) + _noise(N_WINDOW)
    z = G + 0.4*np.sin(2*np.pi*BPFO*T) + _noise(N_WINDOW)
    return x, y, z

def gen_bsf(_=None):
    x = (0.6*np.sin(2*np.pi*FR*T)
         + 1.2*np.sin(2*np.pi*BSF*T)
         + 0.3*np.sin(2*np.pi*2*BSF*T)
         + _noise(N_WINDOW, snr_db=23))
    y = 0.5*np.sin(2*np.pi*FR*T+np.pi/2) + _noise(N_WINDOW)
    z = G + 0.2*np.sin(2*np.pi*BSF*T) + _noise(N_WINDOW)
    return x, y, z

def gen_unbalance(_=None):
    x = (2.5*np.sin(2*np.pi*FR*T)
         + 0.8*np.sin(2*np.pi*2*FR*T)
         + 0.2*np.sin(2*np.pi*3*FR*T)
         + _noise(N_WINDOW, snr_db=28))
    y = 2.2*np.sin(2*np.pi*FR*T+np.pi/2) + _noise(N_WINDOW)
    z = G + 0.5*np.sin(2*np.pi*FR*T) + _noise(N_WINDOW)
    return x, y, z

def gen_misalignment(_=None):
    x = (1.0*np.sin(2*np.pi*FR*T)
         + 1.8*np.sin(2*np.pi*2*FR*T)
         + 0.9*np.sin(2*np.pi*3*FR*T)
         + _noise(N_WINDOW, snr_db=26))
    y = (0.8*np.sin(2*np.pi*FR*T+np.pi/2)
         + 1.2*np.sin(2*np.pi*2*FR*T)
         + _noise(N_WINDOW))
    z = G + 0.3*np.sin(2*np.pi*2*FR*T) + _noise(N_WINDOW)
    return x, y, z

def gen_looseness(_=None):
    harmonics = sum(RNG.uniform(0.1,0.5)*np.sin(2*np.pi*k*FR*T) for k in range(1, 8))
    x = harmonics + _noise(N_WINDOW, snr_db=20)
    y = (0.5*harmonics + _noise(N_WINDOW))
    z = G + 0.2*harmonics + _noise(N_WINDOW)
    return x, y, z

def gen_antigravity(alpha: float | None = None):
    if alpha is None:
        alpha = RNG.uniform(0.0, 0.25)
    A1 = 0.1 + 0.9 * alpha
    prec = 0.43 * FR
    sub_fs = [0.15*FR, 0.25*FR, 0.35*FR]
    x = A1*np.sin(2*np.pi*FR*T)
    for sf in sub_fs:
        x += RNG.uniform(0.08, 0.18) * np.sin(2*np.pi*sf*T + RNG.uniform(0,2*np.pi))
    x += 0.15*np.sin(2*np.pi*prec*T) + _noise(N_WINDOW, snr_db=25)
    y = A1*np.sin(2*np.pi*FR*T+np.pi/4) + _noise(N_WINDOW)
    z = G*alpha + 0.1*np.sin(2*np.pi*FR*T) + _noise(N_WINDOW)
    return x, y, z


GENERATORS = [
    gen_normal, gen_bpfi, gen_bpfo, gen_bsf,
    gen_unbalance, gen_misalignment, gen_looseness, gen_antigravity
]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate feature matrix X (N, 54) and label vector y (N,)."""
    print("\n  Generating synthetic dataset ...")
    all_X, all_y = [], []
    for cls_idx, gen_fn in enumerate(GENERATORS):
        n = N_AG_SAMPLES if cls_idx == 7 else N_PER_CLASS
        for _ in range(n):
            x, y, z = gen_fn()
            fv = extract_features(x, y, z, shaft_rpm=SHAFT_RPM, fs=FS)
            all_X.append(fv.features)
        all_y.extend([cls_idx] * n)
        label = CLASS_NAMES[cls_idx]
        print(f"    [{cls_idx+1}/8] {label:<15} — {n} samples generated")

    return np.array(all_X), np.array(all_y)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Train Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a 7+1 class RF with balanced class weights."""
    print("\n  Training Random Forest (500 trees) ...")
    clf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Output: Classification report
# ══════════════════════════════════════════════════════════════════════════════

def save_classification_report(clf, X_test, y_test) -> None:
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4
    )

    out_path = RESULTS_DIR / "classification_report.txt"
    out_path.write_text(report, encoding="utf-8")

    print("\n" + "═"*60)
    print("  CLASSIFICATION REPORT — Random Forest (8-class, 54 features)")
    print("═"*60)
    print(report)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Graph 1: Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(clf, X_test, y_test) -> None:
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0d12")
    ax.set_facecolor("#111520")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm[i,j] < thresh else "#0a0d12")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=10)
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual", color="white", fontsize=12)
    ax.set_title("Confusion Matrix — CMS 8-Class Classifier", color="white",
                 fontsize=14, fontweight="bold", pad=16)

    ax.spines[:].set_color("#334155")
    ax.tick_params(colors="white")

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graph saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Graph 2: Feature importance (top 20)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(clf) -> None:
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_names = [FEATURE_NAMES[i] for i in indices]
    top_vals  = importances[indices]

    # Colour antigravity features purple, others blue
    ag_names = {"gli", "ser", "hcc", "cpc", "blza", "gpi"}
    bar_colours = ["#7c3aed" if n in ag_names else "#3b82f6" for n in top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0a0d12")
    ax.set_facecolor("#111520")

    bars = ax.barh(range(20), top_vals[::-1], color=bar_colours[::-1],
                   edgecolor="none", height=0.7)

    # Value labels
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color="white", fontsize=9)

    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], color="white", fontsize=10)
    ax.set_xlabel("Mean Decrease in Impurity", color="white", fontsize=12)
    ax.set_title("Top-20 Feature Importances (🟣 = Antigravity features)",
                 color="white", fontsize=14, fontweight="bold", pad=14)

    ax.spines[:].set_color("#334155")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#7c3aed", label="Antigravity feature"),
              Patch(facecolor="#3b82f6", label="Baseline feature")]
    ax.legend(handles=legend, facecolor="#161b2a", labelcolor="white",
              edgecolor="#334155", fontsize=10)

    plt.tight_layout()
    out = RESULTS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graph saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Graph 3: ROC curves (one-vs-rest)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(clf, X_test, y_test) -> None:
    y_score = clf.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0d12")
    ax.set_facecolor("#111520")

    for i, (name, colour) in enumerate(zip(CLASS_NAMES, COLOURS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == "antigravity" else 1.8
        ls = "-"
        ax.plot(fpr, tpr, lw=lw, color=colour, linestyle=ls,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "w--", lw=1, alpha=0.4)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Curves — One-vs-Rest (8-class)", color="white",
                 fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="lower right", facecolor="#161b2a", labelcolor="white",
              edgecolor="#334155", fontsize=10)
    ax.spines[:].set_color("#334155")
    ax.tick_params(colors="white")
    ax.grid(True, color="#1e293b", linewidth=0.8)

    plt.tight_layout()
    out = RESULTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graph saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Graph 4: Antigravity feature distributions per class
# ══════════════════════════════════════════════════════════════════════════════

def plot_ag_feature_distributions(X: np.ndarray, y: np.ndarray) -> None:
    ag_feat_names = ["gli", "ser", "hcc", "cpc", "blza", "gpi"]
    ag_indices    = [FEATURE_NAMES.index(n) for n in ag_feat_names]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0a0d12")
    fig.suptitle("Antigravity Feature Distributions per Fault Class",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    for ax, feat_name, feat_idx in zip(axes.flat, ag_feat_names, ag_indices):
        ax.set_facecolor("#111520")
        for cls_idx, (cls_name, colour) in enumerate(zip(CLASS_NAMES, COLOURS)):
            vals = X[y == cls_idx, feat_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lw = 2.5 if cls_name == "antigravity" else 1.2
            ax.hist(vals, bins=30, alpha=0.55, color=colour,
                    label=cls_name, density=True, linewidth=lw)

        ax.set_title(feat_name.upper(), color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334155")
        ax.set_xlabel("Value", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Density", color="#94a3b8", fontsize=9)

    # Shared legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLOURS[i], label=CLASS_NAMES[i])
                      for i in range(len(CLASS_NAMES))]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               facecolor="#161b2a", labelcolor="white", edgecolor="#334155",
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = RESULTS_DIR / "antigravity_feature_dist.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graph saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═"*60)
    print("  CMS DEMO — Generating Scores, Graphs and Results")
    print("═"*60)
    print(f"  Output directory: {RESULTS_DIR}")

    # 1. Build synthetic dataset
    X, y = build_dataset()
    print(f"\n  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Classes: {CLASS_NAMES}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Train
    clf = train_model(X_train, y_train)

    # 4. Save classification report (console + file)
    save_classification_report(clf, X_test, y_test)

    # 5. Generate graphs
    print("\n  Generating graphs ...")
    plot_confusion_matrix(clf, X_test, y_test)
    plot_feature_importance(clf)
    plot_roc_curves(clf, X_test, y_test)
    plot_ag_feature_distributions(X, y)

    print("\n" + "═"*60)
    print("  ALL DONE!")
    print("═"*60)
    print(f"  Results saved to: {RESULTS_DIR}")
    print("  Files:")
    for f in sorted(RESULTS_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<40} {size_kb:>5} KB")
    print()


if __name__ == "__main__":
    main()
