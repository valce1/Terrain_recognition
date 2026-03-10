#!/usr/bin/env python3
"""
Terrain Recognition - Training & Validation Script
===================================================
Loads IMU data from 3 terrain folders (Asphalt, Cobblestone, Gravel),
extracts statistical features from sliding windows, trains and cross-validates
multiple classifiers, and exports model parameters for Raspberry Pi Pico
(MicroPython).

Feature set uses only accelerometer + gyroscope (6 axes) so the pipeline
remains valid when the sensor is moved from handlebars to behind-the-saddle,
or when the magnetometer is no longer available.
"""

import os
import glob
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT      = os.path.dirname(os.path.abspath(__file__))
TERRAIN_CLASSES = ["Asphalt", "Cobblestone", "Gravel"]
LABEL_MAP       = {name: i for i, name in enumerate(TERRAIN_CLASSES)}

# 6-DOF IMU columns (acc + gyro only — magnetometer intentionally excluded)
ACC_COLS  = ["nicla_accX",  "nicla_accY",  "nicla_accZ"]
GYRO_COLS = ["nicla_gyroX", "nicla_gyroY", "nicla_gyroZ"]
SENSOR_COLS = ACC_COLS + GYRO_COLS

WINDOW_SIZE = 100   # samples per window  (~1 s at 100 Hz)
WINDOW_STEP = 50    # 50 % overlap

RANDOM_STATE = 42
N_FOLDS      = 5

OUTPUT_DIR   = DATA_ROOT   # where to save plots and model_params.py


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _detect_sep(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        header = f.readline()
    return "\t" if "\t" in header else ","


def load_file(path: str) -> pd.DataFrame | None:
    sep = _detect_sep(path)
    try:
        df = pd.read_csv(
            path, sep=sep,
            usecols=lambda c: c in SENSOR_COLS,
            low_memory=False
        )
        for col in SENSOR_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)
        return df if len(df) >= WINDOW_SIZE else None
    except Exception as exc:
        print(f"    [WARN] Could not load {os.path.basename(path)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Feature extraction  (27 features, no sklearn dependency at inference time)
# ---------------------------------------------------------------------------
def extract_features(window: np.ndarray) -> list[float]:
    """
    Input : window  shape (WINDOW_SIZE, 6)  — [accX,accY,accZ,gyroX,gyroY,gyroZ]
    Output: 1-D feature vector (list of floats)

    Features per axis (6 axes × 4 = 24):
        mean, std, rms, peak-to-peak
    Global features (3):
        total-acc-magnitude mean, std, peak-to-peak
    Total: 27 features
    """
    feats = []
    for col in range(window.shape[1]):
        x = window[:, col]
        feats.append(float(np.mean(x)))
        feats.append(float(np.std(x)))
        feats.append(float(math.sqrt(float(np.mean(x ** 2)))))     # RMS
        feats.append(float(np.max(x) - np.min(x)))                 # peak-to-peak

    # Total acceleration magnitude
    mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
    feats.append(float(np.mean(mag)))
    feats.append(float(np.std(mag)))
    feats.append(float(np.max(mag) - np.min(mag)))

    return feats


FEATURE_NAMES = (
    [f"{col}_{s}" for col in SENSOR_COLS for s in ("mean", "std", "rms", "p2p")]
    + ["acc_mag_mean", "acc_mag_std", "acc_mag_p2p"]
)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    print(f"\nBuilding dataset  (window={WINDOW_SIZE}, step={WINDOW_STEP})...\n")

    for terrain in TERRAIN_CLASSES:
        folder = os.path.join(DATA_ROOT, terrain)
        files  = sorted(glob.glob(os.path.join(folder, "*.csv")))
        n_win  = 0

        for path in files:
            df = load_file(path)
            if df is None:
                continue
            data = df[SENSOR_COLS].values.astype(np.float32)

            for start in range(0, len(data) - WINDOW_SIZE, WINDOW_STEP):
                feats = extract_features(data[start : start + WINDOW_SIZE])
                X.append(feats)
                y.append(LABEL_MAP[terrain])
                n_win += 1

        label = LABEL_MAP[terrain]
        print(f"  {terrain:>12s} : {len(files)} files → {n_win:>6d} windows")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\n  Total : {X.shape[0]} windows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _fmt_cm(cm: np.ndarray, labels: list[str]) -> str:
    col_w = max(len(l) for l in labels) + 2
    lines = [" " * col_w + "".join(f"{l:>{col_w}}" for l in labels) + "  (predicted)"]
    for i, row in enumerate(cm):
        lines.append(f"{labels[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(lines)


def evaluate_classifiers(X: np.ndarray, y: np.ndarray) -> dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    classifiers = {
        "Nearest Centroid":   NearestCentroid(),
        "Decision Tree":      DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest":      RandomForestClassifier(n_estimators=100, max_depth=10,
                                                     n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                          random_state=RANDOM_STATE),
        "SVM (RBF)":          SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
    }

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  CLASSIFIER PERFORMANCE  ({N_FOLDS}-fold Stratified Cross-Validation)")
    print(sep)

    results = {}
    for name, clf in classifiers.items():
        print(f"\n▶  {name}")
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv, n_jobs=-1)
        acc    = accuracy_score(y, y_pred)
        print(f"   Accuracy : {acc * 100:.2f} %")
        print(classification_report(y, y_pred, target_names=TERRAIN_CLASSES,
                                    digits=3, zero_division=0))
        cm = confusion_matrix(y, y_pred)
        print(_fmt_cm(cm, TERRAIN_CLASSES))
        results[name] = {"accuracy": acc, "y_pred": y_pred, "cm": cm, "clf": clf}

    return results, scaler, X_scaled


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def save_confusion_matrices(results: dict, y: np.ndarray, out_dir: str) -> None:
    n  = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"],
                                     display_labels=TERRAIN_CLASSES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\n{res['accuracy']*100:.1f} %", fontsize=10)

    fig.suptitle("Terrain Recognition — Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion-matrix plot saved → {path}")


def save_accuracy_bar(results: dict, out_dir: str) -> None:
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(names)))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, accs, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f %%", padding=4)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Terrain Recognition — Classifier Comparison")
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy bar-chart saved    → {path}")


# ---------------------------------------------------------------------------
# Export model parameters for MicroPython
# ---------------------------------------------------------------------------
def export_model_params(scaler: StandardScaler, X_scaled: np.ndarray,
                        y: np.ndarray, out_dir: str) -> None:
    """
    Fits a Nearest Centroid on the full dataset and exports the normalisation
    parameters (mean, std) and class centroids as a plain Python file that can
    be copied onto the Pico.
    """
    nc = NearestCentroid()
    nc.fit(X_scaled, y)

    lines = [
        "# ============================================================",
        "# Auto-generated by train_and_validate.py",
        "# Copy this file to the Raspberry Pi Pico alongside",
        "# terrain_recognition_pico.py",
        "# ============================================================",
        "",
        f"CLASSES     = {TERRAIN_CLASSES}",
        f"WINDOW_SIZE = {WINDOW_SIZE}",
        f"N_FEATURES  = {len(FEATURE_NAMES)}",
        "",
        "# StandardScaler parameters",
        f"SCALER_MEAN = {scaler.mean_.tolist()}",
        f"SCALER_STD  = {scaler.scale_.tolist()}",
        "",
        "# Nearest-Centroid class centroids (one row per class, in CLASSES order)",
        f"CENTROIDS   = {nc.centroids_.tolist()}",
        "",
        "# Feature names (for reference)",
        f"FEATURE_NAMES = {FEATURE_NAMES}",
    ]

    path = os.path.join(out_dir, "model_params.py")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Model parameters saved      → {path}")
    print("\n  ── Nearest Centroid accuracy on full dataset ──")
    y_pred_full = nc.predict(X_scaled)
    print(f"  Accuracy : {accuracy_score(y, y_pred_full) * 100:.2f} %  "
          "(full-dataset fit — not a CV estimate, upper bound)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("  Terrain Recognition — Training & Validation")
    print("=" * 64)

    X, y = build_dataset()

    results, scaler, X_scaled = evaluate_classifiers(X, y)

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_acc  = results[best_name]["accuracy"]

    print("\n" + "=" * 64)
    print(f"  Best classifier : {best_name}  ({best_acc * 100:.2f} %)")
    print("=" * 64)

    save_confusion_matrices(results, y, OUTPUT_DIR)
    save_accuracy_bar(results, OUTPUT_DIR)

    print("\n" + "=" * 64)
    print("  Exporting Nearest Centroid model for Raspberry Pi Pico")
    print("=" * 64)
    export_model_params(scaler, X_scaled, y, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
