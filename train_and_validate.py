#!/usr/bin/env python3
"""
Terrain Recognition - Training, Feature Importance & Validation Script
=======================================================================
Pipeline:
  1. Build windowed dataset from acc+gyro only (no magnetometer).
  2. Train all classifiers on full feature set (27 features).
  3. Feature importance via Random Forest MDI + permutation importance.
  4. Select top-K most important features.
  5. Retrain all classifiers on reduced feature set.
  6. Compare full vs reduced performance and export Pico model.
"""

import os
import glob
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
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
DATA_ROOT       = os.path.dirname(os.path.abspath(__file__))
TERRAIN_CLASSES = ["Asphalt", "Cobblestone", "Gravel"]
LABEL_MAP       = {name: i for i, name in enumerate(TERRAIN_CLASSES)}

# 6-DOF only — magnetometer intentionally excluded
ACC_COLS    = ["nicla_accX",  "nicla_accY",  "nicla_accZ"]
GYRO_COLS   = ["nicla_gyroX", "nicla_gyroY", "nicla_gyroZ"]
SENSOR_COLS = ACC_COLS + GYRO_COLS

WINDOW_SIZE  = 100   # ~1 s at 100 Hz
WINDOW_STEP  = 50    # 50 % overlap
RANDOM_STATE = 42
N_FOLDS      = 5

# How many top features to keep for the reduced model
TOP_K = 10

OUTPUT_DIR = DATA_ROOT


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _detect_sep(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        header = f.readline()
    return "\t" if "\t" in header else ","


def load_file(path: str) -> pd.DataFrame | None:
    sep = _detect_sep(path)
    try:
        df = pd.read_csv(path, sep=sep,
                         usecols=lambda c: c in SENSOR_COLS,
                         low_memory=False)
        for col in SENSOR_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)
        return df if len(df) >= WINDOW_SIZE else None
    except Exception as exc:
        print(f"    [WARN] {os.path.basename(path)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Feature extraction  (27 features, pure Python-compatible)
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    [f"{col}_{s}" for col in SENSOR_COLS for s in ("mean", "std", "rms", "p2p")]
    + ["acc_mag_mean", "acc_mag_std", "acc_mag_p2p"]
)


def extract_features(window: np.ndarray) -> list[float]:
    """window: (WINDOW_SIZE, 6) → 27 features."""
    feats = []
    for col in range(window.shape[1]):
        x = window[:, col]
        feats.append(float(np.mean(x)))
        feats.append(float(np.std(x)))
        feats.append(float(math.sqrt(float(np.mean(x ** 2)))))
        feats.append(float(np.max(x) - np.min(x)))
    mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
    feats += [float(np.mean(mag)), float(np.std(mag)),
              float(np.max(mag) - np.min(mag))]
    return feats


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
                X.append(extract_features(data[start: start + WINDOW_SIZE]))
                y.append(LABEL_MAP[terrain])
                n_win += 1
        print(f"  {terrain:>12s} : {len(files)} files → {n_win:>6d} windows")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\n  Total : {X.shape[0]} windows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------
def _fmt_cm(cm: np.ndarray, labels: list[str]) -> str:
    col_w = max(len(l) for l in labels) + 2
    lines = [" " * col_w + "".join(f"{l:>{col_w}}" for l in labels) + "  (pred)"]
    for i, row in enumerate(cm):
        lines.append(f"{labels[i]:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(lines)


def make_classifiers() -> dict:
    return {
        "Nearest Centroid":  NearestCentroid(),
        "Decision Tree":     DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest":     RandomForestClassifier(n_estimators=100, max_depth=10,
                                                    n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                         random_state=RANDOM_STATE),
        "SVM (RBF)":         SVC(kernel="rbf", C=10, gamma="scale",
                                  random_state=RANDOM_STATE),
    }


def evaluate_classifiers(X_scaled: np.ndarray, y: np.ndarray,
                          label: str = "") -> dict:
    cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sep = "=" * 64
    tag = f"  [{label}]" if label else ""
    print(f"\n{sep}")
    print(f"  CLASSIFIER PERFORMANCE — {N_FOLDS}-fold CV{tag}")
    print(sep)

    results = {}
    for name, clf in make_classifiers().items():
        print(f"\n▶  {name}")
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv, n_jobs=-1)
        acc    = accuracy_score(y, y_pred)
        print(f"   Accuracy : {acc * 100:.2f} %")
        print(classification_report(y, y_pred, target_names=TERRAIN_CLASSES,
                                    digits=3, zero_division=0))
        print(_fmt_cm(confusion_matrix(y, y_pred), TERRAIN_CLASSES))
        results[name] = {"accuracy": acc, "y_pred": y_pred,
                          "cm": confusion_matrix(y, y_pred)}
    return results


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def analyze_feature_importance(X: np.ndarray, y: np.ndarray,
                                scaler: StandardScaler) -> np.ndarray:
    """
    Returns importance array (27,) combining:
      - Random Forest MDI importance (mean decrease in impurity)
      - Permutation importance on a held-out 20 % split
    Both are normalised to [0,1] and averaged for a robust ranking.
    """
    from sklearn.model_selection import train_test_split

    X_scaled = scaler.transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ── MDI importance (Random Forest) ──────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    mdi = rf.feature_importances_
    mdi_norm = mdi / mdi.sum()

    # ── Permutation importance ───────────────────────────────────────────────
    perm  = permutation_importance(rf, X_te, y_te, n_repeats=20,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    perm_imp  = np.maximum(perm.importances_mean, 0)   # clip negatives
    perm_norm = perm_imp / (perm_imp.sum() + 1e-12)

    # ── Combined score ───────────────────────────────────────────────────────
    combined = 0.5 * mdi_norm + 0.5 * perm_norm
    combined /= combined.sum()

    # ── Print ranking ────────────────────────────────────────────────────────
    ranked = np.argsort(combined)[::-1]
    print(f"\n{'='*64}")
    print(f"  FEATURE IMPORTANCE RANKING  (MDI + Permutation, combined)")
    print(f"{'='*64}")
    print(f"  {'#':>3}  {'Feature':<30}  {'MDI':>7}  {'Perm':>7}  {'Combined':>9}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*9}")
    cumulative = 0.0
    for rank, idx in enumerate(ranked):
        cumulative += combined[idx]
        marker = " ◀ top" if rank < TOP_K else ""
        print(f"  {rank+1:>3}. {FEATURE_NAMES[idx]:<30}  "
              f"{mdi_norm[idx]:>6.1%}  {perm_norm[idx]:>6.1%}  "
              f"{combined[idx]:>8.1%}  (cum {cumulative:.0%}){marker}")

    return combined, ranked


def save_importance_plot(combined: np.ndarray, ranked: np.ndarray,
                         out_dir: str, top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if i < top_k else "#95a5a6" for i in range(len(ranked))]
    names_ranked = [FEATURE_NAMES[i] for i in ranked]
    vals_ranked  = combined[ranked]

    bars = ax.barh(names_ranked[::-1], vals_ranked[::-1] * 100,
                   color=colors[::-1], edgecolor="white")
    ax.axhline(len(ranked) - top_k - 0.5, color="red", lw=1.5,
               linestyle="--", label=f"Top-{top_k} cutoff")
    ax.set_xlabel("Combined Importance (%)")
    ax.set_title("Feature Importance — MDI + Permutation (Random Forest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved → {path}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------
def save_comparison_plot(results_full: dict, results_top: dict,
                         out_dir: str, top_k: int) -> None:
    names  = list(results_full.keys())
    acc_full = [results_full[n]["accuracy"] * 100 for n in names]
    acc_top  = [results_top[n]["accuracy"]  * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, acc_full, w, label="All 27 features",
                color="#3498db", edgecolor="white")
    b2 = ax.bar(x + w/2, acc_top,  w, label=f"Top {top_k} features",
                color="#e74c3c", edgecolor="white")
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(50, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Full features vs Top-{top_k} features")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "full_vs_top_features.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved         → {path}")


def save_confusion_matrices(results: dict, label: str, out_dir: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay(res["cm"], display_labels=TERRAIN_CLASSES).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\n{res['accuracy']*100:.1f} %", fontsize=9)
    fig.suptitle(f"Confusion Matrices — {label}", fontsize=12, y=1.02)
    plt.tight_layout()
    fname = "cm_full.png" if "full" in label.lower() else "cm_top.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrices saved      → {path}")


# ---------------------------------------------------------------------------
# Export Pico model
# ---------------------------------------------------------------------------
def export_model_params(scaler_top: StandardScaler, X_top: np.ndarray,
                        y: np.ndarray, top_indices: list[int],
                        top_names: list[str], out_dir: str) -> None:
    nc = NearestCentroid()
    nc.fit(X_top, y)

    lines = [
        "# ============================================================",
        "# Auto-generated by train_and_validate.py",
        f"# Uses only top-{TOP_K} features (acc+gyro, no magnetometer)",
        "# Copy to Raspberry Pi Pico alongside terrain_recognition_pico.py",
        "# ============================================================",
        "",
        f"CLASSES         = {TERRAIN_CLASSES}",
        f"WINDOW_SIZE     = {WINDOW_SIZE}",
        f"N_FEATURES      = {len(top_names)}",
        f"FEATURE_INDICES = {top_indices}   # indices into the full 27-feature vector",
        f"FEATURE_NAMES   = {top_names}",
        "",
        "# StandardScaler parameters (fitted on top features only)",
        f"SCALER_MEAN = {scaler_top.mean_.tolist()}",
        f"SCALER_STD  = {scaler_top.scale_.tolist()}",
        "",
        "# Nearest-Centroid class centroids (one row per class)",
        f"CENTROIDS   = {nc.centroids_.tolist()}",
    ]
    path = os.path.join(out_dir, "model_params.py")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Pico model params saved       → {path}")
    acc_full_fit = accuracy_score(y, nc.predict(X_top))
    print(f"  Nearest Centroid (full-fit)   : {acc_full_fit*100:.2f} %  (upper bound)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("  Terrain Recognition — Feature Importance & Validation")
    print("=" * 64)

    # 1. Build dataset
    X, y = build_dataset()

    # 2. Full-feature evaluation
    scaler_full = StandardScaler()
    X_full      = scaler_full.fit_transform(X)
    results_full = evaluate_classifiers(X_full, y, label="all 27 features")

    best_full = max(results_full, key=lambda k: results_full[k]["accuracy"])
    print(f"\n  Best (full) : {best_full}  "
          f"({results_full[best_full]['accuracy']*100:.2f} %)")

    # 3. Feature importance
    print("\n" + "=" * 64)
    print("  COMPUTING FEATURE IMPORTANCE …")
    print("=" * 64)
    combined, ranked = analyze_feature_importance(X, y, scaler_full)
    save_importance_plot(combined, ranked, OUTPUT_DIR, TOP_K)

    # 4. Reduced dataset with top-K features
    top_indices = sorted(ranked[:TOP_K].tolist())   # keep original order
    top_names   = [FEATURE_NAMES[i] for i in top_indices]

    print(f"\n  Selected top-{TOP_K} features (by index):")
    for i, (idx, name) in enumerate(zip(top_indices, top_names)):
        print(f"    {i+1:>2}. [{idx:>2}] {name}")

    X_top_raw    = X[:, top_indices]
    scaler_top   = StandardScaler()
    X_top_scaled = scaler_top.fit_transform(X_top_raw)

    # 5. Reduced-feature evaluation
    results_top = evaluate_classifiers(X_top_scaled, y,
                                       label=f"top-{TOP_K} features")

    best_top = max(results_top, key=lambda k: results_top[k]["accuracy"])
    print(f"\n  Best (top-{TOP_K}) : {best_top}  "
          f"({results_top[best_top]['accuracy']*100:.2f} %)")

    # 6. Summary table
    print("\n" + "=" * 64)
    print(f"  SUMMARY — Full (27) vs Top-{TOP_K} features")
    print("=" * 64)
    print(f"  {'Classifier':<22}  {'Full (27)':>10}  {'Top-'+str(TOP_K):>8}  {'Δ':>6}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*6}")
    for name in results_full:
        af = results_full[name]["accuracy"] * 100
        at = results_top[name]["accuracy"]  * 100
        print(f"  {name:<22}  {af:>9.2f}%  {at:>7.2f}%  {at-af:>+5.2f}%")

    # 7. Plots
    save_confusion_matrices(results_full, "Full 27 features", OUTPUT_DIR)
    save_confusion_matrices(results_top,  f"Top-{TOP_K} features", OUTPUT_DIR)
    save_comparison_plot(results_full, results_top, OUTPUT_DIR, TOP_K)

    # 8. Export Pico model (uses top-K features)
    print("\n" + "=" * 64)
    print("  Exporting Pico model (Nearest Centroid, top-K features)")
    print("=" * 64)
    export_model_params(scaler_top, X_top_scaled, y,
                        top_indices, top_names, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
