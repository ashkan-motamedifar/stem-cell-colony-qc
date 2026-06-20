"""How many labels do you actually need?

For each backbone (resnet50 / dinov2 / clip / ensemble), train an LR on
varying fractions of the train set and evaluate on the same 5-fold CV val
splits. Plots accuracy vs train set size — the answer to "would 30 labels
have been enough?"

Reads cached embeddings from data/processed/. Run probe_features.py first.

Usage:
    python scripts/data_efficiency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import list_images, parse_label, stratified_kfold

CACHE_DIR = PROJECT / "data" / "processed"
RESULTS_DIR = PROJECT / "results"

BACKBONES = ["resnet50", "dinov2", "clip", "ensemble"]
COLORS = {"resnet50": "#1f77b4", "dinov2": "#2ca02c",
          "clip": "#9467bd", "ensemble": "#ff7f0e"}
LABELS = {"resnet50": "ResNet50", "dinov2": "DINOv2",
          "clip": "CLIP", "ensemble": "Ensemble"}

# Fraction of available train pool to use each step
FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_backbone(backbone: str, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    if backbone == "ensemble":
        parts = []
        for b in ["resnet50", "dinov2", "clip"]:
            data = np.load(CACHE_DIR / f"embeddings_{b}.npz", allow_pickle=True)
            parts.append(data["X"])
        X = np.concatenate(parts, axis=1)
        y = data["y"]
    else:
        data = np.load(CACHE_DIR / f"embeddings_{backbone}.npz", allow_pickle=True)
        X, y = data["X"], data["y"]
    return X, y


def subsample_stratified(idx: np.ndarray, y: np.ndarray, frac: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Return a stratified subsample of idx according to labels y."""
    keep = []
    for cls in np.unique(y[idx]):
        cls_idx = idx[y[idx] == cls]
        n = max(1, int(round(len(cls_idx) * frac)))
        keep.append(rng.choice(cls_idx, size=n, replace=False))
    return np.concatenate(keep)


def main() -> None:
    paths = list_images()
    folds = stratified_kfold(paths, n_splits=5, seed=42)

    curves = {}
    for backbone in BACKBONES:
        X, y = load_backbone(backbone, paths)
        rng = np.random.default_rng(42)
        means, stds, ns = [], [], []
        for frac in FRACTIONS:
            fold_accs = []
            n_train_for_log = None
            for fold_idx, (train_paths, val_paths) in enumerate(folds):
                train_idx = np.array([paths.index(p) for p in train_paths])
                val_idx = np.array([paths.index(p) for p in val_paths])
                sub_idx = subsample_stratified(train_idx, y, frac, rng)
                if fold_idx == 0:
                    n_train_for_log = len(sub_idx)
                X_tr, y_tr = X[sub_idx], y[sub_idx]
                X_va, y_va = X[val_idx], y[val_idx]
                scaler = StandardScaler().fit(X_tr)
                clf = LogisticRegression(C=1.0, max_iter=4000, solver="lbfgs",
                                          random_state=42 + fold_idx)
                clf.fit(scaler.transform(X_tr), y_tr)
                y_pred = clf.predict(scaler.transform(X_va))
                fold_accs.append((y_pred == y_va).mean())
            mean, std = float(np.mean(fold_accs)), float(np.std(fold_accs))
            means.append(mean); stds.append(std); ns.append(n_train_for_log)
            print(f"  {backbone:10s} frac={frac:.2f} n_train={n_train_for_log:3d} "
                  f"acc={mean:.3f} ± {std:.3f}")
        curves[backbone] = {"n": ns, "mean": means, "std": stds}

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for backbone, c in curves.items():
        n = np.array(c["n"])
        m = np.array(c["mean"])
        s = np.array(c["std"])
        ax.plot(n, m, marker="o", label=LABELS[backbone], color=COLORS[backbone])
        ax.fill_between(n, m - s, m + s, alpha=0.15, color=COLORS[backbone])
    ax.axhline(0.65, color="#d62728", linestyle="--", linewidth=0.8,
               label="VGG13 from scratch (215 imgs)")
    ax.axhline(0.89, color="red", linestyle=":", linewidth=0.8,
               label="paper claim (215 imgs)")
    ax.set_xlabel("Training set size (images)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Data efficiency: accuracy vs training set size\n"
                 "5-fold CV, 215-image train pool per fold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    out = RESULTS_DIR / "figures" / "data_efficiency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    (RESULTS_DIR / "tables" / "data_efficiency.json").write_text(
        json.dumps(curves, indent=2)
    )


if __name__ == "__main__":
    main()
