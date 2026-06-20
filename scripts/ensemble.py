"""Ensemble of frozen foundation-model features.

Concatenate cached embeddings from ResNet50 + DINOv2 + CLIP into one
3328-dim vector per image, train a logistic regression with 5-fold CV.
This is the "proposed method": no domain training, no augmentation, just
a single LR over the union of pretrained visual representations.

Run probe_features.py for each backbone first to populate the cache.

Usage:
    python scripts/ensemble.py                       # all 3
    python scripts/ensemble.py --backbones resnet50 dinov2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import list_images, stratified_kfold
from src.metrics import aggregate_folds, compute_metrics, save_predictions

CACHE_DIR = PROJECT / "data" / "processed"
RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"


def load_cached(backbone: str, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    cache = CACHE_DIR / f"embeddings_{backbone}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached embeddings for {backbone}. "
            f"Run: python scripts/probe_features.py --backbone {backbone}"
        )
    data = np.load(cache, allow_pickle=True)
    cached_paths = [Path(p) for p in data["paths"]]
    if cached_paths != list(paths):
        raise RuntimeError(f"Cached paths differ for {backbone}. Re-run probe.")
    return data["X"], data["y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["resnet50", "dinov2", "clip"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or "ensemble_" + "+".join(args.backbones)
    paths = list_images()

    parts = []
    y_ref = None
    for backbone in args.backbones:
        X, y = load_cached(backbone, paths)
        parts.append(X)
        y_ref = y if y_ref is None else y_ref
        print(f"  {backbone:10s}: {X.shape}")

    X = np.concatenate(parts, axis=1)
    y = y_ref
    print(f"  concatenated: {X.shape}")

    folds = stratified_kfold(paths, n_splits=5, seed=args.seed)[: args.folds]

    per_fold = []
    for fold_idx, (train_paths, val_paths) in enumerate(folds):
        train_idx = [paths.index(p) for p in train_paths]
        val_idx = [paths.index(p) for p in val_paths]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        clf = LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs",
                                 random_state=args.seed + fold_idx)
        clf.fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_va_s)[:, 1]
        m = compute_metrics(y_va, y_prob)
        per_fold.append(m)
        save_predictions(PRED_CSV, val_paths, y_va.tolist(), y_prob.tolist(),
                         fold=fold_idx, seed=args.seed, model=tag)
        print(f"  fold {fold_idx}: acc={m.accuracy:.3f} f1={m.f1:.3f} auc={m.auc:.3f}")

    agg = aggregate_folds(per_fold)
    print("\n=== Aggregated (mean ± std) ===")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, std = agg[k]
        print(f"  {k:10s}: {mean:.3f} ± {std:.3f}")

    out = RESULTS_DIR / "tables" / f"{tag}_summary.json"
    out.write_text(json.dumps({
        "model": tag,
        "backbones": args.backbones,
        "config": vars(args),
        "embedding_dim": int(X.shape[1]),
        "per_fold": [m.as_dict() for m in per_fold],
        "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
