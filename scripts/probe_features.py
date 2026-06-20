"""Frozen-backbone + logistic regression probe with 5-fold stratified CV.

One script, three backbones — extract once, then cross-validate a linear
classifier on the embeddings. This is the SHARED pipeline for ResNet50,
DINOv2, and CLIP-image so they're directly comparable.

Usage:
    python scripts/probe_features.py --backbone resnet50
    python scripts/probe_features.py --backbone dinov2
    python scripts/probe_features.py --backbone clip
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

from src.data import get_device, list_images, stratified_kfold
from src.embeddings import embed_clip_image, embed_dinov2, embed_resnet50
from src.metrics import aggregate_folds, compute_metrics, save_predictions

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
CACHE_DIR = PROJECT / "data" / "processed"


def extract_or_load(backbone: str, device: str, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Cache embeddings so reruns are instant."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"embeddings_{backbone}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_paths = [Path(p) for p in data["paths"]]
        if cached_paths == list(paths):
            print(f"  Loaded cached embeddings from {cache}")
            return data["X"], data["y"]
        print(f"  Cache stale (different path set), recomputing.")

    if backbone == "resnet50":
        _, X, y = embed_resnet50(paths, device=device, batch_size=16)
    elif backbone == "dinov2":
        _, X, y = embed_dinov2(paths, device=device, batch_size=8)
    elif backbone == "clip":
        result = embed_clip_image(paths, device=device, batch_size=16)
        _, X, y = result[0], result[1], result[2]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    np.savez(cache, paths=[str(p) for p in paths], X=X, y=y)
    print(f"  Saved embeddings to {cache} (shape={X.shape})")
    return X, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=["resnet50", "dinov2", "clip"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0, help="LR regularization (inverse)")
    p.add_argument("--no-scale", action="store_true",
                   help="Skip StandardScaler (embeddings are already L2-normalized)")
    p.add_argument("--tag", type=str, default=None,
                   help="Defaults to '<backbone>_probe'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or f"{args.backbone}_probe"
    summary_path = RESULTS_DIR / "tables" / f"{tag}_summary.json"
    device = get_device()
    print(f"Backbone: {args.backbone} | tag: {tag} | device: {device}")

    paths = list_images()
    print(f"\nExtracting embeddings for {len(paths)} images...")
    X, y = extract_or_load(args.backbone, device, paths)
    print(f"Embedding shape: {X.shape} | label balance: good={int(y.sum())}/bad={int(len(y)-y.sum())}")

    folds = stratified_kfold(paths, n_splits=5, seed=args.seed)
    folds_to_run = folds[: args.folds]

    per_fold = []
    for fold_idx, (train_paths, val_paths) in enumerate(folds_to_run):
        train_idx = [paths.index(p) for p in train_paths]
        val_idx = [paths.index(p) for p in val_paths]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        if not args.no_scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_va_s = scaler.transform(X_va)
        else:
            X_tr_s, X_va_s = X_tr, X_va

        clf = LogisticRegression(
            C=args.C, max_iter=2000, solver="lbfgs", random_state=args.seed + fold_idx,
        )
        clf.fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_va_s)[:, 1]
        metrics = compute_metrics(y_va, y_prob)
        per_fold.append(metrics)
        save_predictions(
            PRED_CSV, val_paths, y_va.tolist(), y_prob.tolist(),
            fold=fold_idx, seed=args.seed, model=tag,
        )
        print(f"  fold {fold_idx}: acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
              f"auc={metrics.auc:.3f}")

    agg = aggregate_folds(per_fold)
    print("\n=== Aggregated (mean ± std across folds) ===")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, std = agg[k]
        print(f"  {k:10s}: {mean:.3f} ± {std:.3f}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": tag,
        "backbone": args.backbone,
        "config": vars(args),
        "device": device,
        "embedding_dim": int(X.shape[1]),
        "per_fold": [m.as_dict() for m in per_fold],
        "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
