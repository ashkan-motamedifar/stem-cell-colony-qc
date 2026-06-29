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
            f"No cached embeddings for {backbone}. Run: python scripts/probe_features.py --backbone {backbone}"
        )
    data = np.load(cache, allow_pickle=True)
    if [Path(p) for p in data["paths"]] != list(paths):
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

    parts, y = [], None
    for backbone in args.backbones:
        X, y_ = load_cached(backbone, paths)
        parts.append(X)
        y = y_ if y is None else y
        print(f"  {backbone:10s}: {X.shape}")

    X = np.concatenate(parts, axis=1)
    print(f"  concatenated: {X.shape}")

    folds = stratified_kfold(paths, n_splits=5, seed=args.seed)[: args.folds]

    per_fold = []
    for fold_idx, (train_paths, val_paths) in enumerate(folds):
        train_idx = [paths.index(p) for p in train_paths]
        val_idx = [paths.index(p) for p in val_paths]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=args.C, max_iter=4000, solver="lbfgs", random_state=args.seed + fold_idx)
        clf.fit(scaler.transform(X_tr), y_tr)
        y_prob = clf.predict_proba(scaler.transform(X_va))[:, 1]
        m = compute_metrics(y_va, y_prob)
        per_fold.append(m)
        save_predictions(PRED_CSV, val_paths, y_va.tolist(), y_prob.tolist(), fold=fold_idx, seed=args.seed, model=tag)
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
