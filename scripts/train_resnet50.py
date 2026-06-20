"""Fine-tune ImageNet-pretrained ResNet-50 on the hPSC colony dataset.

For fair comparison with VGG13 we use the same paper-faithful preprocessing
(histogram equalization, rotation+crop) by default, but expose --no-equalize
to test how much that preprocessing matters once you start from ImageNet weights.

Usage:
    python scripts/train_resnet50.py                            # full 5-fold CV
    python scripts/train_resnet50.py --folds 1 --epochs 5       # sanity
    python scripts/train_resnet50.py --no-equalize              # plain ImageNet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import (
    MamaevaDataset,
    get_device,
    imagenet_transform,
    list_images,
    paper_transform,
    stratified_kfold,
)
from src.metrics import aggregate_folds, format_results_row, save_predictions
from src.train import TrainConfig, train_one_fold

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
SUMMARY_JSON = RESULTS_DIR / "tables" / "resnet50_summary.json"


def build_resnet50() -> nn.Module:
    """ImageNet-pretrained ResNet-50 with a fresh 2-class head."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-equalize", action="store_true",
                   help="Use plain ImageNet preprocessing instead of paper-faithful.")
    p.add_argument("--tag", type=str, default="resnet50_paper")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    transform_fn = imagenet_transform if args.no_equalize else paper_transform
    if args.no_equalize and args.tag == "resnet50_paper":
        args.tag = "resnet50_plain"
    print(f"Device: {device}")
    print(f"Tag: {args.tag} | transform: {'imagenet' if args.no_equalize else 'paper'}")
    print(f"Config: folds={args.folds} epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} wd={args.weight_decay} seed={args.seed}")

    paths = list_images()
    all_folds = stratified_kfold(paths, n_splits=5, seed=args.seed)
    folds_to_run = all_folds[: args.folds]

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
        log_every=max(args.epochs // 5, 1),
    )

    per_fold = []
    for fold_idx, (train_paths, val_paths) in enumerate(folds_to_run):
        print(f"\n=== Fold {fold_idx+1}/{args.folds} | "
              f"train={len(train_paths)} val={len(val_paths)} ===")
        train_ds = MamaevaDataset(train_paths, transform=transform_fn(train=True))
        val_ds = MamaevaDataset(val_paths, transform=transform_fn(train=False))
        metrics, y_true, y_prob, _ = train_one_fold(
            build_resnet50, train_ds, val_ds, cfg, fold=fold_idx
        )
        per_fold.append(metrics)
        save_predictions(
            PRED_CSV, val_paths, y_true, y_prob,
            fold=fold_idx, seed=args.seed, model=args.tag,
        )
        print(f"  -> fold {fold_idx}: acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
              f"auc={metrics.auc:.3f}")

    agg = aggregate_folds(per_fold)
    row = format_results_row(args.tag, agg)
    print("\n=== Aggregated (mean ± std across folds) ===")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, std = agg[k]
        print(f"  {k:10s}: {mean:.3f} ± {std:.3f}")

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.tag,
        "config": vars(args),
        "device": device,
        "per_fold": [m.as_dict() for m in per_fold],
        "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved predictions: {PRED_CSV}")
    print(f"Saved summary:     {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
