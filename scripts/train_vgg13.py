"""Train Mamaeva 2022's custom VGG13 from scratch on the hPSC colony dataset.

This is the paper-faithful reproduction: the "VGG13" in the paper is a tiny
custom net (~75k params) — 12 conv + 1 FC, grayscale input, thickness=4,
sigmoid + BCE. NOT torchvision's ImageNet VGG13 (133M params).

Same training config as the paper:
  - Adam (default lr=1e-3), ReduceLROnPlateau scheduler
  - BCELoss
  - 100 epochs
  - Best model by val F1
  - Random crop 256 from 512, hflip + vflip + transpose, histogram equalization

Wraps it in 5-fold stratified CV for publication-grade rigor.

Usage:
    python scripts/train_vgg13.py                          # full 5-fold CV
    python scripts/train_vgg13.py --folds 1 --epochs 5     # sanity
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import get_device, list_images, stratified_kfold
from src.metrics import aggregate_folds, compute_metrics, save_predictions
from src.paper_model import PaperDataset, PaperVGG13

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
SUMMARY_JSON = RESULTS_DIR / "tables" / "vgg13_summary.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x).squeeze(-1)  # (B,)
        y_true.extend(y.tolist())
        y_prob.extend(out.cpu().tolist())
    return compute_metrics(y_true, y_prob), y_true, y_prob


def train_fold(train_paths, val_paths, args, device, fold):
    set_seed(args.seed + fold)
    model = PaperVGG13(thickness=args.thickness).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if fold == 0:
        print(f"  Model: PaperVGG13(thickness={args.thickness}), params={n_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_ds = PaperDataset(train_paths, train=True)
    val_ds = PaperDataset(val_paths, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_f1, best_state, best_metrics, best_preds = -1.0, None, None, (None, None)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total_loss / max(n, 1)

        metrics, y_true, y_prob = evaluate(model, val_loader, device)
        scheduler.step(train_loss)

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            best_preds = (y_true, y_prob)

        log_every = max(args.epochs // 10, 1)
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"  fold {fold} epoch {epoch:3d}/{args.epochs} | "
                  f"loss={train_loss:.4f} | "
                  f"val acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} auc={metrics.auc:.3f} | "
                  f"best_f1={best_f1:.3f} | {elapsed:.1f}s",
                  flush=True)

    return best_metrics, best_preds, best_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--thickness", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="vgg13_paper")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Tag: {args.tag}")
    print(f"Config: folds={args.folds} epochs={args.epochs} bs={args.batch_size} "
          f"lr={args.lr} thickness={args.thickness} seed={args.seed}")

    paths = list_images()
    all_folds = stratified_kfold(paths, n_splits=5, seed=args.seed)
    folds_to_run = all_folds[: args.folds]

    per_fold = []
    for fold_idx, (train_paths, val_paths) in enumerate(folds_to_run):
        print(f"\n=== Fold {fold_idx+1}/{args.folds} | "
              f"train={len(train_paths)} val={len(val_paths)} ===", flush=True)
        metrics, preds, _ = train_fold(train_paths, val_paths, args, device, fold_idx)
        y_true, y_prob = preds
        per_fold.append(metrics)
        save_predictions(
            PRED_CSV, val_paths, y_true, y_prob,
            fold=fold_idx, seed=args.seed, model=args.tag,
        )
        print(f"  -> fold {fold_idx}: acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
              f"auc={metrics.auc:.3f}", flush=True)

    agg = aggregate_folds(per_fold)
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
