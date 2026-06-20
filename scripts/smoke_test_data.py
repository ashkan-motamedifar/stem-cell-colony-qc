"""Sanity check: dataset loads, labels parse, splits are balanced, MPS works.

Run from project root: ``python scripts/smoke_test_data.py``
"""
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import (
    MamaevaDataset,
    get_device,
    imagenet_transform,
    list_images,
    paper_transform,
    stratified_kfold,
    stratified_split,
)


def main() -> None:
    paths = list_images()
    labels = [Counter(p.name.split("_")[-1] for p in paths)]
    print(f"Total images: {len(paths)}")
    print(f"Class balance: {dict(Counter(['good' if 'good' in p.name else 'bad' for p in paths]))}")

    train_paths, val_paths = stratified_split(paths, val_ratio=0.2, seed=42)
    print(f"\nSingle split (4:1):")
    print(f"  train: {len(train_paths)}  (good={sum('good' in p.name for p in train_paths)}, bad={sum('bad' in p.name for p in train_paths)})")
    print(f"  val:   {len(val_paths)}  (good={sum('good' in p.name for p in val_paths)}, bad={sum('bad' in p.name for p in val_paths)})")

    folds = stratified_kfold(paths, n_splits=5, seed=42)
    print(f"\n5-fold CV:")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: train={len(tr)} (good={sum('good' in p.name for p in tr)}/bad={sum('bad' in p.name for p in tr)})  "
              f"val={len(va)} (good={sum('good' in p.name for p in va)}/bad={sum('bad' in p.name for p in va)})")

    ds_paper = MamaevaDataset(train_paths, transform=paper_transform(train=True))
    ds_imnet = MamaevaDataset(val_paths, transform=imagenet_transform(train=False))
    x_p, y_p = ds_paper[0]
    x_i, y_i = ds_imnet[0]
    print(f"\nPaper transform sample:    shape={tuple(x_p.shape)}, dtype={x_p.dtype}, "
          f"min={x_p.min():.3f}, max={x_p.max():.3f}, label={y_p}")
    print(f"ImageNet transform sample: shape={tuple(x_i.shape)}, dtype={x_i.dtype}, "
          f"min={x_i.min():.3f}, max={x_i.max():.3f}, label={y_i}")

    device = get_device()
    print(f"\nDevice: {device}")
    import torch
    x_p_dev = x_p.unsqueeze(0).to(device)
    print(f"Moved sample to {device}: OK (shape={tuple(x_p_dev.shape)})")


if __name__ == "__main__":
    main()
