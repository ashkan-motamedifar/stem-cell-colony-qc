"""Data loading for the Mamaeva 2022 hPSC colony dataset.

The Zenodo archive (DOI 10.5281/zenodo.7316404) extracts to a single folder
of 269 PNG images named like ``100Ax40_good.png`` / ``100Ax40_bad.png``,
where the substring before ``.png`` encodes the binary label.

Images are 1280x960 phase-contrast micrographs; we resize to the model's
expected input size at transform time.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

LABEL_MAP: dict[str, int] = {"bad": 0, "good": 1}
CLASS_NAMES: list[str] = ["bad", "good"]
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "H9p36"

_LABEL_RE = re.compile(r"_(good|bad)\.png$", re.IGNORECASE)


def parse_label(path: str | Path) -> int:
    name = Path(path).name
    m = _LABEL_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {name}")
    return LABEL_MAP[m.group(1).lower()]


def list_images(root: str | Path = DEFAULT_ROOT) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. "
            f"Run scripts/download_data.py first."
        )
    paths = sorted(p for p in root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {root}")
    return paths


class HistogramEqualize:
    """PIL/tensor histogram equalization, used in the paper's best config."""

    def __call__(self, img):
        return TF.equalize(img)


class MamaevaDataset(Dataset):
    """Image + binary label (0=bad, 1=good) dataset.

    Pass the subset of paths for this split — train/val partitioning happens
    upstream (see :func:`stratified_split` and :func:`stratified_kfold`).
    """

    def __init__(self, paths: Sequence[Path], transform: Callable | None = None):
        self.paths = list(paths)
        self.transform = transform
        self.labels = [parse_label(p) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def paper_transform(train: bool, image_size: int = 224) -> transforms.Compose:
    """Histogram equalization + rotation/crop augmentation, matching the paper's best config."""
    ops: list = [HistogramEqualize()]
    if train:
        ops += [
            transforms.RandomRotation(degrees=15, fill=128),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def imagenet_transform(train: bool, image_size: int = 224) -> transforms.Compose:
    """Plain ImageNet preprocessing, no histogram equalization."""
    if train:
        ops = [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
        ]
    return transforms.Compose(
        ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def stratified_split(
    paths: Sequence[Path],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Single train/val split, stratified by label. 4:1 by default (matches paper)."""
    labels = [parse_label(p) for p in paths]
    train_paths, val_paths = train_test_split(
        list(paths),
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )
    return train_paths, val_paths


def stratified_kfold(
    paths: Sequence[Path],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path]]]:
    """5-fold stratified CV. Returns list of (train_paths, val_paths) per fold."""
    paths = list(paths)
    labels = [parse_label(p) for p in paths]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(paths, labels):
        folds.append(([paths[i] for i in train_idx], [paths[i] for i in val_idx]))
    return folds


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
