from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

LABEL_MAP: dict[str, int] = {"bad": 0, "good": 1}
CLASS_NAMES: list[str] = ["bad", "good"]
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "H9p36"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_LABEL_RE = re.compile(r"_(good|bad)\.png$", re.IGNORECASE)


def parse_label(path: str | Path) -> int:
    m = _LABEL_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {Path(path).name}")
    return LABEL_MAP[m.group(1).lower()]


def list_images(root: str | Path = DEFAULT_ROOT) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}. Run scripts/download_data.py first.")
    paths = sorted(p for p in root.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {root}")
    return paths


class HistogramEqualize:
    def __call__(self, img):
        return TF.equalize(img)


class MamaevaDataset(Dataset):
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


def paper_transform(train: bool, image_size: int = 224) -> transforms.Compose:
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
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def stratified_split(paths: Sequence[Path], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    labels = [parse_label(p) for p in paths]
    return train_test_split(list(paths), test_size=val_ratio, stratify=labels, random_state=seed)


def stratified_kfold(paths: Sequence[Path], n_splits: int = 5, seed: int = 42) -> list[tuple[list[Path], list[Path]]]:
    paths = list(paths)
    labels = [parse_label(p) for p in paths]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [([paths[i] for i in tr], [paths[i] for i in va]) for tr, va in skf.split(paths, labels)]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
