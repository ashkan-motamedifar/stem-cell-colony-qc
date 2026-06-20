"""Generic supervised training loop for binary image classification.

Used by VGG13 (from scratch) and ResNet50 (transfer learning) scripts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .metrics import BinaryMetrics, compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "mps"
    seed: int = 42
    num_workers: int = 0  # MPS + multi-worker is buggy; keep 0
    log_every: int = 10


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[BinaryMetrics, list[int], list[float]]:
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(y.tolist())
        y_prob.extend(probs.cpu().tolist())
    return compute_metrics(y_true, y_prob), y_true, y_prob


def train_one_fold(
    model_builder: Callable[[], nn.Module],
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
    fold: int = 0,
) -> tuple[BinaryMetrics, list[int], list[float], nn.Module]:
    """Train a fresh model on train_ds, evaluate on val_ds. Returns final-epoch metrics."""
    set_seed(cfg.seed + fold)
    model = model_builder().to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False,
    )

    best_acc, best_state, best_preds = 0.0, None, (None, None)
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        metrics, y_true, y_prob = evaluate(model, val_loader, cfg.device)
        if metrics.accuracy >= best_acc:
            best_acc = metrics.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = (y_true, y_prob)
            best_metrics = metrics
        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            elapsed = time.time() - t0
            print(f"  fold {fold} epoch {epoch:3d}/{cfg.epochs} | train_loss={loss:.4f} | "
                  f"val_acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} auc={metrics.auc:.3f} | "
                  f"best_acc={best_acc:.3f} | elapsed={elapsed:.1f}s")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_prob = best_preds
    return best_metrics, y_true, y_prob, model
