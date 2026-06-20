"""Metrics for binary classification, computed from per-image predictions.

We always save per-image (path, true label, predicted prob) so we can:
- recompute any metric without re-running models
- run paired statistical tests across models
- pull out the misclassified images for error analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    confusion: np.ndarray  # 2x2: rows=true (bad, good), cols=pred (bad, good)

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "auc": float(self.auc),
        }


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> BinaryMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return BinaryMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan"),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def aggregate_folds(per_fold: list[BinaryMetrics]) -> dict[str, tuple[float, float]]:
    """Returns dict of metric -> (mean, std) across folds."""
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    out = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in per_fold])
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


def format_results_row(model_name: str, agg: dict[str, tuple[float, float]]) -> dict:
    row = {"model": model_name}
    for k, (mean, std) in agg.items():
        row[k] = f"{mean:.3f} ± {std:.3f}"
        row[f"{k}_mean"] = mean
        row[f"{k}_std"] = std
    return row


def save_predictions(
    csv_path: Path,
    paths: Sequence[Path],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    fold: int,
    seed: int,
    model: str,
) -> None:
    """Append per-image predictions to a CSV so we can do paired tests later."""
    df = pd.DataFrame({
        "model": model,
        "fold": fold,
        "seed": seed,
        "image": [Path(p).name for p in paths],
        "y_true": list(y_true),
        "y_prob": list(y_prob),
        "y_pred": [int(p >= 0.5) for p in y_prob],
    })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=header)
