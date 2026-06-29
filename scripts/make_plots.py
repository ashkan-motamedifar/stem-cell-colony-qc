from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.metrics import compute_metrics

RESULTS_DIR = PROJECT / "results"
TABLES = RESULTS_DIR / "tables"
FIGS = RESULTS_DIR / "figures"

MODELS = [
    ("vgg13_paper", "VGG13\n(from scratch)", "#d62728"),
    ("resnet50_probe", "ResNet50\n(ImageNet)", "#1f77b4"),
    ("dinov2_probe", "DINOv2\n(SSL)", "#2ca02c"),
    ("clip_probe", "CLIP\n(VL probe)", "#9467bd"),
    ("ensemble_resnet50+dinov2+clip", "Ensemble\n(proposed)", "#ff7f0e"),
    ("clip_zeroshot_simple", "CLIP\n(zero-shot)", "#8c564b"),
]


def load_summary(tag: str) -> dict | None:
    candidates = [
        TABLES / f"{tag}_summary.json",
        TABLES / "vgg13_summary.json" if tag == "vgg13_paper" else None,
        TABLES / "clip_zeroshot_summary.json" if tag.startswith("clip_zeroshot") else None,
    ]
    for c in [x for x in candidates if x is not None]:
        if c.exists():
            return json.loads(c.read_text())
    return None


def metric_with_std(summary: dict, key: str) -> tuple[float, float]:
    if "aggregate" in summary:
        v = summary["aggregate"].get(key)
        if v is not None:
            return v["mean"], v["std"]
    if summary.get("model") == "clip_zeroshot":
        best = summary["best_prompt"]
        for prompt in summary["prompts"]:
            if prompt["name"] == best:
                return prompt[key], 0.0
    raise KeyError(f"No {key} in summary for {summary.get('model')}")


def bar_chart() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1", "F1 score")]:
        labels, vals, errs, colors = [], [], [], []
        for tag, label, color in MODELS:
            s = load_summary(tag)
            if s is None:
                continue
            m, e = metric_with_std(s, metric)
            labels.append(label); vals.append(m); errs.append(e); colors.append(color)
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, 0.7, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, label="chance")
        ax.axhline(0.89, color="red", linestyle=":", linewidth=0.8, label="paper")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle("Foundation models vs from-scratch CNN on hPSC colony QC\n5-fold stratified CV, n=269 (137 good / 132 bad)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "comparison_bars.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def confusion_matrices() -> None:
    df = pd.read_csv(TABLES / "predictions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()
    for i, (tag, label, _) in enumerate(MODELS):
        ax = axes[i]
        sub = df[df["model"] == tag]
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{label}\n(no data)", fontsize=10)
            continue
        m = compute_metrics(sub["y_true"].values, sub["y_prob"].values)
        cm = m.confusion
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}", ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["bad", "good"]); ax.set_yticklabels(["bad", "good"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{label.replace(chr(10), ' ')}\nacc={m.accuracy:.2f}", fontsize=10)
    for j in range(len(MODELS), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Confusion matrices (predictions aggregated across all 5 folds)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "confusion_matrices.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def csv_table() -> None:
    rows = []
    for tag, label, _ in MODELS:
        s = load_summary(tag)
        if s is None:
            continue
        row = {"model": label.replace("\n", " "), "tag": tag}
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            m, e = metric_with_std(s, k)
            row[k] = f"{m:.3f} ± {e:.3f}" if e > 0 else f"{m:.3f}"
        rows.append(row)
    out = TABLES / "comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    bar_chart()
    confusion_matrices()
    csv_table()
