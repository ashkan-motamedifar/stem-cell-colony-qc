"""DINOv2 attention heatmaps overlaid on input images.

For 6 example images (correct-good, correct-bad, misclassified), pull the
last-layer self-attention from the CLS token to all patch tokens, reshape
to the 2D image grid, and overlay on the original image.

This is the "where is the model looking?" panel for the poster.

Usage:
    python scripts/attention_viz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import get_device, list_images, parse_label

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
FIGS = RESULTS_DIR / "figures"
MODEL_NAME = "facebook/dinov2-base"


def pick_examples(n_per: int = 1) -> list[tuple[Path, int, str]]:
    """Return list of (path, true_label, kind)."""
    df = pd.read_csv(PRED_CSV)
    # Use the ensemble's predictions if available, otherwise dinov2
    pref = "ensemble_resnet50+dinov2+clip"
    if (df["model"] == pref).any():
        sub = df[df["model"] == pref].copy()
    else:
        sub = df[df["model"] == "dinov2_probe"].copy()
    sub["correct"] = sub["y_true"] == sub["y_pred"]
    sub["margin"] = (sub["y_prob"] - 0.5).abs()

    paths_by_name = {p.name: p for p in list_images()}

    picks = []
    # Most-confident correct good
    confident_good = sub[(sub["correct"]) & (sub["y_true"] == 1)].nlargest(2, "margin")
    for _, r in confident_good.iterrows():
        picks.append((paths_by_name[r["image"]], 1, "correct good"))
    # Most-confident correct bad
    confident_bad = sub[(sub["correct"]) & (sub["y_true"] == 0)].nlargest(2, "margin")
    for _, r in confident_bad.iterrows():
        picks.append((paths_by_name[r["image"]], 0, "correct bad"))
    # Misclassified
    misclassified = sub[~sub["correct"]].nlargest(2, "margin")
    for _, r in misclassified.iterrows():
        picks.append((paths_by_name[r["image"]], int(r["y_true"]),
                      f"miss ({'bad' if r['y_true']==0 else 'good'} pred as {'good' if r['y_pred']==1 else 'bad'})"))
    return picks


def attention_map(model, processor, img_path: Path, device: str) -> tuple[np.ndarray, Image.Image]:
    """Return (HxW attention heatmap, original PIL image)."""
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    # last layer, average across heads, take CLS row, drop CLS column
    attn = out.attentions[-1]  # (1, n_heads, n_tokens, n_tokens)
    cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (n_patches,)
    n_patches = cls_attn.shape[0]
    side = int(round(n_patches ** 0.5))
    grid = cls_attn[: side * side].reshape(side, side).float().cpu().numpy()
    # normalize
    g = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return g, img


def main() -> None:
    device = get_device()
    print(f"Device: {device}  (with MPS fallback)")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()

    picks = pick_examples()
    print(f"Visualizing {len(picks)} images")
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    label_str = {0: "BAD", 1: "GOOD"}
    for i, (path, true_label, kind) in enumerate(picks):
        grid, img = attention_map(model, processor, path, device)
        # Upsample heatmap to image resolution for overlay
        heat = torch.from_numpy(grid)[None, None]
        heat_up = F.interpolate(heat, size=img.size[::-1], mode="bicubic", align_corners=False)
        heat_up = heat_up.squeeze().numpy()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{kind}\n{path.name}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(img.convert("L"), cmap="gray")
        axes[1, i].imshow(heat_up, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"true: {label_str[true_label]}", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle("DINOv2 attention (CLS → patches, last layer): where the model looks",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / "dinov2_attention.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
