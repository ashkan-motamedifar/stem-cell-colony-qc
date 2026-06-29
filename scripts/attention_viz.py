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

from src.data import get_device, list_images

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
FIGS = RESULTS_DIR / "figures"
MODEL_NAME = "facebook/dinov2-base"


def pick_examples() -> list[tuple[Path, int, str]]:
    df = pd.read_csv(PRED_CSV)
    pref = "ensemble_resnet50+dinov2+clip"
    sub = df[df["model"] == (pref if (df["model"] == pref).any() else "dinov2_probe")].copy()
    sub["correct"] = sub["y_true"] == sub["y_pred"]
    sub["margin"] = (sub["y_prob"] - 0.5).abs()

    paths_by_name = {p.name: p for p in list_images()}
    picks = []
    for _, r in sub[(sub["correct"]) & (sub["y_true"] == 1)].nlargest(2, "margin").iterrows():
        picks.append((paths_by_name[r["image"]], 1, "correct good"))
    for _, r in sub[(sub["correct"]) & (sub["y_true"] == 0)].nlargest(2, "margin").iterrows():
        picks.append((paths_by_name[r["image"]], 0, "correct bad"))
    for _, r in sub[~sub["correct"]].nlargest(2, "margin").iterrows():
        picks.append((paths_by_name[r["image"]], int(r["y_true"]),
                      f"miss ({'bad' if r['y_true']==0 else 'good'} pred as {'good' if r['y_pred']==1 else 'bad'})"))
    return picks


def attention_map(model, processor, img_path: Path, device: str) -> tuple[np.ndarray, Image.Image]:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    cls_attn = out.attentions[-1][0, :, 0, 1:].mean(dim=0)
    side = int(round(cls_attn.shape[0] ** 0.5))
    grid = cls_attn[: side * side].reshape(side, side).float().cpu().numpy()
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
        heat = torch.from_numpy(grid)[None, None]
        heat_up = F.interpolate(heat, size=img.size[::-1], mode="bicubic", align_corners=False).squeeze().numpy()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{kind}\n{path.name}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img.convert("L"), cmap="gray")
        axes[1, i].imshow(heat_up, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"true: {label_str[true_label]}", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle("DINOv2 attention (CLS → patches, last layer): where the model looks", fontsize=11)
    fig.tight_layout()
    out = FIGS / "dinov2_attention.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
