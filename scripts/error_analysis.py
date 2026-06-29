from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import list_images

PRED_CSV = PROJECT / "results" / "tables" / "predictions.csv"
FIGS = PROJECT / "results" / "figures"
TABLES = PROJECT / "results" / "tables"

MAIN_MODELS = ["vgg13_paper", "resnet50_probe", "dinov2_probe", "clip_probe",
               "ensemble_resnet50+dinov2+clip"]


def main() -> None:
    df = pd.read_csv(PRED_CSV)
    df = df[df["model"].isin(MAIN_MODELS)].copy()
    df["correct"] = df["y_true"] == df["y_pred"]

    correct = df.pivot_table(
        index=["image", "y_true"], columns="model", values="correct", aggfunc="first"
    ).reset_index()

    available = [m for m in MAIN_MODELS if m in correct.columns]
    correct["n_correct"] = correct[available].sum(axis=1)
    correct["n_wrong"] = len(available) - correct["n_correct"]

    print(f"Models considered: {available}")
    print(f"Total images with predictions: {len(correct)}\n")
    print("Distribution of (# models correct) per image:")
    print(correct["n_correct"].value_counts().sort_index().to_string())

    hard = correct.sort_values(["n_correct", "image"]).head(8).copy()
    print("\nHardest images (lowest # models correct):")
    for _, r in hard.iterrows():
        print(f"  {r['image']} true={int(r['y_true'])} n_correct={int(r['n_correct'])}/{len(available)}")

    hard.to_csv(TABLES / "hardest_images.csv", index=False)

    paths_by_name = {p.name: p for p in list_images()}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()
    for i, (_, r) in enumerate(hard.head(6).iterrows()):
        path = paths_by_name.get(r["image"])
        if path is None:
            axes[i].axis("off")
            continue
        axes[i].imshow(Image.open(path).convert("L"), cmap="gray")
        true_lbl = "GOOD" if r["y_true"] == 1 else "BAD"
        axes[i].set_title(f"{r['image']}  true={true_lbl}\n{int(r['n_correct'])}/{len(available)} models correct", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Hardest cases: images most models got wrong\n(candidates for biological/morphological discussion on the poster)", fontsize=10)
    fig.tight_layout()
    out = FIGS / "hardest_images.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
