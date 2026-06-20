"""Zero-shot CLIP classification — no training at all.

Encode each image with CLIP. Encode candidate text prompts for both classes.
Predict the class whose prompt-embedding is closest (highest cosine similarity)
to the image embedding.

We test multiple prompt variants because zero-shot performance is sensitive
to wording. Reports per-prompt metrics and saves the best one's per-image
predictions for paired stat tests.

Usage:
    python scripts/clip_zeroshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import get_device, list_images
from src.embeddings import embed_clip_image
from src.metrics import compute_metrics, save_predictions

RESULTS_DIR = PROJECT / "results"
PRED_CSV = RESULTS_DIR / "tables" / "predictions.csv"
SUMMARY_JSON = RESULTS_DIR / "tables" / "clip_zeroshot_summary.json"

# Each entry is (label_for_bad, label_for_good). We pair them as a binary classifier.
PROMPT_VARIANTS = [
    {
        "name": "simple",
        "bad": "a microscope image of an unhealthy stem cell colony",
        "good": "a microscope image of a healthy stem cell colony",
    },
    {
        "name": "domain",
        "bad": "a phase-contrast microscopy image of a differentiating, low-quality human pluripotent stem cell colony with irregular borders",
        "good": "a phase-contrast microscopy image of a uniform, high-quality undifferentiated human pluripotent stem cell colony with smooth tight borders",
    },
    {
        "name": "contrast",
        "bad": "a photo of cells that look sick, irregular, or differentiated",
        "good": "a photo of cells that look healthy, uniform, and undifferentiated",
    },
    {
        "name": "morphological",
        "bad": "a phase-contrast image of a stem cell colony with rough edges and differentiating cells in the center",
        "good": "a phase-contrast image of a stem cell colony with a tight smooth border and uniform morphology",
    },
]


def main() -> None:
    device = get_device()
    paths = list_images()
    print(f"Device: {device}")
    print(f"Encoding {len(paths)} images with CLIP ViT-B/32...")

    # Returns (paths, X, y, model, tokenizer)
    _, X_img, y_true, model, tokenizer = embed_clip_image(
        paths, device=device, batch_size=16
    )
    print(f"Image embeddings: {X_img.shape}")

    X_img_t = torch.from_numpy(X_img).to(device).float()
    X_img_t = torch.nn.functional.normalize(X_img_t, dim=-1)
    y_true = np.asarray(y_true).astype(int)

    results_per_prompt = []
    best = {"name": None, "accuracy": -1.0}
    print()
    for v in PROMPT_VARIANTS:
        with torch.no_grad():
            tokens = tokenizer([v["bad"], v["good"]]).to(device)
            text_emb = model.encode_text(tokens)
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
            # Cosine similarities, shape (N_images, 2)
            sims = X_img_t @ text_emb.T  # high if similar
        sims_np = sims.cpu().float().numpy()
        # softmax over the two classes to get a probability for "good" (idx 1)
        # CLIP's standard practice uses temperature 100; we use that for calibration
        logits = 100.0 * sims_np
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob_good = exp[:, 1] / exp.sum(axis=1)

        m = compute_metrics(y_true, prob_good)
        results_per_prompt.append({
            "name": v["name"],
            "bad_prompt": v["bad"],
            "good_prompt": v["good"],
            **m.as_dict(),
            "confusion": m.confusion.tolist(),
        })
        print(f"  prompt '{v['name']}': acc={m.accuracy:.3f} f1={m.f1:.3f} auc={m.auc:.3f}")

        if m.accuracy > best["accuracy"]:
            best = {"name": v["name"], "accuracy": m.accuracy, "metrics": m, "prob_good": prob_good}

    print(f"\nBest prompt: '{best['name']}' (acc={best['accuracy']:.3f})")

    # Save best-prompt predictions for paired tests (use fold=-1 to mark "no fold").
    save_predictions(
        PRED_CSV, paths, y_true.tolist(), best["prob_good"].tolist(),
        fold=-1, seed=0, model=f"clip_zeroshot_{best['name']}",
    )

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps({
        "model": "clip_zeroshot",
        "device": device,
        "n_images": len(paths),
        "embedding_dim": int(X_img.shape[1]),
        "prompts": results_per_prompt,
        "best_prompt": best["name"],
        "best_accuracy": float(best["accuracy"]),
    }, indent=2))
    print(f"\nSaved summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
