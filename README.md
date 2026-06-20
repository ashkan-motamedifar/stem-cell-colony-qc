# Stem Cell Colony QC: Foundation Models vs From-Scratch CNN

Benchmarking pretrained / self-supervised / zero-shot vision models against a
from-scratch VGG13 baseline on a small (269-image) phase-contrast microscopy
dataset of human pluripotent stem cell colonies labeled good / bad.

Baseline reference: Mamaeva et al., IJMS 2022 — https://doi.org/10.3390/ijms24010140
Dataset: https://doi.org/10.5281/zenodo.7316404

## Research question

Can pretrained / self-supervised / zero-shot vision models match or beat a
from-scratch CNN baseline (89% accuracy) on this small biomedical dataset,
without the heavy preprocessing + augmentation the original paper needed?

## Models compared

| # | Model | Paradigm | Training on this dataset |
|---|-------|----------|---------------------------|
| 1 | VGG13 | From-scratch CNN | Full training |
| 2 | ResNet-50 | ImageNet pretrained | Fine-tune last layers |
| 3 | DINOv2 | Self-supervised | Linear probe on frozen features |
| 4 | CLIP | Vision-language | Zero-shot, text prompts |

Plus: GradCAM / attention visualizations, data efficiency curves
(accuracy vs training set size), and error analysis grounded in colony biology.

## Layout

```
src/         # shared modules (data loader, metrics, viz)
scripts/     # one entry-point per experiment
data/raw/    # downloaded Zenodo archive (gitignored)
results/     # figures + tables (gitignored, reproducible)
notebooks/   # exploratory analysis
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then download the dataset (see `scripts/download_data.py`).
