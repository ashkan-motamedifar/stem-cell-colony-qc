# Stem cell colony QC: foundation models vs from-scratch CNN

Frozen ResNet-50 + DINOv2 + CLIP features beat a from-scratch CNN by 21 points on phase-contrast hPSC colony classification (86% vs 65% accuracy, 0.93 ROC-AUC, 5-fold CV, n=269). ~21 labels per fold to match the baseline.

Poster: [results/poster.pdf](results/poster.pdf) · Baseline: [Mamaeva 2022](https://doi.org/10.3390/ijms24010140) · Dataset: [Zenodo 7316404](https://doi.org/10.5281/zenodo.7316404)

## Run

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/download_data.py
    python scripts/train_vgg13.py
    for b in resnet50 dinov2 clip; do python scripts/probe_features.py --backbone $b; done
    python scripts/ensemble.py
    python scripts/make_plots.py
