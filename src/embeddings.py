from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .data import imagenet_transform, parse_label


@torch.no_grad()
def embed_dinov2(
    paths: Sequence[Path],
    device: str = "mps",
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 16,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    feats, labels = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"DINOv2 ({model_name})"):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        cls = model(**inputs).last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, dim=-1)
        feats.append(cls.cpu().numpy())
        labels.extend(parse_label(p) for p in batch)

    return list(paths), np.concatenate(feats, axis=0), np.asarray(labels)


@torch.no_grad()
def embed_clip_image(
    paths: Sequence[Path],
    device: str = "mps",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 16,
) -> tuple[list[Path], np.ndarray, np.ndarray, "open_clip.CLIP", any]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    feats, labels = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"CLIP image ({model_name})"):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        emb = torch.nn.functional.normalize(model.encode_image(imgs), dim=-1)
        feats.append(emb.cpu().float().numpy())
        labels.extend(parse_label(p) for p in batch)

    return list(paths), np.concatenate(feats, axis=0), np.asarray(labels), model, tokenizer


@torch.no_grad()
def embed_resnet50(
    paths: Sequence[Path],
    device: str = "mps",
    batch_size: int = 16,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    model.fc = torch.nn.Identity()

    transform = imagenet_transform(train=False)
    feats, labels = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="ResNet-50 (frozen)"):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        emb = torch.nn.functional.normalize(model(imgs), dim=-1)
        feats.append(emb.cpu().numpy())
        labels.extend(parse_label(p) for p in batch)

    return list(paths), np.concatenate(feats, axis=0), np.asarray(labels)
