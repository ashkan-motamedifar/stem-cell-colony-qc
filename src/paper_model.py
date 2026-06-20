"""Paper-faithful reproduction of Mamaeva et al. 2022's "VGG13" — which is NOT
torchvision's VGG13. It's a tiny custom 12-conv + 1-FC net with grayscale input,
narrow channels (thickness=4), histogram equalization, and BCE loss.

Architecture copied from mamaeva_et_al_2022.py in the Zenodo archive (Net class).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import exposure
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from .data import parse_label


def _conv3x3(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
    ]
    if pool:
        layers.append(nn.MaxPool2d((2, 2)))
    return nn.Sequential(*layers)


class PaperVGG13(nn.Module):
    """Mamaeva 2022 custom VGG13: 12 conv + 1 FC, grayscale, sigmoid binary output.

    Input shape: (B, 1, 256, 256) — grayscale, range [0, 1].
    Output:      (B, 1) sigmoid probability for class "good".
    """

    def __init__(self, thickness: int = 4):
        super().__init__()
        c = thickness
        self.conv1 = _conv3x3(1, c)
        self.conv2 = _conv3x3(c, c, pool=True)
        self.conv3 = _conv3x3(c, c * 2)
        self.conv4 = _conv3x3(c * 2, c * 2, pool=True)
        self.conv5 = _conv3x3(c * 2, c * 4)
        self.conv6 = _conv3x3(c * 4, c * 4, pool=True)
        self.conv7 = _conv3x3(c * 4, c * 8)
        self.conv8 = _conv3x3(c * 8, c * 8, pool=True)
        self.conv9 = _conv3x3(c * 8, c * 8)
        self.conv10 = _conv3x3(c * 8, c * 8, pool=True)
        self.conv11 = _conv3x3(c * 8, c * 8)
        self.conv12 = _conv3x3(c * 8, c * 8, pool=True)
        self.fc1 = nn.Linear(c * 8 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7, self.conv8,
            self.conv9, self.conv10, self.conv11, self.conv12,
        ]:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return torch.sigmoid(x)


class PaperDataset(torch.utils.data.Dataset):
    """Paper-faithful preprocessing pipeline.

    Train: resize 512x512 → random crop 256x256 → hflip + vflip + transpose → equalize → grayscale.
    Val:   resize 256x256 → equalize → grayscale.

    Returns (tensor (1, 256, 256) float in [0, 1], label float scalar).
    """

    def __init__(self, paths, train: bool = True):
        self.paths = list(paths)
        self.labels = [parse_label(p) for p in self.paths]
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # arr shape: (H, W, C) uint8
        if rng.random() < 0.5:
            arr = arr[:, ::-1, :]  # horizontal flip
        if rng.random() < 0.5:
            arr = arr[::-1, :, :]  # vertical flip
        if rng.random() < 0.5:
            arr = arr.transpose(1, 0, 2)  # transpose H<->W
        return np.ascontiguousarray(arr)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.train:
            img = img.resize((512, 512), Image.LANCZOS)
            arr = np.array(img)
            # random crop 256x256
            rng = np.random.default_rng()
            top = rng.integers(0, arr.shape[0] - 256 + 1)
            left = rng.integers(0, arr.shape[1] - 256 + 1)
            arr = arr[top : top + 256, left : left + 256, :]
            arr = self._augment(arr, rng)
        else:
            img = img.resize((256, 256), Image.LANCZOS)
            arr = np.array(img)

        arr_u8 = img_as_ubyte(arr)
        arr_eq = exposure.equalize_hist(arr_u8)  # → float in [0, 1]
        gray = rgb2gray(arr_eq).astype(np.float32)  # (256, 256)
        gray = gray[None, :, :]  # (1, 256, 256)
        x = torch.from_numpy(gray)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
