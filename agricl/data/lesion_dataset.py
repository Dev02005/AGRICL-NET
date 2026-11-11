"""PyTorch dataset: PlantSeg image -> binary lesion mask."""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode

from .plantseg import Record, load_image, load_lesion_mask

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """RGB PIL image -> normalised 1x3xSxS tensor for inference."""
    img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
    return TF.normalize(TF.to_tensor(img), MEAN, STD)[None]


class LesionDataset(Dataset):
    """Images are squashed to size x size (as SAM2-UNet does). Masks use
    nearest-neighbour resizing so they stay binary; the SAM2-UNet repo's
    bicubic resize of 'L' masks would blur PlantSeg's 1..115 index values."""

    def __init__(self, records: list[Record], size: int = 352, train: bool = False,
                 soft_dir: str | None = None) -> None:
        """soft_dir: teacher probability maps for distillation, 8-bit PNGs at
        size x size in soft_dir/<split>/<mask file name> (see edge_train)."""
        self.records = list(records)
        self.size = size
        self.train = train
        self.soft_dir = soft_dir
        self.jitter = ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        s = self.size
        img = TF.resize(load_image(r), [s, s], interpolation=InterpolationMode.BILINEAR)
        mask = Image.fromarray(load_lesion_mask(r) * 255)
        mask = TF.resize(mask, [s, s], interpolation=InterpolationMode.NEAREST)
        maps = [mask]
        if self.soft_dir:
            maps.append(Image.open(f"{self.soft_dir}/{r.split}/{r.mask.name}"))
        if self.train:
            if random.random() < 0.5:
                img, maps = TF.hflip(img), [TF.hflip(m) for m in maps]
            if random.random() < 0.5:
                img, maps = TF.vflip(img), [TF.vflip(m) for m in maps]
            k = random.randint(0, 3)
            if k:  # exact 90-degree turns: no interpolation, mask stays binary
                img, maps = img.rotate(90 * k), [m.rotate(90 * k) for m in maps]
            img = self.jitter(img)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = torch.from_numpy((np.array(maps[0]) > 127).astype(np.float32))[None]
        out = {"image": x, "mask": y, "idx": i}
        if self.soft_dir:
            out["soft"] = torch.from_numpy(np.array(maps[1], np.float32) / 255.0)[None]
        return out
