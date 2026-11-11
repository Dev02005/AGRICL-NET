"""PyTorch dataset for disease classification on lesion-centred crops."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ..roi import roi_box
from .lesion_dataset import MEAN, STD
from .plantseg import Record, load_image, load_lesion_mask


def build_class_index(records: list[Record]) -> dict[str, int]:
    """Disease name -> class id, sorted by name for a stable order."""
    return {d: i for i, d in enumerate(sorted({r.disease for r in records}))}


def eval_transform(size: int):
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class CropDataset(Dataset):
    """Crops a square around the true lesions (training jitters it), resizes to
    size x size. Diseases missing from class_to_idx get label -1 (unknown)."""

    def __init__(self, records: list[Record], class_to_idx: dict[str, int], size: int = 224,
                 train: bool = False, margin: float = 0.15, min_frac: float = 0.3, jitter: float = 0.15) -> None:
        self.records = list(records)
        self.class_to_idx = class_to_idx
        self.train = train
        self.margin, self.min_frac, self.jitter = margin, min_frac, jitter
        if train:
            self.tf = T.Compose([
                T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ])
        else:
            self.tf = eval_transform(size)

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.array([self.class_to_idx.get(r.disease, -1) for r in self.records])

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        img = load_image(r)
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)))) if self.train else None
        box = roi_box(load_lesion_mask(r), self.margin, self.min_frac,
                      self.jitter if self.train else 0.0, rng)
        return {"image": self.tf(img.crop(box)), "label": self.class_to_idx.get(r.disease, -1), "idx": i}
