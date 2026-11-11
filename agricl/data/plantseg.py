"""PlantSeg v7 index and mask loading.

Verified layout (zenodo.org/records/17719108, plantseg.zip, md5 9358a66d...):
    plantseg/Metadata.csv                 one row per image (Index, Plant, Disease, Split, ...)
    plantseg/images/{train,val,test}/*.jpg
    plantseg/annotations/{train,val,test}/*.png
Mask pixels: 0 = background, Index + 1 = lesion of that image's disease
(one disease per image). 7,774 images, 115 diseases, 34 plants, no healthy images.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_DIRS = {"Training": "train", "Validation": "val", "Test": "test"}

# Diseases that also appear in the PDSNets dataset (cedar apple rust = PlantSeg
# "apple rust"). They are held out of all training so severity can be tested on
# diseases the system has never seen: the project's key experiment.
UNSEEN_DISEASES = ("soybean bacterial blight", "wheat stripe rust", "apple rust")


@dataclass(frozen=True)
class Record:
    image: Path
    mask: Path
    index: int  # 0..114, PlantSeg class index
    plant: str
    disease: str
    split: str  # "train" | "val" | "test"
    lesion_ratio: float  # lesion share of the whole image, from Metadata.csv


def load_records(root: str | Path) -> list[Record]:
    root = Path(root)
    out = []
    with open(root / "Metadata.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            split = SPLIT_DIRS[r["Split"]]
            out.append(
                Record(
                    image=root / "images" / split / r["Name"],
                    mask=root / "annotations" / split / r["Label file"],
                    index=int(r["Index"]),
                    plant=r["Plant"],
                    disease=r["Disease"],
                    split=split,
                    lesion_ratio=float(r["Mask ratio"]),
                )
            )
    return out


def split_seen_unseen(records: list[Record], unseen=UNSEEN_DISEASES):
    """Return (seen, unseen) records. Unseen diseases are excluded from training."""
    held = {d.lower() for d in unseen}
    seen = [r for r in records if r.disease.lower() not in held]
    hidden = [r for r in records if r.disease.lower() in held]
    return seen, hidden


def load_lesion_mask(record: Record) -> np.ndarray:
    """Binary lesion mask (uint8, 1 = lesion), class-agnostic as the design requires."""
    m = np.array(Image.open(record.mask))
    return (m > 0).astype(np.uint8)


def load_image(record: Record) -> Image.Image:
    return Image.open(record.image).convert("RGB")


def resized_copy(records: list[Record], out_root: str | Path, max_side: int = 640) -> list[Record]:
    """Save copies with the longer side at most max_side (masks resized to the
    same size with nearest-neighbour) and return records pointing at them.
    Existing copies are reused, so this is cheap to call again."""
    out_root = Path(out_root)
    out = []
    for r in records:
        img_dst = out_root / "images" / r.split / r.image.name
        mask_dst = out_root / "annotations" / r.split / r.mask.name
        if not (img_dst.exists() and mask_dst.exists()):
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            mask_dst.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(r.image).convert("RGB")
            img.thumbnail((max_side, max_side), Image.BICUBIC)
            img.save(img_dst, quality=92)
            Image.open(r.mask).resize(img.size, Image.NEAREST).save(mask_dst)
        out.append(replace(r, image=img_dst, mask=mask_dst))
    return out
