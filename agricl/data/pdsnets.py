"""PDSNets field dataset (Field-PDD; Zhao et al., Agriculture 2025, 15(6), 583).

Kaggle: alexzcheny/testdataset (DOI 10.34740/kaggle/ds/6620728), CC BY-NC 4.0.
Verified layout (file list, 2026-11-20):
    Dataset/{cedar_apple_rust,soybean_bacterial_blight,wheat_stripe_rust}/
        <stem>.jpg          photo, padded to square and downsampled to 640x640
        <stem>_black.png    photo with the background removed (not used)
        <stem>_label.png    3-class label: background black, healthy leaf gray, spots white
    OriginalData/...        full-size photos without labels (not used)
80 labelled photos per disease. These three diseases are exactly the ones held
out of all PlantSeg training, so this dataset measures true severity (with a
true leaf mask) on diseases the models have never seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BACKGROUND, HEALTHY, SPOT = 0, 1, 2

# folder name -> PlantSeg disease name (cedar apple rust = PlantSeg "apple rust")
FOLDER_TO_DISEASE = {
    "soybean_bacterial_blight": "soybean bacterial blight",
    "wheat_stripe_rust": "wheat stripe rust",
    "cedar_apple_rust": "apple rust",
}


@dataclass(frozen=True)
class PdsRecord:
    image: Path
    label: Path
    disease: str


def find_root(search_dir: str | Path) -> Path:
    """The folder that holds the disease folders (the dataset's 'Dataset' folder)."""
    for p in sorted(Path(search_dir).rglob("*_label.png")):
        return p.parent.parent
    raise FileNotFoundError(f"no *_label.png under {search_dir}: attach the PDSNets dataset")


def load_records(root: str | Path) -> list[PdsRecord]:
    out = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        disease = FOLDER_TO_DISEASE.get(folder.name, folder.name.replace("_", " "))
        for img in sorted(folder.glob("*.jpg")):
            label = folder / f"{img.stem}_label.png"
            if label.exists():
                out.append(PdsRecord(img, label, disease))
    return out


def decode_label(arr: np.ndarray) -> np.ndarray:
    """Label image -> HxW uint8 of BACKGROUND / HEALTHY / SPOT.

    Accepts gray levels (black / gray / white, with resampling in-between values
    snapped to the nearest class) or class indices 0 / 1 / 2."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    if a.max() <= 2:
        return a.astype(np.uint8)
    out = np.full(a.shape, BACKGROUND, np.uint8)
    out[a >= 64] = HEALTHY
    out[a >= 192] = SPOT
    return out


def load_label(rec: PdsRecord) -> np.ndarray:
    img = Image.open(rec.label)
    if img.mode == "P":  # palette: map indices to their colours first
        img = img.convert("RGB")
    return decode_label(np.asarray(img))


def load_image(rec: PdsRecord, size: tuple[int, int] | None = None) -> Image.Image:
    """The photo as RGB. If an EXIF rotation is stored and the raw pixels do not
    match the label size but the rotated ones do, the rotated view is used."""
    raw = Image.open(rec.image)
    img = raw.convert("RGB")
    if size is not None and img.size != size:
        turned = ImageOps.exif_transpose(raw).convert("RGB")
        if turned.size == size:
            return turned
    return img
