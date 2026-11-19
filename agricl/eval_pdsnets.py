"""True severity accuracy on the PDSNets field dataset.

PDSNets labels both the leaf and the disease spots, so the true severity is
spots / (spots + healthy leaf) x 100. Each "system" returns a lesion mask and a
leaf mask for a photo; comparing systems separates the two sources of error:

* predicted lesions + true leaf     -> error of the lesion model alone
* true lesions + predicted leaf     -> error of the leaf mask alone
* predicted lesions + predicted leaf -> the full system as the app runs it

All three PDSNets diseases are held out of training, so every number here is on
diseases the models have never seen.
"""

from __future__ import annotations

import csv
import zlib
from collections import defaultdict
from typing import Callable

import numpy as np

from .data.pdsnets import HEALTHY, SPOT, load_image, load_label
from .metrics import dice, iou, r2_score, rmse
from .severity import compute_severity, severity_from_pdsnets_label, severity_level

# system(img, rgb, true_spot, true_leaf, cache) -> (lesion mask, leaf mask), both HxW
System = Callable[..., tuple[np.ndarray, np.ndarray]]


def split_of(name: str) -> str:
    """Fixed half of the photos ("dev") for choosing between methods; the
    other half ("test") is only for reporting, so the choice is not tuned on
    the numbers that are reported."""
    return "dev" if zlib.crc32(name.encode("utf-8")) % 2 == 0 else "test"


def summarize(rows: list[dict]) -> dict:
    t = [r["severity_true"] for r in rows]
    p = [r["severity_est"] for r in rows]
    return {
        "images": len(rows),
        "severity_r2": r2_score(t, p),
        "severity_rmse": rmse(t, p),
        "severity_mae": float(np.mean(np.abs(np.subtract(t, p)))),
        "level_accuracy": float(np.mean([r["level_true"] == r["level_est"] for r in rows])),
        "mean_lesion_dice": float(np.mean([r["lesion_dice"] for r in rows])),
        "mean_leaf_iou": float(np.mean([r["leaf_iou"] for r in rows])),
        "mean_true_severity": float(np.mean(t)),
    }


def evaluate_pdsnets(records, systems: dict[str, System], out_csv: str | None = None,
                     progress: int = 0) -> dict:
    """Returns {system: {"all": summary, "by_disease": {...}, "by_split": {"dev": ..., "test": ...}}}."""
    rows = []
    for i, rec in enumerate(records):
        label = load_label(rec)
        h, w = label.shape
        img = load_image(rec, (w, h))
        rgb = np.array(img)
        if rgb.shape[:2] != label.shape:
            raise ValueError(f"{rec.image.name}: image {rgb.shape[:2]} vs label {label.shape}")
        spot = (label == SPOT).astype(np.uint8)
        leaf_true = (label >= HEALTHY).astype(np.uint8)
        true = severity_from_pdsnets_label(label, HEALTHY, SPOT)
        cache: dict = {}
        for name, system in systems.items():
            lesion, leaf = system(img, rgb, spot, leaf_true, cache)
            est = compute_severity(leaf, lesion)
            rows.append({
                "system": name, "image": rec.image.name, "disease": rec.disease, "split": split_of(rec.image.name),
                "severity_true": true, "severity_est": est.percent,
                "level_true": severity_level(true), "level_est": est.level,
                "lesion_dice": dice(lesion, spot), "leaf_iou": iou(leaf, leaf_true),
            })
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{len(records)} photos")

    if out_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    out = {}
    for name, rs in by.items():
        diseases, splits = defaultdict(list), defaultdict(list)
        for r in rs:
            diseases[r["disease"]].append(r)
            splits[r["split"]].append(r)
        out[name] = {"all": summarize(rs), "by_disease": {d: summarize(v) for d, v in sorted(diseases.items())},
                     "by_split": {k: summarize(v) for k, v in sorted(splits.items())}}
    return out


def clip_to(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Mask pixels inside region only (so a true leaf is not grown by stray lesions)."""
    return ((mask > 0) & (region > 0)).astype(np.uint8)
