"""Leaf (plant tissue) masks: the denominator of severity.

Two methods:
- vegetation_mask: Excess Green index + Otsu threshold. NumPy only, runs on
  any CPU; the offline fallback and the source of SAM2 prompts.
- SAM2LeafSegmenter: SAM2 (pretrained, no training) prompted with points
  sampled from the vegetation mask and the lesions; the candidate mask that
  best agrees with the vegetation mask is kept.
Lesion pixels are always added to the leaf, since diseased tissue is often
brown or yellow and missed by a green index.
"""

from __future__ import annotations

import numpy as np


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on chromatic coordinates (each channel / channel sum)."""
    x = rgb.astype(np.float32)
    s = x.sum(axis=2) + 1e-6
    r, g, b = x[..., 0] / s, x[..., 1] / s, x[..., 2] / s
    return 2 * g - r - b


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold that maximises between-class variance (Otsu, 1979)."""
    v = values.ravel()
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def vegetation_mask(rgb: np.ndarray, lesion: np.ndarray | None = None) -> np.ndarray:
    """Rough plant mask (uint8 0/1) from colour alone, lesions included."""
    exg = excess_green(rgb)
    veg = exg > otsu_threshold(exg)
    if lesion is not None:
        veg |= lesion > 0
    return veg.astype(np.uint8)


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion with a (2r+1) square, NumPy only."""
    m = mask > 0
    if r <= 0:
        return m.astype(np.uint8)
    p = np.pad(m, r, constant_values=False)
    out = np.ones_like(m)
    h, w = m.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out &= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def sample_points(mask: np.ndarray, n: int, rng: np.random.Generator, margin: int = 4) -> np.ndarray:
    """Up to n (x, y) points well inside the mask (falls back to any mask pixel)."""
    inner = erode(mask, margin)
    ys, xs = np.nonzero(inner if inner.any() else mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def choose_mask(masks: np.ndarray, scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pick the candidate (K x H x W) that best matches the reference mask;
    SAM's own score breaks ties."""
    keys = [(mask_iou(m, reference), float(s)) for m, s in zip(masks, scores)]
    best = max(range(len(keys)), key=lambda i: keys[i])
    return (masks[best] > 0).astype(np.uint8)


class SAM2LeafSegmenter:
    def __init__(self, repo_dir: str, size: str = "l", checkpoint: str | None = None,
                 device: str = "cuda", n_veg_points: int = 5, n_lesion_points: int = 2, seed: int = 0) -> None:
        from .models.sam2unet import CONFIGS, _import_repo

        build_sam2 = _import_repo(repo_dir)[0]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # bundled with SAM2-UNet

        self.predictor = SAM2ImagePredictor(build_sam2(CONFIGS[size], checkpoint, device=device))
        self.n_veg, self.n_les = n_veg_points, n_lesion_points
        self.rng = np.random.default_rng(seed)

    def segment(self, rgb: np.ndarray, lesion: np.ndarray | None = None) -> np.ndarray:
        """rgb: HxWx3 uint8. lesion: HxW mask or None. Returns HxW uint8 leaf mask."""
        veg = vegetation_mask(rgb, lesion)
        pts = [sample_points(veg, self.n_veg, self.rng)]
        if lesion is not None and lesion.any():
            pts.append(sample_points(lesion, self.n_les, self.rng, margin=1))
        pts = np.concatenate(pts)
        if len(pts) == 0:
            return veg
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=np.ones(len(pts), np.int32), multimask_output=True
        )
        leaf = choose_mask(masks, scores, veg)
        if lesion is not None:
            leaf |= (lesion > 0).astype(np.uint8)
        return leaf
