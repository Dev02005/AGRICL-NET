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


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation with a (2r+1) square, NumPy only."""
    m = mask > 0
    if r <= 0:
        return m.astype(np.uint8)
    p = np.pad(m, r, constant_values=False)
    out = np.zeros_like(m)
    h, w = m.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + h, dx:dx + w]
    return out.astype(np.uint8)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components: (labels, count); label 0 is background.
    OpenCV when available (fast), else a pure-NumPy/Python flood fill."""
    fg = (mask > 0).astype(np.uint8)
    try:
        import cv2

        n, labels = cv2.connectedComponents(fg, connectivity=8)
        return labels.astype(np.int32), int(n - 1)
    except ImportError:
        pass
    labels = np.zeros(fg.shape, np.int32)
    h, w = fg.shape
    n = 0
    for sy, sx in zip(*np.nonzero(fg)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for ny in range(max(y - 1, 0), min(y + 2, h)):
                for nx in range(max(x - 1, 0), min(x + 2, w)):
                    if fg[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = n
                        stack.append((ny, nx))
    return labels, n


def anchored_vegetation_mask(rgb: np.ndarray, lesion: np.ndarray | None, reach: int = 5) -> np.ndarray:
    """Colour-index leaf restricted to the green regions that touch the lesions.

    The lesions sit on the leaf being assessed, so green regions elsewhere in
    the photo (other leaves, grass, a blurred background) are dropped. Falls
    back to the plain vegetation mask when there are no lesions or none of
    the green regions touch them."""
    veg = excess_green(rgb) > otsu_threshold(excess_green(rgb))
    if lesion is None or not (lesion > 0).any():
        return veg.astype(np.uint8)
    les = lesion > 0
    labels, n = label_components(veg & ~les)
    keep = np.unique(labels[(dilate(les, reach) > 0) & (labels > 0)])
    if n == 0 or len(keep) == 0:
        return (veg | les).astype(np.uint8)
    return (np.isin(labels, keep) | les).astype(np.uint8)


def choose_leaf_mask(masks: np.ndarray, scores: np.ndarray, lesion: np.ndarray,
                     min_cover: float = 0.5, min_growth: float = 1.15, max_area: float = 0.9) -> np.ndarray | None:
    """Leaf among SAM's candidates, anchored on the lesions.

    Valid candidates contain at least min_cover of the lesion pixels, are
    larger than the lesions themselves (else it is just a spot) and do not
    fill the photo (else it is the background). Among those that cover the
    lesions about as well as the best one, the smallest is the leaf: the next
    larger candidates are the whole plant or the scene. None if no candidate
    is valid."""
    les = lesion > 0
    n_les = int(les.sum())
    if n_les == 0:
        return None
    valid = []
    for m, s in zip(masks, scores):
        m = m > 0
        area = int(m.sum())
        cover = (m & les).sum() / n_les
        if cover >= min_cover and area >= min_growth * n_les and area <= max_area * m.size:
            valid.append((cover, area, float(s), m))
    if not valid:
        return None
    best = max(v[0] for v in valid)
    near = [v for v in valid if v[0] >= 0.9 * best]
    return min(near, key=lambda v: (v[1], -v[2]))[3].astype(np.uint8)


class SAM2LeafSegmenter:
    """prompt="vegetation": points from the green-index mask plus the lesions,
    candidate closest to the green-index mask (the original method).
    prompt="lesion": one prompt per lesion point, the leaf is the smallest
    candidate that contains the lesions (choose_leaf_mask); used when field
    photos have green backgrounds or other leaves."""

    def __init__(self, repo_dir: str, size: str = "l", checkpoint: str | None = None,
                 device: str = "cuda", n_veg_points: int = 5, n_lesion_points: int = 2, seed: int = 0,
                 prompt: str = "vegetation") -> None:
        from .models.sam2unet import CONFIGS, _import_repo

        build_sam2 = _import_repo(repo_dir)[0]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # bundled with SAM2-UNet

        if prompt not in ("vegetation", "lesion"):
            raise ValueError(f"prompt must be 'vegetation' or 'lesion', not {prompt!r}")
        self.predictor = SAM2ImagePredictor(build_sam2(CONFIGS[size], checkpoint, device=device))
        self.n_veg, self.n_les = n_veg_points, n_lesion_points
        self.rng = np.random.default_rng(seed)
        self.prompt = prompt
        self._image = None  # the photo currently embedded in the predictor

    def _set_image(self, rgb: np.ndarray) -> None:
        if self._image is not rgb:  # same array object: reuse the embedding
            self.predictor.set_image(rgb)
            self._image = rgb

    def segment(self, rgb: np.ndarray, lesion: np.ndarray | None = None, prompt: str | None = None) -> np.ndarray:
        """rgb: HxWx3 uint8. lesion: HxW mask or None. Returns HxW uint8 leaf mask."""
        if (prompt or self.prompt) == "lesion" and lesion is not None and (lesion > 0).any():
            return self._segment_from_lesions(rgb, lesion)
        veg = vegetation_mask(rgb, lesion)
        pts = [sample_points(veg, self.n_veg, self.rng)]
        if lesion is not None and lesion.any():
            pts.append(sample_points(lesion, self.n_les, self.rng, margin=1))
        pts = np.concatenate(pts)
        if len(pts) == 0:
            return veg
        self._set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=pts, point_labels=np.ones(len(pts), np.int32), multimask_output=True
        )
        leaf = choose_mask(masks, scores, veg)
        if lesion is not None:
            leaf |= (lesion > 0).astype(np.uint8)
        return leaf

    def _segment_from_lesions(self, rgb: np.ndarray, lesion: np.ndarray, n_points: int = 3) -> np.ndarray:
        self._set_image(rgb)
        cands, scores = [], []
        for p in sample_points(lesion, n_points, np.random.default_rng(0), margin=1):
            m, s, _ = self.predictor.predict(point_coords=p[None], point_labels=np.ones(1, np.int32),
                                             multimask_output=True)
            cands.extend(m)
            scores.extend(s)
        leaf = choose_leaf_mask(np.asarray(cands), np.asarray(scores), lesion) if cands else None
        if leaf is None:
            return anchored_vegetation_mask(rgb, lesion)
        return (leaf | (lesion > 0)).astype(np.uint8)
