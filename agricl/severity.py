"""Severity estimation from leaf and lesion masks.

Severity = lesion pixels inside the leaf / leaf pixels x 100.
Levels follow PlantHealthNet: Mild < 20%, Moderate 20-50%, Severe > 50%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MILD_MAX = 20.0
MODERATE_MAX = 50.0


@dataclass(frozen=True)
class SeverityResult:
    percent: float
    level: str
    leaf_pixels: int
    lesion_pixels: int


def severity_level(percent: float) -> str:
    if percent < MILD_MAX:
        return "Mild"
    if percent <= MODERATE_MAX:
        return "Moderate"
    return "Severe"


def compute_severity(leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> SeverityResult:
    """Both masks are HxW arrays; any non-zero value counts as foreground.

    Lesion pixels outside the leaf are ignored, and lesion pixels always count
    as leaf (a lesion is part of the leaf even if the leaf mask missed it).
    """
    if leaf_mask.shape != lesion_mask.shape:
        raise ValueError(f"mask shapes differ: {leaf_mask.shape} vs {lesion_mask.shape}")
    lesion = lesion_mask > 0
    leaf = (leaf_mask > 0) | lesion
    leaf_px = int(leaf.sum())
    lesion_px = int((lesion & leaf).sum())
    if leaf_px == 0:
        return SeverityResult(0.0, "No leaf", 0, 0)
    pct = 100.0 * lesion_px / leaf_px
    return SeverityResult(pct, severity_level(pct), leaf_px, lesion_px)


def severity_from_pdsnets_label(label: np.ndarray, healthy: int, spot: int) -> float:
    """Ground-truth severity from a 3-class label map (background / healthy / spot),
    as in the PDSNets dataset: spots / (spots + healthy) x 100."""
    h = int((label == healthy).sum())
    s = int((label == spot).sum())
    return 0.0 if h + s == 0 else 100.0 * s / (h + s)


def lesion_boxes(lesion_mask: np.ndarray, min_area: int = 50) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1) of connected lesion regions.

    Replaces the DETR stage: boxes come straight from the segmentation mask.
    Uses OpenCV when available, else a pure-NumPy flood fill.
    """
    fg = (lesion_mask > 0).astype(np.uint8)
    try:
        import cv2

        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        return [
            (int(x), int(y), int(x + w), int(y + h))
            for x, y, w, h, area in stats[1:n]
            if area >= min_area
        ]
    except ImportError:
        return _boxes_numpy(fg, min_area)


def _boxes_numpy(fg: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    seen = np.zeros_like(fg, dtype=bool)
    h, w = fg.shape
    boxes = []
    for sy, sx in zip(*np.nonzero(fg)):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        ys, xs = [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(ys) >= min_area:
            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return boxes
