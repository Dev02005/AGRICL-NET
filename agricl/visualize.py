"""Overlays for the app: leaf (green tint), lesions (red), boxes (yellow), heatmaps."""

from __future__ import annotations

import numpy as np

LEAF = np.array([0, 200, 0], np.float32)
LESION = np.array([230, 30, 30], np.float32)
BOX = np.array([255, 220, 0], np.uint8)


def overlay_masks(rgb: np.ndarray, leaf: np.ndarray | None, lesion: np.ndarray | None,
                  boxes=(), alpha: float = 0.35, box_width: int = 3) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if leaf is not None:
        m = leaf > 0
        out[m] = (1 - alpha) * out[m] + alpha * LEAF
    if lesion is not None:
        m = lesion > 0
        out[m] = 0.3 * out[m] + 0.7 * LESION
    out = out.clip(0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        t = box_width
        out[y0:min(y0 + t, y1), x0:x1] = BOX
        out[max(y1 - t, y0):y1, x0:x1] = BOX
        out[y0:y1, x0:min(x0 + t, x1)] = BOX
        out[y0:y1, max(x1 - t, x0):x1] = BOX
    return out


def heatmap(values: np.ndarray) -> np.ndarray:
    """0..1 values -> RGB, blue (low) through green and yellow to red (high)."""
    v = np.clip(values, 0, 1)[..., None]
    stops = np.array([[0, 0, 160], [0, 160, 255], [0, 220, 0], [255, 230, 0], [230, 0, 0]], np.float32)
    pos = v * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = pos - lo
    rgb = stops[lo[..., 0]] * (1 - frac) + stops[lo[..., 0] + 1] * frac
    return rgb.astype(np.uint8)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """cam: HxW in 0..1, same size as rgb."""
    return ((1 - alpha) * rgb.astype(np.float32) + alpha * heatmap(cam)).clip(0, 255).astype(np.uint8)
