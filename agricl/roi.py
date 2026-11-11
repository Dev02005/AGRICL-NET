"""Region of interest for the classifier: a square crop around the lesions.

Training uses the true lesion mask, the app uses the predicted one, so the
classifier always sees the symptoms with some surrounding leaf for context
instead of the whole cluttered field photo.
"""

from __future__ import annotations

import numpy as np


def roi_box(mask: np.ndarray, margin: float = 0.15, min_frac: float = 0.3,
            jitter: float = 0.0, rng: np.random.Generator | None = None) -> tuple[int, int, int, int]:
    """Square (x0, y0, x1, y1) box around the mask's foreground, clipped to the image.

    margin: extra space on each side, as a fraction of the lesion box size.
    min_frac: smallest side, as a fraction of the image's shorter side (context for tiny spots).
    jitter: training-time random shift and scale, as a fraction of the side.
    An empty mask returns the whole image.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, w, h
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(bx1 - bx0, by1 - by0) * (1 + 2 * margin)
    side = max(side, min_frac * min(h, w))
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    if jitter and rng is not None:
        side *= 1 + rng.uniform(-jitter, jitter)
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
    side = min(side, h, w)
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    s = int(round(side))
    return x0, y0, min(x0 + s, w), min(y0 + s, h)
