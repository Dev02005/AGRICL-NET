"""Photo quality check before diagnosis: too blurred, too dark, too bright, too small."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ANALYSIS_SIDE = 512  # measure on a fixed size so the blur score does not depend on resolution


@dataclass
class QualityResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    sharpness: float = 0.0
    brightness: float = 0.0


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: low values mean a blurred image."""
    g = gray.astype(np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def check_quality(img: Image.Image, min_sharpness: float = 30.0, min_brightness: float = 40.0,
                  max_brightness: float = 225.0, min_side: int = 128) -> QualityResult:
    issues = []
    if min(img.size) < min_side:
        issues.append(f"The photo is very small ({img.width}x{img.height}); use a larger photo.")
    small = img.convert("L")
    small.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))
    gray = np.asarray(small)
    sharp = laplacian_variance(gray)
    bright = float(gray.mean())
    if sharp < min_sharpness:
        issues.append("The photo looks blurred; hold the camera still and focus on the leaf.")
    if bright < min_brightness:
        issues.append("The photo is too dark; take it in daylight.")
    if bright > max_brightness:
        issues.append("The photo is too bright; avoid direct sun glare on the leaf.")
    return QualityResult(not issues, issues, sharp, bright)
