"""The single result object the app displays (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .advice import Advice
from .decision import Decision
from .quality import QualityResult
from .severity import SeverityResult


@dataclass
class Diagnosis:
    image: np.ndarray  # HxWx3 uint8
    quality: QualityResult
    lesion: np.ndarray  # HxW uint8
    leaf: np.ndarray  # HxW uint8
    severity: SeverityResult
    boxes: list[tuple[int, int, int, int]]
    decision: Decision
    advice: Advice
    crop_box: tuple[int, int, int, int]
    overlay: np.ndarray  # image with leaf, lesions and boxes drawn
    cam_overlay: np.ndarray | None = None  # Grad-CAM over the classifier's crop
    timings: dict = field(default_factory=dict)  # seconds per stage
    demo: bool = False
