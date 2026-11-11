"""Demo predictor: previews the app without trained models (NumPy only).

NOT a diagnosis. Lesions are guessed from colour (yellow/brown tissue), the
leaf from the green index, and the "classifier" returns placeholder names.
The app shows a clear DEMO banner whenever this is used.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .advice import advice_for
from .decision import decide
from .leaf import vegetation_mask
from .predictor import min_box_area
from .quality import check_quality
from .result import Diagnosis
from .roi import roi_box
from .severity import compute_severity, lesion_boxes
from .visualize import overlay_masks

DEMO_CLASSES = ["(demo) leaf spot", "(demo) rust", "(demo) blight"]


def colour_lesions(rgb: np.ndarray) -> np.ndarray:
    """Yellow-to-brown, reasonably saturated pixels: a crude stand-in for the lesion model."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), np.float32) / 255.0
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    return ((h >= 10) & (h <= 55) & (s > 0.3) & (v > 0.15)).astype(np.uint8)


class DemoPredictor:
    def diagnose(self, img: Image.Image, explain: bool = True) -> Diagnosis:
        s = time.perf_counter()
        img = img.convert("RGB")
        rgb = np.array(img)
        lesion = colour_lesions(rgb)
        leaf = vegetation_mask(rgb, lesion)
        lesion &= leaf
        sev = compute_severity(leaf, lesion)
        boxes = lesion_boxes(lesion, min_box_area(*lesion.shape))
        decision = decide(np.array([0.62, 0.25, 0.13]), DEMO_CLASSES, 0.5, sev.percent)
        return Diagnosis(
            image=rgb, quality=check_quality(img), lesion=lesion, leaf=leaf, severity=sev, boxes=boxes,
            decision=decision, advice=advice_for(decision.disease, sev.level), crop_box=roi_box(lesion),
            overlay=overlay_masks(rgb, leaf, lesion, boxes), timings={"demo": time.perf_counter() - s}, demo=True,
        )
