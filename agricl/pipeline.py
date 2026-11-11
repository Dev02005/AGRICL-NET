"""Severity pipeline: photo -> lesion mask, leaf mask, severity, boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .leaf import vegetation_mask
from .models.sam2unet import LesionSAM2UNet
from .severity import SeverityResult, compute_severity, lesion_boxes
from .train_lesion import _load_trainable, predict_mask


@dataclass
class SeverityDiagnosis:
    lesion: np.ndarray  # HxW uint8
    leaf: np.ndarray  # HxW uint8
    severity: SeverityResult
    boxes: list[tuple[int, int, int, int]]


def load_lesion_model(repo_dir: str, size: str, checkpoint: str, best_path: str, device: str = "cuda"):
    """SAM2-UNet with trained weights from a train_lesion run (best.pt)."""
    model = LesionSAM2UNet(repo_dir, size, checkpoint)
    _load_trainable(model, torch.load(best_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def diagnose_severity(img: Image.Image, lesion_model, img_size: int, device: str,
                      leaf_segmenter=None, lesion: np.ndarray | None = None) -> SeverityDiagnosis:
    """leaf_segmenter=None uses the colour (ExG) fallback. Pass `lesion` to
    reuse a known lesion mask (e.g. ground truth) instead of predicting one."""
    img = img.convert("RGB")
    rgb = np.array(img)
    if lesion is None:
        lesion = predict_mask(lesion_model, img, img_size, device)
    leaf = leaf_segmenter.segment(rgb, lesion) if leaf_segmenter else vegetation_mask(rgb, lesion)
    return SeverityDiagnosis(lesion, leaf, compute_severity(leaf, lesion), lesion_boxes(lesion))
