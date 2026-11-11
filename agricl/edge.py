"""Offline edge predictor: ONNX Runtime + NumPy + Pillow only (no PyTorch).

Runs the distilled students (LinkNet-ResNet18 lesion model, MobileNetV3
classifier), FP32 or INT8, with the colour-index leaf mask, so it fits a
Raspberry Pi or Jetson. No Grad-CAM here (it needs gradients).

Edge bundle layout (written by notebook 06):
    edge_bundle.json   {"lesion": {"onnx", "img_size"},
                        "classifier": {"onnx", "img_size", "classes", "calibration"}}
    *.onnx, classes.json, calibration.json
Preprocessing matches training: PIL resize (bilinear for lesions, bicubic for
the classifier) and ImageNet normalisation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .advice import advice_for
from .calibration import softmax
from .decision import decide
from .leaf import vegetation_mask
from .predictor import min_box_area
from .quality import check_quality
from .result import Diagnosis
from .roi import roi_box
from .severity import compute_severity, lesion_boxes
from .visualize import overlay_masks

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

Model = Callable[[np.ndarray], np.ndarray]  # (1,3,S,S) float32 -> output array


class OnnxModel:
    """Thin ONNX Runtime wrapper: call it with a float32 NCHW array."""

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]


def to_input(img: Image.Image, size: int, resample=Image.BILINEAR) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((size, size), resample), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def lesion_from_logits(logits: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(1,1,S,S) logits -> HxW uint8 mask. Bilinear resize of the logits, then
    logit > 0 (same as sigmoid > 0.5), as in the PyTorch predictor."""
    h, w = out_hw
    up = Image.fromarray(logits[0, 0].astype(np.float32), mode="F").resize((w, h), Image.BILINEAR)
    return (np.asarray(up) > 0).astype(np.uint8)


class EdgePredictor:
    def __init__(self, bundle_dir: str | Path | None = None, *, lesion_model: Model | None = None,
                 classifier: Model | None = None, classes: list[str] | None = None,
                 temperature: float = 1.0, threshold: float = 0.0,
                 lesion_size: int = 352, cls_size: int = 224, threads: int | None = None) -> None:
        if bundle_dir is not None:
            d = Path(bundle_dir)
            cfg = json.loads((d / "edge_bundle.json").read_text())
            les, cls = cfg["lesion"], cfg["classifier"]
            lesion_model = OnnxModel(d / les["onnx"], threads)
            classifier = OnnxModel(d / cls["onnx"], threads)
            classes = json.loads((d / cls["classes"]).read_text())
            cal = json.loads((d / cls["calibration"]).read_text())
            temperature, threshold = float(cal["temperature"]), float(cal["unknown_threshold"])
            lesion_size, cls_size = int(les["img_size"]), int(cls["img_size"])
        if lesion_model is None or classifier is None or not classes:
            raise ValueError("give a bundle_dir, or lesion_model, classifier and classes")
        self.lesion_model, self.classifier, self.classes = lesion_model, classifier, classes
        self.temperature, self.threshold = temperature, threshold
        self.lesion_size, self.cls_size = lesion_size, cls_size

    def diagnose(self, img: Image.Image, explain: bool = False) -> Diagnosis:
        t = {}
        s = time.perf_counter()
        img = img.convert("RGB")
        rgb = np.array(img)
        quality = check_quality(img)
        lesion = lesion_from_logits(self.lesion_model(to_input(img, self.lesion_size)), rgb.shape[:2])
        t["lesion"] = time.perf_counter() - s

        s = time.perf_counter()
        leaf = vegetation_mask(rgb, lesion)
        sev = compute_severity(leaf, lesion)
        boxes = lesion_boxes(lesion, min_box_area(*lesion.shape))
        t["leaf_and_severity"] = time.perf_counter() - s

        s = time.perf_counter()
        crop_box = roi_box(lesion)
        logits = self.classifier(to_input(img.crop(crop_box), self.cls_size, Image.BICUBIC))
        probs = softmax(np.asarray(logits, np.float32).reshape(1, -1), self.temperature)[0]
        decision = decide(probs, self.classes, self.threshold, sev.percent)
        t["classify"] = time.perf_counter() - s

        return Diagnosis(
            image=rgb, quality=quality, lesion=lesion, leaf=leaf, severity=sev, boxes=boxes,
            decision=decision, advice=advice_for(decision.disease, sev.level), crop_box=crop_box,
            overlay=overlay_masks(rgb, leaf, lesion, boxes), cam_overlay=None, timings=t,
        )
