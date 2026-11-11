"""Full AgriCL-Net diagnosis from one photo, using a model bundle folder.

Bundle layout (written by notebook 05):
    agricl_bundle.json      settings, see BUNDLE_KEYS below
    lesion_best.pt          trainable SAM2-UNet weights (notebook 01)
    sam2_hiera_<size>.pt    SAM2 checkpoint (frozen encoder, also used for leaves)
    SAM2-UNet/              SAM2-UNet code at the pinned commit
    classifier.pt           classifier weights (notebook 03 or 04)
    classes.json            class names in output order
    calibration.json        {"temperature": T, "unknown_threshold": thr}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .advice import advice_for
from .calibration import softmax
from .decision import decide
from .leaf import vegetation_mask
from .quality import check_quality
from .result import Diagnosis
from .roi import roi_box
from .severity import compute_severity, lesion_boxes
from .visualize import overlay_heatmap, overlay_masks

BUNDLE_KEYS = {
    "lesion": ["size", "img_size", "weights", "sam2_checkpoint", "sam2unet_repo"],
    "classifier": ["kind", "model_name", "weights", "classes", "calibration"],
    "leaf": ["method"],
}


def min_box_area(h: int, w: int) -> int:
    """Ignore specks: lesion regions smaller than 0.05% of the image get no box."""
    return max(50, int(0.0005 * h * w))


class AgriCLPredictor:
    def __init__(self, bundle_dir: str | Path, device: str | None = None, leaf_method: str | None = None) -> None:
        import torch

        from .cl.model import IncrementalNet
        from .models.classifier import build_classifier
        from .pipeline import load_lesion_model

        self.dir = Path(bundle_dir)
        cfg = json.loads((self.dir / "agricl_bundle.json").read_text())
        for part, keys in BUNDLE_KEYS.items():
            missing = [k for k in keys if k not in cfg.get(part, {})]
            if missing:
                raise ValueError(f"agricl_bundle.json: {part} is missing {missing}")
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        les, cls = cfg["lesion"], cfg["classifier"]
        repo = str(self.dir / les["sam2unet_repo"])
        sam_ckpt = str(self.dir / les["sam2_checkpoint"])

        self.lesion_size = int(les["img_size"])
        self.lesion_model = load_lesion_model(repo, les["size"], sam_ckpt, str(self.dir / les["weights"]), self.device)

        self.leaf_method = leaf_method or cfg["leaf"]["method"]
        self.leaf_segmenter = None
        if self.leaf_method == "sam2":
            from .leaf import SAM2LeafSegmenter

            self.leaf_segmenter = SAM2LeafSegmenter(repo, les["size"], sam_ckpt, device=self.device)

        self.classes = json.loads((self.dir / cls["classes"]).read_text())
        cal = json.loads((self.dir / cls["calibration"]).read_text())
        self.temperature, self.threshold = float(cal["temperature"]), float(cal["unknown_threshold"])
        if cls["kind"] == "incremental":
            self.classifier = IncrementalNet(len(self.classes), cls["model_name"], pretrained=False)
        else:
            self.classifier = build_classifier(len(self.classes), cls["model_name"], pretrained=False)
        state = torch.load(self.dir / cls["weights"], map_location="cpu", weights_only=True)
        self.classifier.load_state_dict(state)
        self.classifier.to(self.device).eval()
        self.cls_size = int(cls.get("img_size", 224))

    def diagnose(self, img: Image.Image, explain: bool = True) -> Diagnosis:
        import torch

        from .data.cls_dataset import eval_transform
        from .explain import gradcam
        from .train_lesion import predict_mask

        t = {}
        s = time.perf_counter()
        img = img.convert("RGB")
        rgb = np.array(img)
        quality = check_quality(img)
        t["quality"] = time.perf_counter() - s

        s = time.perf_counter()
        lesion = predict_mask(self.lesion_model, img, self.lesion_size, self.device)
        t["lesion"] = time.perf_counter() - s

        s = time.perf_counter()
        leaf = self.leaf_segmenter.segment(rgb, lesion) if self.leaf_segmenter else vegetation_mask(rgb, lesion)
        sev = compute_severity(leaf, lesion)
        boxes = lesion_boxes(lesion, min_box_area(*lesion.shape))
        t["leaf_and_severity"] = time.perf_counter() - s

        s = time.perf_counter()
        crop_box = roi_box(lesion)
        crop = img.crop(crop_box)
        x = eval_transform(self.cls_size)(crop)[None].to(self.device)
        with torch.no_grad():
            logits = self.classifier(x).float().cpu().numpy()
        probs = softmax(logits, self.temperature)[0]
        decision = decide(probs, self.classes, self.threshold, sev.percent)
        t["classify"] = time.perf_counter() - s

        cam_overlay = None
        if explain:
            s = time.perf_counter()
            cam = gradcam(self.classifier, x, int(np.argmax(probs)))
            crop_rgb = np.array(crop.resize((self.cls_size, self.cls_size)))
            cam_overlay = overlay_heatmap(crop_rgb, cam)
            t["explain"] = time.perf_counter() - s

        return Diagnosis(
            image=rgb, quality=quality, lesion=lesion, leaf=leaf, severity=sev, boxes=boxes,
            decision=decision, advice=advice_for(decision.disease, sev.level), crop_box=crop_box,
            overlay=overlay_masks(rgb, leaf, lesion, boxes), cam_overlay=cam_overlay, timings=t,
        )
