"""Grad-CAM (Selvaraju et al., 2017) for timm classifiers and IncrementalNet.

Uses the last feature map (forward_features). timm Swin returns it
channels-last (output_fmt 'NHWC'); CNNs such as MobileNetV3 return NCHW.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _parts(model):
    """(backbone, head_fn) where head_fn maps the feature map to logits."""
    if hasattr(model, "backbone") and hasattr(model, "head"):  # IncrementalNet
        bb = model.backbone
        return bb, lambda f: model.head(bb.forward_head(f))
    return model, model.forward_head


def gradcam(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """x: 1x3xHxW normalised input. Returns an HxW map scaled to 0..1."""
    backbone, head_fn = _parts(model)
    model.eval()
    with torch.enable_grad():
        feats = backbone.forward_features(x)
        feats.retain_grad()
        logits = head_fn(feats)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    a, g = feats.detach(), feats.grad.detach()
    if getattr(backbone, "output_fmt", "NCHW") == "NHWC":
        a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()
