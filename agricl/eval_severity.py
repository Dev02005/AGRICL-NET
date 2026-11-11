"""How much does the lesion model change severity, on seen vs. unseen diseases?

For each image the same leaf mask is used twice: with the true lesion mask
(reference severity) and with the predicted lesion mask. The gap between the
two isolates the lesion model's error. If unseen diseases show a similar gap
to seen ones, severity works on diseases the system was never trained on.
True severity (with a true leaf mask) needs the PDSNets dataset.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .data.plantseg import load_image, load_lesion_mask
from .leaf import mask_iou, vegetation_mask
from .metrics import dice, r2_score, rmse
from .severity import compute_severity
from .train_lesion import predict_mask


def severity_agreement(groups: dict, lesion_model, img_size: int, device: str,
                       leaf_segmenter=None, out_csv: str | None = None) -> dict:
    """groups: {"seen": [Record, ...], "unseen": [...]}. Returns per-group summary."""
    rows = []
    for group, records in groups.items():
        for r in records:
            img = load_image(r)
            rgb = np.array(img)
            gt = load_lesion_mask(r)
            pred = predict_mask(lesion_model, img, img_size, device)
            exg_leaf = vegetation_mask(rgb, gt)
            leaf = leaf_segmenter.segment(rgb, gt) if leaf_segmenter else exg_leaf
            ref, est = compute_severity(leaf, gt), compute_severity(leaf, pred)
            rows.append({
                "group": group, "image": r.image.name, "disease": r.disease,
                "severity_true_lesion": ref.percent, "severity_pred_lesion": est.percent,
                "level_true": ref.level, "level_pred": est.level,
                "lesion_dice": dice(pred, gt), "leaf_share": float(leaf.mean()),
                "leaf_iou_sam_vs_exg": mask_iou(leaf, exg_leaf),
            })
    if out_csv and rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    by = defaultdict(list)
    for row in rows:
        by[row["group"]].append(row)
    summary = {}
    for group, rs in by.items():
        t = [x["severity_true_lesion"] for x in rs]
        p = [x["severity_pred_lesion"] for x in rs]
        summary[group] = {
            "images": len(rs),
            "severity_r2": r2_score(t, p),
            "severity_rmse": rmse(t, p),
            "level_accuracy": float(np.mean([x["level_true"] == x["level_pred"] for x in rs])),
            "mean_lesion_dice": float(np.mean([x["lesion_dice"] for x in rs])),
            "median_leaf_iou_sam_vs_exg": float(np.median([x["leaf_iou_sam_vs_exg"] for x in rs])),
        }
    return summary
