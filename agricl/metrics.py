"""Segmentation and severity metrics (NumPy only, so they run anywhere)."""

from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU of two binary masks. Both empty counts as a perfect match."""
    p, g = pred > 0, gt > 0
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / total)


def r2_score(y_true, y_pred) -> float:
    t, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y_true, y_pred) -> float:
    t, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(((t - p) ** 2).mean()))


class SegMeter:
    """Collects per-image IoU/Dice plus pixel totals and lesion shares.

    mean_* = average over images (each image counts equally);
    global_iou = IoU over all pixels pooled (large lesions dominate).
    share = lesion pixels / all image pixels, a stand-in for severity on
    PlantSeg, which has no healthy-leaf labels.
    """

    def __init__(self) -> None:
        self.ious: list[float] = []
        self.dices: list[float] = []
        self.inter = 0
        self.union = 0
        self.true_share: list[float] = []
        self.pred_share: list[float] = []

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        p, g = pred > 0, gt > 0
        self.ious.append(iou(p, g))
        self.dices.append(dice(p, g))
        self.inter += int(np.logical_and(p, g).sum())
        self.union += int(np.logical_or(p, g).sum())
        self.true_share.append(100.0 * g.mean())
        self.pred_share.append(100.0 * p.mean())

    def summary(self) -> dict:
        n = len(self.ious)
        if n == 0:
            return {"images": 0}
        return {
            "images": n,
            "mean_iou": float(np.mean(self.ious)),
            "mean_dice": float(np.mean(self.dices)),
            "global_iou": self.inter / self.union if self.union else 1.0,
            "share_r2": r2_score(self.true_share, self.pred_share),
            "share_rmse": rmse(self.true_share, self.pred_share),
        }
