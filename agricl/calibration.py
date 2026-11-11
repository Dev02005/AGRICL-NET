"""Classifier metrics, confidence calibration and unknown-disease detection (NumPy only)."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, 1)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): the T that minimises validation NLL.
    Coarse log-spaced search, then a finer search around the best value."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
    best = grid[np.argmin([nll(logits, labels, t) for t in grid])]
    fine = np.linspace(best * 0.9, best * 1.1, 41)
    return float(fine[np.argmin([nll(logits, labels, t) for t in fine])])


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: gap between confidence and accuracy, binned."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def macro_f1(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Mean F1 over classes that appear in labels or predictions."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):  # average ranks over ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_tpr(known_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Confidence threshold that keeps `tpr` of known-disease images as known."""
    return float(np.quantile(np.asarray(known_scores, float), 1 - tpr))
