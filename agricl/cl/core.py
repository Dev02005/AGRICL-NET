"""Continual-learning building blocks that need only NumPy (tested locally).

- make_tasks: class-incremental rounds grouped by crop (plant).
- herding_select: iCaRL exemplar selection (Rebuffi et al., 2017).
- average_incremental_accuracy / forgetting: standard CL metrics.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def make_tasks(records, n_tasks: int = 5, base_frac: float = 0.5, seed: int = 0):
    """Split diseases into rounds by plant: a base round with about base_frac of
    the diseases, then n_tasks - 1 rounds with balanced disease counts.
    Returns (diseases per round, plants per round), both sorted lists."""
    by_plant = defaultdict(set)
    for r in records:
        by_plant[r.plant].add(r.disease)
    plants = sorted(by_plant)
    random.Random(seed).shuffle(plants)
    total = sum(len(v) for v in by_plant.values())

    base, count, i = [], 0, 0
    while i < len(plants) and count < base_frac * total:
        base.append(plants[i])
        count += len(by_plant[plants[i]])
        i += 1
    rest = plants[i:]
    groups, sizes = [[] for _ in range(n_tasks - 1)], [0] * (n_tasks - 1)
    for p in sorted(rest, key=lambda p: (-len(by_plant[p]), p)):  # largest first, to the smallest round
        j = int(np.argmin(sizes))
        groups[j].append(p)
        sizes[j] += len(by_plant[p])
    rounds = [base] + groups
    if any(not g for g in rounds):
        raise ValueError(f"not enough plants for {n_tasks} rounds")
    diseases = [sorted(d for p in g for d in by_plant[p]) for g in rounds]
    return diseases, [sorted(g) for g in rounds]


def herding_select(features: np.ndarray, m: int) -> list[int]:
    """Pick m rows whose running mean best approximates the class mean
    (features are L2-normalised first, as in iCaRL). Returns row indices."""
    f = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    mu = f.mean(axis=0)
    chosen: list[int] = []
    running = np.zeros_like(mu)
    for k in range(1, min(m, len(f)) + 1):
        dist = np.linalg.norm(mu - (running + f) / k, axis=1)
        dist[chosen] = np.inf
        i = int(np.argmin(dist))
        chosen.append(i)
        running += f[i]
    return chosen


def average_incremental_accuracy(step_accuracy) -> float:
    """Mean over rounds of the accuracy on all diseases seen so far."""
    return float(np.mean(step_accuracy))


def forgetting(acc: np.ndarray) -> float:
    """acc[t, j] = accuracy on round j's diseases after training round t (NaN if j > t).
    Forgetting of round j = best accuracy it ever had before the last round minus
    its accuracy after the last round; averaged over all rounds but the last."""
    acc = np.asarray(acc, float)
    last = acc.shape[0] - 1
    if last == 0:
        return 0.0
    drops = [np.nanmax(acc[j:last, j]) - acc[last, j] for j in range(last)]
    return float(np.mean(drops))
