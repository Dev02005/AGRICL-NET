import math

import numpy as np
import pytest

from agricl.metrics import SegMeter, dice, iou, r2_score, rmse


def test_iou_dice_half_overlap():
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[0, 1, 1, 0]])
    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a, b) == pytest.approx(0.5)


def test_empty_masks_match():
    z = np.zeros((3, 3))
    assert iou(z, z) == 1.0 and dice(z, z) == 1.0


def test_r2_rmse():
    assert r2_score([1, 2, 3], [1, 2, 3]) == 1.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert math.isnan(r2_score([2, 2], [1, 3]))


def test_meter():
    m = SegMeter()
    m.update(np.array([[1, 1, 0, 0]]), np.array([[1, 1, 0, 0]]))
    m.update(np.array([[1, 0, 0, 0]]), np.array([[1, 1, 0, 0]]))
    s = m.summary()
    assert s["images"] == 2
    assert s["mean_iou"] == pytest.approx(0.75)
    assert s["global_iou"] == pytest.approx(3 / 4)
    assert s["share_rmse"] == pytest.approx(math.sqrt((0 + 25**2) / 2))
