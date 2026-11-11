import numpy as np
import pytest

from agricl.calibration import (
    auroc, ece, fit_temperature, macro_f1, nll, softmax, threshold_at_tpr, topk_accuracy,
)
from agricl.roi import roi_box


# ---------------------------------------------------------------- roi_box
def test_roi_empty_mask_is_whole_image():
    assert roi_box(np.zeros((50, 80))) == (0, 0, 80, 50)


def test_roi_square_contains_lesion_and_stays_inside():
    m = np.zeros((100, 200), np.uint8)
    m[40:50, 100:130] = 1
    x0, y0, x1, y1 = roi_box(m, margin=0.15, min_frac=0.0)
    assert x0 <= 100 and x1 >= 130 and y0 <= 40 and y1 >= 50
    assert 0 <= x0 and 0 <= y0 and x1 <= 200 and y1 <= 100
    assert abs((x1 - x0) - (y1 - y0)) <= 1


def test_roi_min_frac_gives_context_for_tiny_spot():
    m = np.zeros((100, 100), np.uint8)
    m[50, 50] = 1
    x0, y0, x1, y1 = roi_box(m, min_frac=0.3)
    assert x1 - x0 >= 29


def test_roi_jitter_stays_inside():
    m = np.zeros((60, 60), np.uint8)
    m[0:10, 50:60] = 1
    rng = np.random.default_rng(0)
    for _ in range(50):
        x0, y0, x1, y1 = roi_box(m, jitter=0.2, rng=rng)
        assert 0 <= x0 < x1 <= 60 and 0 <= y0 < y1 <= 60


# ---------------------------------------------------------- calibration
def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(1), 1)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    true_logits = rng.normal(size=(n, k)) * 2
    labels = np.array([rng.choice(k, p=softmax(true_logits[i:i + 1])[0]) for i in range(n)])
    t = fit_temperature(true_logits * 3, labels)  # model 3x too confident
    assert 2.4 < t < 3.6
    assert nll(true_logits * 3, labels, t) < nll(true_logits * 3, labels, 1.0)


def test_ece_zero_when_perfectly_calibrated_and_correct():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, np.array([0, 1])) == pytest.approx(0.0)


def test_topk_and_macro_f1():
    logits = np.array([[3, 2, 1], [1, 3, 2], [1, 2, 3]], float)
    labels = np.array([1, 1, 2])
    assert topk_accuracy(logits, labels, 1) == pytest.approx(2 / 3)
    assert topk_accuracy(logits, labels, 2) == pytest.approx(1.0)
    assert macro_f1(logits.argmax(1), labels, 3) == pytest.approx((0 + 2 / 3 + 1) / 3)


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5


def test_threshold_keeps_95_percent():
    scores = np.linspace(0, 1, 101)
    t = threshold_at_tpr(scores, 0.95)
    assert np.mean(scores >= t) == pytest.approx(0.95, abs=0.01)
