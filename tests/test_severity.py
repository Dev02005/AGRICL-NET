import numpy as np
import pytest

from agricl.severity import (
    compute_severity,
    lesion_boxes,
    severity_from_pdsnets_label,
    severity_level,
)


def test_levels():
    assert severity_level(0) == "Mild"
    assert severity_level(19.9) == "Mild"
    assert severity_level(20) == "Moderate"
    assert severity_level(50) == "Moderate"
    assert severity_level(50.1) == "Severe"


def test_compute_severity_basic():
    leaf = np.zeros((10, 10), np.uint8)
    leaf[:, :5] = 1  # 50 leaf pixels
    lesion = np.zeros_like(leaf)
    lesion[:2, :5] = 1  # 10 lesion pixels
    r = compute_severity(leaf, lesion)
    assert r.percent == pytest.approx(20.0)
    assert r.level == "Moderate"


def test_lesion_counts_as_leaf_even_if_leaf_mask_missed_it():
    leaf = np.zeros((4, 4), np.uint8)
    leaf[:2] = 1  # 8 leaf pixels
    lesion = np.zeros_like(leaf)
    lesion[2, 0] = 1  # 1 lesion pixel outside the leaf mask
    r = compute_severity(leaf, lesion)
    assert r.leaf_pixels == 9
    assert r.lesion_pixels == 1


def test_no_leaf():
    z = np.zeros((3, 3), np.uint8)
    assert compute_severity(z, z).level == "No leaf"


def test_shape_mismatch():
    with pytest.raises(ValueError):
        compute_severity(np.zeros((2, 2)), np.zeros((3, 3)))


def test_pdsnets_ground_truth():
    label = np.array([[0, 1, 1, 2]])  # 0 background, 1 healthy, 2 spot
    assert severity_from_pdsnets_label(label, healthy=1, spot=2) == pytest.approx(100 / 3)


def test_boxes_two_regions():
    m = np.zeros((20, 20), np.uint8)
    m[1:4, 1:4] = 1
    m[10:15, 12:18] = 1
    boxes = sorted(lesion_boxes(m, min_area=1))
    assert boxes == [(1, 1, 4, 4), (12, 10, 18, 15)]


def test_boxes_min_area_filters_noise():
    m = np.zeros((10, 10), np.uint8)
    m[0, 0] = 1
    assert lesion_boxes(m, min_area=5) == []
