import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from agricl.advice import DISCLAIMER, advice_for, disease_type
from agricl.decision import decide
from agricl.quality import check_quality, laplacian_variance
from agricl.visualize import heatmap, overlay_heatmap, overlay_masks

CLASSES_CSV = Path(__file__).resolve().parent.parent / "data" / "plantseg_classes.csv"


# ------------------------------------------------------------------ advice
@pytest.mark.parametrize("name,kind", [
    ("tomato mosaic virus", "vector"),
    ("grapevine leafroll disease", "vector"),
    ("cabbage black rot", "bacterial"),
    ("grape black rot", "fungal"),
    ("cucumber bacterial wilt", "bacterial"),
    ("banana panama disease", "wilt"),
    ("bell pepper blossom end rot", "physiological"),
    ("wheat stripe rust", "fungal"),
    ("tomato yellow leaf curl virus", "vector"),
    ("peach leaf curl", "fungal"),
])
def test_disease_types(name, kind):
    assert disease_type(name) == kind


@pytest.mark.skipif(not CLASSES_CSV.exists(), reason="needs data/plantseg_classes.csv")
def test_every_plantseg_disease_gets_a_specific_type():
    names = [r["disease"] for r in csv.DictReader(open(CLASSES_CSV, encoding="utf-8"))]
    general = [n for n in names if disease_type(n) == "general"]
    assert general == [], general


def test_advice_has_severity_note_and_disclaimer():
    a = advice_for("potato late blight", "Severe")
    assert a.disease_type == "fungal" and "Severe" in a.severity_note and a.disclaimer == DISCLAIMER
    assert all("ml" not in s and "kg" not in s for s in a.steps)  # never prescribes a dose


def test_unknown_disease_advice_refers_to_expert():
    a = advice_for(None, "Moderate")
    assert "expert" in a.steps[0]


# ---------------------------------------------------------------- decision
NAMES = ["a rust", "b blight", "c mosaic virus", "d spot"]


def test_decide_disease():
    d = decide(np.array([0.9, 0.05, 0.03, 0.02]), NAMES, threshold=0.5, lesion_share=12)
    assert d.status == "disease" and d.disease == "a rust" and [n for n, _ in d.top3] == NAMES[:3]


def test_decide_unknown_when_not_confident():
    d = decide(np.array([0.3, 0.3, 0.2, 0.2]), NAMES, threshold=0.5, lesion_share=12)
    assert d.status == "unknown" and d.disease is None


def test_decide_uncertain_and_no_disease_without_lesions():
    assert decide(np.array([0.9, 0.05, 0.03, 0.02]), NAMES, 0.5, 0.1).status == "uncertain"
    assert decide(np.array([0.3, 0.3, 0.2, 0.2]), NAMES, 0.5, 0.1).status == "no_disease"


# ---------------------------------------------------------------- quality
def checkerboard(n=256, sq=8):
    yy, xx = np.mgrid[:n, :n]
    return Image.fromarray((((yy // sq + xx // sq) % 2) * 200 + 30).astype(np.uint8)).convert("RGB")


def test_sharp_image_passes_and_blurred_fails():
    sharp = checkerboard()
    assert check_quality(sharp).ok
    blurred = sharp.filter(ImageFilter.GaussianBlur(8))
    q = check_quality(blurred)
    assert not q.ok and any("blurred" in i for i in q.issues)


def test_dark_and_small_images_flagged():
    dark = Image.fromarray((np.asarray(checkerboard()) * 0.1).astype(np.uint8))
    assert any("dark" in i for i in check_quality(dark).issues)
    assert any("small" in i for i in check_quality(checkerboard(64)).issues)


def test_laplacian_flat_is_zero():
    assert laplacian_variance(np.full((10, 10), 7)) == 0.0


# --------------------------------------------------------------- visualize
def test_overlay_colours_and_boxes():
    rgb = np.full((20, 20, 3), 100, np.uint8)
    lesion = np.zeros((20, 20), np.uint8)
    lesion[5:8, 5:8] = 1
    out = overlay_masks(rgb, np.ones((20, 20)), lesion, boxes=[(2, 2, 12, 12)], box_width=1)
    assert out[6, 6, 0] > out[6, 6, 1]  # lesion is red
    assert out[15, 15, 1] > out[15, 15, 0]  # leaf is green-tinted
    assert tuple(out[2, 5]) == (255, 220, 0)  # box edge
    assert out.shape == rgb.shape and out.dtype == np.uint8


def test_heatmap_ends():
    h = heatmap(np.array([[0.0, 1.0]]))
    assert h[0, 0, 2] > h[0, 0, 0] and h[0, 1, 0] > h[0, 1, 2]
    assert overlay_heatmap(np.zeros((1, 2, 3), np.uint8), np.array([[0.0, 1.0]])).shape == (1, 2, 3)
