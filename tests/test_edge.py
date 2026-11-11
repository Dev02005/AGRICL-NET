import json

import numpy as np
import pytest
from PIL import Image

from agricl.edge import MEAN, STD, EdgePredictor, lesion_from_logits, to_input

CLASSES = ["a rust", "b blight", "c spot"]


def leaf_photo():
    """Green leaf filling the left 3/4, soil on the right; a brown patch on the leaf."""
    a = np.zeros((200, 300, 3), np.uint8)
    a[:] = (130, 100, 70)
    a[:, :225] = (40, 160, 40)
    a[60:120, 60:120] = (120, 80, 30)
    return Image.fromarray(a)


def fake_lesion_model(x):
    """Marks the brown patch: logits > 0 where the input is reddish-brown."""
    assert x.dtype == np.float32 and x.shape[:2] == (1, 3)
    rgb = x[0].transpose(1, 2, 0) * STD + MEAN
    brown = (rgb[..., 0] > rgb[..., 1]) & (rgb[..., 2] < 0.2)  # patch: blue 0.12; soil: blue 0.27
    return np.where(brown, 5.0, -5.0)[None, None].astype(np.float32)


def fake_classifier(x):
    assert x.shape == (1, 3, 224, 224)
    return np.array([[4.0, 1.0, 0.0]], np.float32)


def test_to_input_normalises():
    x = to_input(Image.new("RGB", (10, 20), (255, 255, 255)), 8)
    assert x.shape == (1, 3, 8, 8) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], (1 - MEAN) / STD, atol=1e-5)


def test_lesion_from_logits_resizes():
    logits = np.full((1, 1, 4, 4), -1.0, np.float32)
    logits[0, 0, :, 2:] = 1.0
    m = lesion_from_logits(logits, (8, 16))
    assert m.shape == (8, 16) and m[:, :6].sum() == 0 and m[:, 10:].all()


def test_edge_predictor_end_to_end():
    p = EdgePredictor(lesion_model=fake_lesion_model, classifier=fake_classifier,
                      classes=CLASSES, temperature=1.0, threshold=0.5, lesion_size=64)
    d = p.diagnose(leaf_photo())
    assert d.lesion.shape == (200, 300)
    assert d.lesion[90, 90] == 1 and d.lesion[10, 10] == 0
    # 60x60 lesion on a 200x225 leaf -> about 8%
    assert d.severity.percent == pytest.approx(100 * 3600 / (200 * 225), abs=2.5)
    assert d.severity.level == "Mild"
    assert d.decision.status == "disease" and d.decision.disease == "a rust"
    assert len(d.boxes) == 1 and d.cam_overlay is None and not d.demo
    x0, y0, x1, y1 = d.crop_box
    assert x0 <= 60 and y0 <= 60 and x1 >= 120 and y1 >= 120  # crop surrounds the lesion


def test_edge_predictor_unknown_when_flat_logits():
    p = EdgePredictor(lesion_model=fake_lesion_model, classifier=lambda x: np.zeros((1, 3), np.float32),
                      classes=CLASSES, threshold=0.5, lesion_size=64)
    assert p.diagnose(leaf_photo()).decision.status == "unknown"


def test_edge_predictor_needs_models():
    with pytest.raises(ValueError):
        EdgePredictor()


# ------------------------------------------------------------- edge_eval
def test_latency_and_top1_with_fake_models():
    from agricl.edge_eval import classifier_top1, latency_ms

    calls = []
    model = lambda x: (calls.append(1), np.array([[0.0, 2.0, 1.0]], np.float32))[1]  # noqa: E731
    assert latency_ms(model, np.zeros((1, 3, 8, 8), np.float32), runs=4, warmup=2) >= 0
    assert len(calls) == 6
    inputs = [np.zeros((1, 3, 8, 8), np.float32)] * 3
    assert classifier_top1(model, inputs, np.array([1, 1, 0])) == pytest.approx(2 / 3)


def test_lesion_scores_with_fake_model(tmp_path):
    from types import SimpleNamespace

    from agricl.edge_eval import lesion_scores

    m = np.zeros((40, 40), np.uint8)
    m[:, 20:] = 3
    Image.fromarray(m).save(tmp_path / "a.png")
    rec = SimpleNamespace(mask=tmp_path / "a.png")
    logits = np.full((1, 1, 8, 8), -3.0, np.float32)
    logits[..., 4:] = 3.0
    s = lesion_scores(lambda x: logits, [rec], [np.zeros((1, 3, 8, 8), np.float32)])
    assert s["images"] == 1 and s["mean_dice"] > 0.9
