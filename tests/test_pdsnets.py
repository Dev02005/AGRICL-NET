import numpy as np
import pytest
from PIL import Image

from agricl.data.pdsnets import (
    BACKGROUND, HEALTHY, SPOT, decode_label, find_root, load_image, load_label, load_records,
)
from agricl.eval_pdsnets import clip_to, evaluate_pdsnets


def gray_label(spot_rows: int) -> np.ndarray:
    """40x40: background left half, leaf right half, spots in the top spot_rows of the leaf."""
    a = np.zeros((40, 40), np.uint8)
    a[:, 20:] = 128
    a[:spot_rows, 20:] = 255
    return a


def make_dataset(root, spots=(4, 10, 20)):
    folders = ["cedar_apple_rust", "wheat_stripe_rust", "soybean_bacterial_blight"]
    for i, (folder, s) in enumerate(zip(folders, spots)):
        d = root / "data" / "Dataset" / folder
        d.mkdir(parents=True)
        Image.new("RGB", (40, 40), (40, 160, 40)).save(d / f"img_{i}.jpg")
        Image.new("RGB", (40, 40)).save(d / f"img_{i}_black.png")
        Image.fromarray(gray_label(s)).save(d / f"img_{i}_label.png")
        Image.new("RGB", (40, 40)).save(d / "unlabelled.jpg")  # skipped: no label
    return root / "data" / "Dataset"


def test_records_and_labels(tmp_path):
    root = make_dataset(tmp_path)
    assert find_root(tmp_path) == root
    recs = load_records(root)
    assert [r.disease for r in recs] == ["apple rust", "soybean bacterial blight", "wheat stripe rust"]
    lab = load_label(recs[0])
    assert lab.shape == (40, 40)
    assert (lab[:, :20] == BACKGROUND).all() and lab[0, 30] == SPOT and lab[39, 30] == HEALTHY
    assert load_image(recs[0], (40, 40)).size == (40, 40)


def test_decode_label_variants():
    g = gray_label(5)
    assert np.array_equal(decode_label(g), decode_label(np.stack([g] * 3, -1)))
    idx = np.array([[0, 1, 2]])
    assert decode_label(idx).tolist() == [[0, 1, 2]]
    # resampled edge values snap to the nearest class
    assert decode_label(np.array([[10, 100, 150, 230]])).tolist() == [[0, 1, 1, 2]]


def test_palette_label(tmp_path):
    p = Image.fromarray(np.array([[0, 1, 2]], np.uint8), mode="P")
    p.putpalette([0, 0, 0, 128, 128, 128, 255, 255, 255] + [0] * 759)
    p.save(tmp_path / "x_label.png")
    from agricl.data.pdsnets import PdsRecord
    rec = PdsRecord(tmp_path / "x.jpg", tmp_path / "x_label.png", "d")
    assert load_label(rec).tolist() == [[0, 1, 2]]


def test_find_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_root(tmp_path)


def test_evaluate_perfect_and_leaf_error(tmp_path):
    recs = load_records(make_dataset(tmp_path))
    systems = {
        "perfect": lambda img, rgb, spot, leaf, cache: (spot, leaf),
        "whole image as leaf": lambda img, rgb, spot, leaf, cache: (spot, np.ones_like(leaf)),
        "lesions clipped": lambda img, rgb, spot, leaf, cache: (clip_to(np.ones_like(spot), spot), leaf),
    }
    out = evaluate_pdsnets(recs, systems, out_csv=str(tmp_path / "rows.csv"))
    p = out["perfect"]["all"]
    assert p["images"] == 3 and p["severity_rmse"] == pytest.approx(0) and p["severity_r2"] == pytest.approx(1)
    assert p["mean_lesion_dice"] == 1 and p["mean_leaf_iou"] == 1 and p["level_accuracy"] == 1
    # true severities 10%, 25%, 50% of the leaf; a leaf twice too big halves them
    assert p["mean_true_severity"] == pytest.approx((10 + 25 + 50) / 3)
    w = out["whole image as leaf"]["all"]
    assert w["mean_leaf_iou"] == pytest.approx(0.5)
    assert w["severity_mae"] == pytest.approx((5 + 12.5 + 25) / 3)
    assert out["lesions clipped"]["all"]["severity_rmse"] == pytest.approx(0)
    assert set(out["perfect"]["by_disease"]) == {"apple rust", "soybean bacterial blight", "wheat stripe rust"}
    assert (tmp_path / "rows.csv").read_text().count("\n") == 1 + 3 * 3


def test_cache_is_per_photo(tmp_path):
    recs = load_records(make_dataset(tmp_path))
    calls = []

    def first(img, rgb, spot, leaf, cache):
        cache["pred"] = spot
        calls.append(1)
        return spot, leaf

    def second(img, rgb, spot, leaf, cache):
        assert cache["pred"] is spot  # set by the first system for this same photo
        return cache["pred"], leaf

    evaluate_pdsnets(recs, {"a": first, "b": second})
    assert len(calls) == 3
