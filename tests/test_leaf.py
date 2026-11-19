import numpy as np

from agricl.leaf import choose_mask, erode, excess_green, otsu_threshold, sample_points, vegetation_mask


def green_leaf_on_soil():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:] = (120, 90, 60)  # brown soil
    img[10:30, 10:30] = (40, 160, 40)  # green leaf
    img[18:22, 18:22] = (110, 80, 30)  # brown lesion inside the leaf
    lesion = np.zeros((40, 40), np.uint8)
    lesion[18:22, 18:22] = 1
    return img, lesion


def test_excess_green_higher_on_leaf():
    img, _ = green_leaf_on_soil()
    exg = excess_green(img)
    assert exg[15, 15] > exg[2, 2]


def test_otsu_splits_two_groups():
    v = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    t = otsu_threshold(v)
    assert 0.1 <= t < 0.9


def test_vegetation_mask_includes_lesion():
    img, lesion = green_leaf_on_soil()
    veg = vegetation_mask(img, lesion)
    assert veg[15, 15] == 1 and veg[2, 2] == 0
    assert veg[19, 19] == 1  # brown lesion added back
    assert veg.sum() == 400  # exactly the 20x20 leaf


def test_erode_shrinks_square():
    m = np.zeros((10, 10), np.uint8)
    m[2:8, 2:8] = 1
    assert erode(m, 1).sum() == 16


def test_sample_points_inside_mask():
    m = np.zeros((30, 30), np.uint8)
    m[5:25, 5:25] = 1
    pts = sample_points(m, 10, np.random.default_rng(0))
    assert pts.shape == (10, 2)
    assert all(m[int(y), int(x)] for x, y in pts)


def test_sample_points_empty():
    assert sample_points(np.zeros((5, 5)), 3, np.random.default_rng(0)).shape == (0, 2)


def test_choose_mask_prefers_agreement():
    ref = np.zeros((4, 4), np.uint8)
    ref[:2] = 1
    good = ref.copy()
    bad = np.ones_like(ref)
    masks = np.stack([bad, good])
    assert (choose_mask(masks, np.array([0.99, 0.5]), ref) == good).all()


# ---------------------------------------------------------------- lesion-anchored leaf
from agricl.leaf import anchored_vegetation_mask, choose_leaf_mask, dilate, label_components  # noqa: E402


def two_leaves():
    """Diseased leaf on the left, a second healthy leaf on the right, soil between."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[:] = (120, 90, 60)
    img[10:30, 5:25] = (40, 160, 40)
    img[10:30, 35:55] = (40, 160, 40)
    img[18:22, 12:16] = (110, 80, 30)
    lesion = np.zeros((40, 60), np.uint8)
    lesion[18:22, 12:16] = 1
    return img, lesion


def test_label_components_and_dilate():
    m = np.zeros((10, 10), np.uint8)
    m[1:3, 1:3] = 1
    m[6:9, 6:9] = 1
    labels, n = label_components(m)
    assert n == 2 and labels[1, 1] != labels[7, 7] and labels[0, 9] == 0
    assert dilate(m[:5, :5], 1).sum() == 16


def test_anchored_vegetation_keeps_only_the_diseased_leaf():
    img, lesion = two_leaves()
    plain = vegetation_mask(img, lesion)
    anchored = anchored_vegetation_mask(img, lesion)
    assert plain[20, 45] == 1 and anchored[20, 45] == 0  # other leaf dropped
    assert anchored[12, 8] == 1 and anchored[19, 13] == 1  # diseased leaf and its lesion kept
    assert anchored.sum() == 400
    assert (anchored_vegetation_mask(img, None) == vegetation_mask(img)).all()


def test_choose_leaf_mask_picks_smallest_leaf_containing_lesions():
    lesion = np.zeros((20, 20), np.uint8)
    lesion[8:10, 8:10] = 1                   # 4 px
    spot = lesion.copy()                     # just the spot: rejected
    leaf = np.zeros_like(lesion)
    leaf[4:14, 4:14] = 1                     # the leaf: chosen
    plant = np.zeros_like(lesion)
    plant[1:17, 1:17] = 1                    # bigger, covers as well: not chosen
    scene = np.ones_like(lesion)             # fills the photo: rejected
    other = np.zeros_like(lesion)
    other[15:20, 15:20] = 1                  # no lesion inside: rejected
    masks = np.stack([spot, plant, leaf, scene, other])
    out = choose_leaf_mask(masks, np.array([0.99, 0.95, 0.8, 0.99, 0.99]), lesion)
    assert (out == leaf).all()
    assert choose_leaf_mask(np.stack([spot, scene]), np.array([0.9, 0.9]), lesion) is None
    assert choose_leaf_mask(masks, np.ones(5), np.zeros_like(lesion)) is None
