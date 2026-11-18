import numpy as np
from PIL import Image

from agricl.data.plantseg import load_image, load_lesion_mask, load_records, resized_copy, split_seen_unseen

HEADER = "Name,Index,Plant,Disease,Resolution,Label file,Mask ratio,URL,License,Split\n"


def make_dataset(tmp_path):
    root = tmp_path / "plantseg"
    rows = [
        ("a_1.jpg", 0, "Apple", "apple rust", "a_1.png", "Training"),
        ("w_1.jpg", 7, "Wheat", "wheat leaf rust", "w_1.png", "Test"),
    ]
    lines = [HEADER]
    for name, idx, plant, dis, lab, split in rows:
        lines.append(f"{name},{idx},{plant},{dis},4x4,{lab},0.25,http://x,CC-BY-NC,{split}\n")
        sd = {"Training": "train", "Test": "test"}[split]
        (root / "images" / sd).mkdir(parents=True, exist_ok=True)
        (root / "annotations" / sd).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(root / "images" / sd / name)
        m = np.zeros((4, 4), np.uint8)
        m[:2, :2] = idx + 1  # PlantSeg stores Index + 1
        Image.fromarray(m).save(root / "annotations" / sd / lab)
    (root / "Metadata.csv").write_text("﻿" + "".join(lines), encoding="utf-8")
    return root


def test_load_and_split(tmp_path):
    recs = load_records(make_dataset(tmp_path))
    assert [r.split for r in recs] == ["train", "test"]
    seen, unseen = split_seen_unseen(recs)
    assert [r.disease for r in unseen] == ["apple rust"]
    assert [r.disease for r in seen] == ["wheat leaf rust"]


def test_mask_is_binary_even_for_index_zero(tmp_path):
    recs = load_records(make_dataset(tmp_path))
    m = load_lesion_mask(recs[0])  # Index 0 -> stored value 1
    assert m.dtype == np.uint8
    assert m.sum() == 4 and set(np.unique(m)) == {0, 1}


def test_resized_copy_keeps_mask_aligned(tmp_path):
    root = tmp_path / "big"
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations" / "train").mkdir(parents=True)
    Image.new("RGB", (1000, 500), (0, 200, 0)).save(root / "images" / "train" / "a.jpg")
    m = np.zeros((500, 1000), np.uint8)
    m[:, 500:] = 8  # right half is lesion
    Image.fromarray(m).save(root / "annotations" / "train" / "a.png")
    (root / "Metadata.csv").write_text(
        HEADER + "a.jpg,7,Apple,apple scab,1000x500,a.png,0.5,http://x,CC-BY-NC,Training\n", encoding="utf-8")
    small = resized_copy(load_records(root), tmp_path / "small", max_side=200)
    img = Image.open(small[0].image)
    mask = load_lesion_mask(small[0])
    assert img.size == (200, 100) and mask.shape == (100, 200)
    assert mask[:, :95].sum() == 0 and mask[:, 105:].all()
    again = resized_copy(load_records(root), tmp_path / "small", max_side=200)  # reuses files
    assert again[0].image == small[0].image


def test_exif_rotated_photo_matches_its_mask(tmp_path):
    # Phone photo stored landscape with EXIF orientation 6 (shown rotated 90
    # degrees clockwise); the mask was drawn on the upright, portrait view.
    root = tmp_path / "rot"
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations" / "train").mkdir(parents=True)
    raw = np.zeros((40, 80, 3), np.uint8)
    raw[:, :40] = (255, 0, 0)  # left half of the stored pixels: red lesion
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(raw).save(root / "images" / "train" / "r.jpg", exif=exif, quality=95)
    m = np.zeros((80, 40), np.uint8)
    m[:40, :] = 5  # after a 90 degree clockwise turn the stored left half is the top half
    Image.fromarray(m).save(root / "annotations" / "train" / "r.png")
    (root / "Metadata.csv").write_text(
        HEADER + "r.jpg,4,Apple,apple scab,40x80,r.png,0.5,http://x,CC-BY-NC,Training\n", encoding="utf-8")
    rec = load_records(root)[0]

    img = np.array(load_image(rec))
    mask = load_lesion_mask(rec)
    assert img.shape[:2] == mask.shape == (80, 40)
    red = (img[..., 0] > 128) & (img[..., 1] < 100)
    assert red[mask == 1].mean() > 0.95 and red[mask == 0].mean() < 0.05

    small = resized_copy([rec], tmp_path / "small", max_side=40)[0]
    s_img, s_mask = Image.open(small.image), load_lesion_mask(small)
    assert s_img.size == (20, 40) and s_mask.shape == (40, 20)
    assert s_mask[:18].all() and s_mask[22:].sum() == 0
