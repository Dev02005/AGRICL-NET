import numpy as np
from PIL import Image

from agricl.data.plantseg import load_lesion_mask, load_records, split_seen_unseen

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
