"""Summarise the extracted PlantSeg dataset.

Usage: python scripts/inspect_plantseg.py data/plantseg
Writes data/plantseg_classes.csv (one row per disease with split counts).
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PDSNETS_DISEASES = ["soybean bacterial blight", "wheat stripe rust", "cedar apple rust", "apple rust"]


def main(root: str) -> None:
    root = Path(root)
    with open(root / "Metadata.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"metadata rows: {len(rows)}")
    print("splits:", dict(Counter(r["Split"] for r in rows)))

    plants = Counter(r["Plant"] for r in rows)
    diseases = {}
    per_split = defaultdict(Counter)
    for r in rows:
        diseases[r["Disease"]] = (int(r["Index"]), r["Plant"])
        per_split[r["Disease"]][r["Split"]] += 1
    print(f"plants: {len(plants)}  diseases: {len(diseases)}")
    idx = sorted(i for i, _ in diseases.values())
    print(f"class index range: {idx[0]}..{idx[-1]} (unique {len(set(idx))})")

    counts = sorted((sum(c.values()), d) for d, c in per_split.items())
    print("fewest images:", counts[:5])
    print("most images:", counts[-5:])
    totals = np.array([c for c, _ in counts])
    print(f"images per disease: median {np.median(totals):.0f}, min {totals.min()}, max {totals.max()}")
    print("diseases with < 10 test images:", sum(1 for d in per_split if per_split[d]["Test"] < 10))

    ratio = np.array([float(r["Mask ratio"]) for r in rows])
    print(f"lesion share of whole image: median {np.median(ratio):.3f}, "
          f"p90 {np.percentile(ratio, 90):.3f}, max {ratio.max():.3f}")

    print("\ntop plants:", plants.most_common(10))

    print("\nPDSNets disease overlap:")
    for name in PDSNETS_DISEASES:
        hits = [d for d in diseases if name in d.lower()]
        print(f"  {name!r}: {hits if hits else 'not in PlantSeg'}")

    # mask pixel values on a random sample
    random.seed(0)
    sample = random.sample(rows, 40)
    split_dir = {"Training": "train", "Validation": "val", "Test": "test"}
    values = Counter()
    mismatched = 0
    for r in sample:
        m = np.array(Image.open(root / "annotations" / split_dir[r["Split"]] / r["Label file"]))
        vals = set(np.unique(m).tolist())
        values.update(vals)
        fg = vals - {0}
        if fg and fg != {int(r["Index"]) + 1} and fg != {int(r["Index"])}:
            mismatched += 1
    print(f"\nmask dtype/mode sample: {m.dtype}, shape {m.shape}")
    print(f"mask values seen in 40 samples: {sorted(values)[:20]}")
    print(f"masks whose value is not Index or Index+1: {mismatched}/40")

    out = root.parent / "plantseg_classes.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "plant", "disease", "train", "val", "test", "total"])
        for d, (i, p) in sorted(diseases.items(), key=lambda kv: kv[1][0]):
            c = per_split[d]
            w.writerow([i, p, d, c["Training"], c["Validation"], c["Test"], sum(c.values())])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
