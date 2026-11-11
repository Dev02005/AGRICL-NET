"""Run the edge models on a folder of photos and report time per photo.

Needs only: numpy, pillow, onnxruntime (see requirements-edge.txt), so it runs
on a Raspberry Pi or Jetson.

Usage:
    python scripts/edge_benchmark.py --bundle models/edge_bundle --images path/to/photos [--threads 4]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agricl.edge import EdgePredictor  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="edge bundle folder (edge_bundle.json inside)")
    ap.add_argument("--images", required=True, help="folder of leaf photos")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for ONNX Runtime (default: all)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    photos = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)[: args.limit]
    if not photos:
        sys.exit(f"no photos in {args.images}")
    t = time.perf_counter()
    p = EdgePredictor(args.bundle, threads=args.threads)
    print(f"models loaded in {time.perf_counter() - t:.1f} s")
    p.diagnose(Image.open(photos[0]))  # warm-up

    totals = []
    for path in photos:
        t = time.perf_counter()
        d = p.diagnose(Image.open(path))
        totals.append(time.perf_counter() - t)
        print(f"{path.name:<40} {d.decision.status:<10} {str(d.decision.disease):<32} "
              f"{d.severity.percent:5.1f}% {d.severity.level:<8} {totals[-1] * 1000:6.0f} ms")
    print(f"\nmedian {np.median(totals) * 1000:.0f} ms per photo over {len(totals)} photos "
          f"(threads: {args.threads or 'all'})")


if __name__ == "__main__":
    main()
