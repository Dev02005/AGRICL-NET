"""Print the layout of a dataset zip without extracting it.

Usage: python scripts/inspect_zip.py data/raw/plantseg.zip
"""

import sys
import zipfile
from collections import Counter
from pathlib import PurePosixPath


def main(path: str) -> None:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        print(f"files: {len(names)}")

        dirs = Counter(str(PurePosixPath(n).parent) for n in names)
        print("\nfiles per folder (top 30):")
        for d, c in dirs.most_common(30):
            print(f"  {c:6d}  {d}")

        exts = Counter(PurePosixPath(n).suffix.lower() for n in names)
        print("\nextensions:", dict(exts))

        meta = [n for n in names if PurePosixPath(n).suffix.lower() in {".csv", ".txt", ".json", ".md", ".xlsx"}]
        print(f"\nmetadata-like files ({len(meta)}):")
        for n in meta[:20]:
            print("  ", n, z.getinfo(n).file_size, "bytes")

        print("\nfirst 15 entries:")
        for n in names[:15]:
            print("  ", n)


if __name__ == "__main__":
    main(sys.argv[1])
