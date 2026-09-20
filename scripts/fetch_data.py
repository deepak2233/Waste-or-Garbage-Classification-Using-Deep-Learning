#!/usr/bin/env python3
"""Help set up the dataset directory.

The images live in Google Drive rather than in this repository, and Drive
folder links cannot be fetched reliably without credentials. This script prints
the options and then validates whatever you put in place.

    python scripts/fetch_data.py --check data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DRIVE_URL = "https://drive.google.com/drive/folders/1IHlDN1oXhsnb9xAwUOQJrfJ3s83mkDeg"

EXPECTED_CLASSES = ("cardboard", "compost", "glass", "metal", "paper", "plastic", "trash")

INSTRUCTIONS = f"""
The dataset is not in this repository. Three ways to get it:

1. The original Google Drive folder
     {DRIVE_URL}
   Download it and unpack so that data/raw/<class>/*.jpg exists.

2. gdown, if you prefer the command line
     pip install gdown
     gdown --folder {DRIVE_URL} -O data/raw

3. TrashNet, the public dataset this taxonomy is based on
     https://github.com/garythung/trashnet
   It covers six of the seven classes; it has no 'compost' folder.

Expected layout:

    data/raw/
    ├── cardboard/
    ├── compost/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/

Then check it:

    python scripts/fetch_data.py --check data/raw
    wasteclf scan --data-root data/raw
"""


def check(root: Path) -> int:
    """Validate a dataset directory and report what is there."""
    if not root.is_dir():
        print(f"{root} does not exist.\n{INSTRUCTIONS}", file=sys.stderr)
        return 1

    found = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not found:
        print(f"{root} has no class subdirectories.\n{INSTRUCTIONS}", file=sys.stderr)
        return 1

    print(f"{root}\n")
    total = 0
    for name in found:
        count = sum(
            1
            for p in (root / name).rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        total += count
        marker = " " if name in EXPECTED_CLASSES else "?"
        print(f" {marker} {name:<14}{count:>6} images")

    print(f"\n{total} images across {len(found)} classes")

    missing = [c for c in EXPECTED_CLASSES if c not in found]
    if missing:
        print(f"\nmissing the expected class(es): {', '.join(missing)}", file=sys.stderr)
    extra = [c for c in found if c not in EXPECTED_CLASSES]
    if extra:
        print(f"unexpected directories (marked ?): {', '.join(extra)}", file=sys.stderr)

    if total == 0:
        print("\nno images found", file=sys.stderr)
        return 1
    if total < 100:
        print(f"\nonly {total} images; training will overfit badly", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", metavar="DIR", help="validate a dataset directory")
    args = parser.parse_args()

    if args.check:
        return check(Path(args.check))
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
