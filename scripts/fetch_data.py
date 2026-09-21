#!/usr/bin/env python3
"""Download and verify the waste datasets.

Two datasets are supported.

``garbage12`` is Garbage Classification V2: 15,515 images across the twelve
categories this project targets. It lives on Kaggle, so downloading it needs
Kaggle credentials (``~/.kaggle/kaggle.json`` or the KAGGLE_USERNAME and
KAGGLE_KEY environment variables).

``trashnet`` is the original six-class set from Stanford CS229. It is in a
public GitHub repository, needs no credentials, and is a useful smaller set to
develop against.

    python scripts/fetch_data.py garbage12 --out data/raw
    python scripts/fetch_data.py trashnet  --out data/trashnet
    python scripts/fetch_data.py --check data/raw
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GARBAGE12_CLASSES = (
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
)
GARBAGE12_SLUG = "mostafaabla/garbage-classification"
#: Published size. Treated as a guide, not an assertion: Kaggle datasets get
#: reuploaded, and a hard equality check would fail for the wrong reason.
GARBAGE12_EXPECTED = 15_515

TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
TRASHNET_REPO = "https://github.com/garythung/trashnet"
TRASHNET_EXPECTED = 2_527

KAGGLE_HELP = f"""
Kaggle needs credentials before it will serve a dataset.

  1. Sign in at https://www.kaggle.com, open Settings, and choose
     "Create New Token". That downloads kaggle.json.
  2. Put it where the client looks:

         mkdir -p ~/.kaggle
         mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
         chmod 600 ~/.kaggle/kaggle.json

     or export the same values:

         export KAGGLE_USERNAME=...
         export KAGGLE_KEY=...

  3. Accept the dataset's terms once, on its page:
         https://www.kaggle.com/datasets/{GARBAGE12_SLUG}

If this machine has no outbound access to Kaggle, download the zip elsewhere
and unpack it so that <out>/<class>/*.jpg exists, then run:

    python scripts/fetch_data.py --check <out>
"""


def count_images(root: Path) -> dict[str, int]:
    """Images per class directory."""
    if not root.is_dir():
        return {}
    counts = {}
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            counts[entry.name] = sum(
                1 for p in entry.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
    return counts


def flatten_into(source: Path, destination: Path, classes: tuple[str, ...]) -> None:
    """Move class directories out of whatever nesting the archive used.

    Kaggle archives vary: sometimes the classes sit at the top level, sometimes
    under one or two wrapper directories. Rather than hardcode a depth, find the
    directory that actually contains the expected class names.
    """
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set(classes)

    candidates = [source, *(d for d in source.rglob("*") if d.is_dir())]
    root = next(
        (d for d in candidates if wanted & {c.name for c in d.iterdir() if c.is_dir()}),
        None,
    )
    if root is None:
        raise SystemExit(
            f"could not find the class directories under {source}. "
            f"Expected some of: {', '.join(sorted(wanted))}"
        )

    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        target = destination / entry.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(entry), str(target))


def fetch_garbage12(out: Path) -> int:
    """Pull Garbage Classification V2 from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install kagglehub", file=sys.stderr)
        return 1

    print(f"downloading {GARBAGE12_SLUG} (about 2 GB, this takes a while)...")
    try:
        cached = Path(kagglehub.dataset_download(GARBAGE12_SLUG))
    except Exception as exc:  # noqa: BLE001 - auth, network and licence all land here
        print(f"download failed: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        print(KAGGLE_HELP, file=sys.stderr)
        return 1

    print(f"unpacked at {cached}, arranging into {out}")
    flatten_into(cached, out, GARBAGE12_CLASSES)
    return check(out, GARBAGE12_CLASSES, GARBAGE12_EXPECTED)


def fetch_trashnet(out: Path) -> int:
    """Pull TrashNet from GitHub. No credentials needed."""
    work = out.parent / ".trashnet-checkout"
    if work.exists():
        shutil.rmtree(work)

    print(f"cloning {TRASHNET_REPO}...")
    done = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", TRASHNET_REPO, str(work)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        print(f"clone failed: {done.stderr.strip()}", file=sys.stderr)
        return 1

    archive = work / "data" / "dataset-resized.zip"
    if not archive.exists():
        print(f"{archive} is missing; the repository layout changed", file=sys.stderr)
        return 1

    print("extracting...")
    staging = work / "extracted"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)
    # The archive was built on macOS and carries a __MACOSX sidecar.
    shutil.rmtree(staging / "__MACOSX", ignore_errors=True)

    flatten_into(staging, out, TRASHNET_CLASSES)
    shutil.rmtree(work, ignore_errors=True)
    return check(out, TRASHNET_CLASSES, TRASHNET_EXPECTED)


def check(root: Path, classes: tuple[str, ...] | None = None, expected: int | None = None) -> int:
    """Report what is in a dataset directory and whether it looks complete."""
    counts = count_images(root)
    if not counts:
        print(f"{root} has no class subdirectories", file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"\n{root}\n")
    for name, count in counts.items():
        marker = " " if classes is None or name in classes else "?"
        print(f" {marker} {name:<14}{count:>7}")
    print(f"\n{total} images across {len(counts)} classes")

    status = 0
    if classes is not None:
        missing = [c for c in classes if c not in counts]
        if missing:
            print(f"\nmissing class(es): {', '.join(missing)}", file=sys.stderr)
            status = 1
        unexpected = [c for c in counts if c not in classes]
        if unexpected:
            print(f"unexpected directories (marked ?): {', '.join(unexpected)}", file=sys.stderr)
        empty = [c for c, n in counts.items() if n == 0]
        if empty:
            print(f"empty class(es): {', '.join(empty)}", file=sys.stderr)
            status = 1

    if expected is not None and total != expected:
        # A reupload changes the count; say so without treating it as failure.
        print(f"note: expected about {expected} images, found {total}", file=sys.stderr)

    if status == 0:
        print("\nlooks good. Next: wasteclf scan --data-root", root)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["garbage12", "trashnet"],
        help="which dataset to download",
    )
    parser.add_argument("--out", help="destination directory")
    parser.add_argument(
        "--check", metavar="DIR", help="validate a directory instead of downloading"
    )
    args = parser.parse_args()

    if args.check:
        root = Path(args.check)
        classes = GARBAGE12_CLASSES if len(count_images(root)) > 6 else None
        return check(root, classes)

    if args.dataset is None:
        parser.print_help()
        return 0

    if args.dataset == "garbage12":
        return fetch_garbage12(Path(args.out or "data/raw"))
    return fetch_trashnet(Path(args.out or "data/trashnet"))


if __name__ == "__main__":
    raise SystemExit(main())
