#!/usr/bin/env python3
"""Generate a small synthetic dataset shaped like the real one.

Lets CI, the test suite and a first-time contributor exercise the whole pipeline
without the 2,187-image download. Each class gets a distinct colour and texture,
so a model can actually learn it and a smoke test that reports 1/7 accuracy is
signalling a real bug rather than the data being noise.

    python scripts/make_synthetic_data.py --out data/synthetic --per-class 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Deliberately imbalanced, in the same direction as the real dataset: compost
# and trash are the small classes.
CLASS_SPEC: dict[str, tuple[tuple[int, int, int], float]] = {
    "cardboard": ((181, 136, 99), 1.0),
    "compost": ((86, 125, 70), 0.35),
    "glass": ((132, 186, 201), 1.0),
    "metal": ((168, 169, 173), 0.9),
    "paper": ((238, 236, 225), 1.0),
    "plastic": ((214, 93, 86), 0.9),
    "trash": ((110, 100, 96), 0.4),
}


def make_image(rng: np.random.Generator, colour: tuple[int, int, int], size: int) -> Image.Image:
    """A colour field with class-specific texture and per-image jitter."""
    base = np.array(colour, dtype=np.float64)
    jitter = rng.normal(0, 12, size=3)
    canvas = np.ones((size, size, 3)) * np.clip(base + jitter, 0, 255)

    # Horizontal banding, with a frequency that varies per image.
    freq = rng.uniform(2, 9)
    rows = np.sin(np.linspace(0, freq * np.pi, size))[:, None, None]
    canvas += rows * rng.uniform(8, 22)

    # A few blobs so the texture is not purely periodic.
    for _ in range(rng.integers(2, 6)):
        cy, cx = rng.integers(0, size, size=2)
        radius = rng.integers(size // 10, size // 4)
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        canvas[mask] += rng.normal(0, 18, size=3)

    canvas += rng.normal(0, 6, size=canvas.shape)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="data/synthetic", help="output directory")
    parser.add_argument("--per-class", type=int, default=24, help="images for the largest class")
    parser.add_argument("--size", type=int, default=64, help="image side in pixels")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    total = 0

    for name, (colour, share) in CLASS_SPEC.items():
        count = max(3, int(round(args.per_class * share)))
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            make_image(rng, colour, args.size).save(folder / f"{name}_{i:03d}.jpg", quality=92)
        total += count
        print(f"{name:<12} {count:>4} images")

    print(f"\n{total} images in {len(CLASS_SPEC)} classes under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
