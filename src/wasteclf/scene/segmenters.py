"""Region proposals over a waste scene.

Two strategies, both of which need no training data.

``GridSegmenter`` tiles the frame. It is the literal reading of the
architecture diagram, it is exhaustive, and it is the right choice when waste
is spread roughly evenly across the frame.

``ContentSegmenter`` keeps grid tiles but drops the ones that are almost
certainly background. On a photograph of a dump most tiles are ground, sky or
tarmac; classifying them wastes inference and pollutes the composition estimate
with confident nonsense, because a 12-way softmax has no way to answer "nothing
here".

Learned proposals (SAM, YOLO-seg) fit the same :class:`Segmenter` protocol and
can be dropped in without the pipeline changing. They are not implemented here:
both need weights this project does not ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Region:
    """A rectangular area of the scene, in pixels."""

    x: int
    y: int
    width: int
    height: int
    #: How likely this region is to hold an object at all. Grid tiling has no
    #: opinion and returns 1.0; content filtering returns its activity measure.
    score: float = 1.0
    source: str = "grid"

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Cut this region out of an ``(H, W, C)`` image."""
        return image[self.y : self.y + self.height, self.x : self.x + self.width]

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": round(self.score, 4),
            "source": self.source,
        }


@runtime_checkable
class Segmenter(Protocol):
    """Anything that proposes regions for a scene."""

    def segment(self, image: np.ndarray) -> list[Region]: ...


@dataclass
class GridSegmenter:
    """Split the frame into a regular grid.

    Args:
        rows: Tiles down the frame.
        cols: Tiles across the frame.
        overlap: Fraction of a tile that neighbouring tiles share. Some overlap
            stops an object that straddles a tile boundary from being cut in
            half and missed twice.
    """

    rows: int = 6
    cols: int = 8
    overlap: float = 0.0

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"rows and cols must be >= 1, got {self.rows}x{self.cols}")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")

    def segment(self, image: np.ndarray) -> list[Region]:
        height, width = image.shape[:2]
        tile_h = height / self.rows
        tile_w = width / self.cols
        pad_h = int(tile_h * self.overlap / 2)
        pad_w = int(tile_w * self.overlap / 2)

        regions: list[Region] = []
        for r in range(self.rows):
            for c in range(self.cols):
                # Clamp to the frame so overlap never produces an out-of-bounds
                # crop, which numpy would silently return short rather than error.
                y0 = max(0, int(r * tile_h) - pad_h)
                x0 = max(0, int(c * tile_w) - pad_w)
                y1 = min(height, int((r + 1) * tile_h) + pad_h)
                x1 = min(width, int((c + 1) * tile_w) + pad_w)
                if y1 > y0 and x1 > x0:
                    regions.append(Region(x0, y0, x1 - x0, y1 - y0, 1.0, "grid"))
        return regions


@dataclass
class ContentSegmenter:
    """Grid tiling, minus the tiles that look like empty background.

    "Activity" is the mean per-channel standard deviation inside the tile,
    normalised to ``[0, 1]``. Waste is cluttered and high-variance; tarmac, soil
    and sky are not. This is a heuristic, not a detector, and it will discard a
    large flat object such as a single sheet of cardboard filling a tile.

    Args:
        rows: Tiles down the frame.
        cols: Tiles across the frame.
        overlap: Shared fraction between neighbouring tiles.
        min_activity: Tiles scoring below this are dropped. 0.0 keeps everything.
        keep_top: If set, keep only this many highest-scoring tiles regardless
            of the threshold. Useful for bounding inference cost per frame.
    """

    rows: int = 6
    cols: int = 8
    overlap: float = 0.0
    min_activity: float = 0.06
    keep_top: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_activity <= 1.0:
            raise ValueError(f"min_activity must be in [0, 1], got {self.min_activity}")
        if self.keep_top is not None and self.keep_top < 1:
            raise ValueError(f"keep_top must be >= 1 or None, got {self.keep_top}")
        self._grid = GridSegmenter(self.rows, self.cols, self.overlap)

    @staticmethod
    def activity(tile: np.ndarray) -> float:
        """Mean channel standard deviation, scaled to roughly [0, 1]."""
        if tile.size == 0:
            return 0.0
        values = tile.astype(np.float64)
        if values.max() <= 1.0:  # already normalised
            values = values * 255.0
        # 128 is the largest std a 0-255 channel can reach (half black, half white).
        return float(np.clip(values.std(axis=(0, 1)).mean() / 128.0, 0.0, 1.0))

    def segment(self, image: np.ndarray) -> list[Region]:
        scored = [
            Region(r.x, r.y, r.width, r.height, self.activity(r.crop(image)), "content")
            for r in self._grid.segment(image)
        ]
        kept = [r for r in scored if r.score >= self.min_activity]

        if self.keep_top is not None:
            # Fall back to the best tiles when the threshold rejected everything,
            # so a uniformly low-contrast frame still yields something to classify.
            pool = kept or scored
            kept = sorted(pool, key=lambda r: r.score, reverse=True)[: self.keep_top]

        return sorted(kept, key=lambda r: (r.y, r.x))
