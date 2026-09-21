"""End-to-end scene analysis: segment, recognise, aggregate.

Takes one photograph of a waste area and returns what is in it and in what
proportion, rather than a single label for the whole frame.

The classifier is closed-set: every region gets one of the trained classes,
whatever it actually contains. A tile of bare ground still comes back as
something. ``min_confidence`` is the defence against that, and it is why the
result separates accepted regions from rejected ones instead of quietly
dropping them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from wasteclf.scene.segmenters import GridSegmenter, Region, Segmenter
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


class RegionClassifier(Protocol):
    """The slice of the predictor interface this pipeline needs.

    Both :class:`wasteclf.inference.predictor.Predictor` and
    :class:`wasteclf.inference.onnx_predictor.OnnxPredictor` satisfy it.
    """

    class_names: list[str]
    image_size: tuple[int, int]

    def predict_array(self, images: np.ndarray, paths: Sequence[str] | None = None) -> list: ...


@dataclass
class SceneResult:
    """What one scene contained."""

    #: ``(region, label, confidence)`` for regions that cleared the threshold.
    detections: list[tuple[Region, str, float]] = field(default_factory=list)
    #: Regions that were classified but fell below the threshold.
    rejected: list[tuple[Region, str, float]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)

    @property
    def counts(self) -> dict[str, int]:
        """Accepted regions per class, highest first."""
        tally = dict.fromkeys(self.class_names, 0)
        for _, label, _ in self.detections:
            tally[label] += 1
        return dict(sorted(tally.items(), key=lambda kv: kv[1], reverse=True))

    @property
    def composition(self) -> dict[str, float]:
        """Share of accepted *area* per class.

        Area rather than region count, because a grid tile is a unit of
        sampling, not a unit of waste. Two tiles of cardboard and one of battery
        do not mean the pile is one third battery.
        """
        by_class: dict[str, int] = dict.fromkeys(self.class_names, 0)
        for region, label, _ in self.detections:
            by_class[label] += region.area
        total = sum(by_class.values())
        if total == 0:
            return dict.fromkeys(self.class_names, 0.0)
        shares = {name: area / total for name, area in by_class.items() if area}
        return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))

    @property
    def coverage(self) -> float:
        """Fraction of the frame covered by accepted regions.

        Exceeds 1.0 when the segmenter uses overlapping tiles.
        """
        height, width = self.image_size
        if not height or not width:
            return 0.0
        return sum(r.area for r, _, _ in self.detections) / float(height * width)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "regions_accepted": len(self.detections),
            "regions_rejected": len(self.rejected),
            "coverage": round(self.coverage, 4),
            "counts": {k: v for k, v in self.counts.items() if v},
            "composition": {k: round(v, 4) for k, v in self.composition.items()},
            "detections": [
                {**region.to_dict(), "label": label, "confidence": round(confidence, 4)}
                for region, label, confidence in self.detections
            ],
        }

    def format_summary(self) -> str:
        """A short report for the terminal."""
        lines = [
            f"{len(self.detections)} region(s) accepted, "
            f"{len(self.rejected)} below threshold, "
            f"coverage {self.coverage:.0%}",
            "",
            f"{'class':<16}{'regions':>9}{'area share':>13}",
            "-" * 38,
        ]
        counts = self.counts
        # composition always lists every class, so test for a non-zero share
        # rather than for an empty mapping.
        present = {name: share for name, share in self.composition.items() if share > 0}
        for name, share in present.items():
            lines.append(f"{name:<16}{counts.get(name, 0):>9}{share:>12.1%}")
        if not present:
            lines.append("(nothing above the confidence threshold)")
        return "\n".join(lines)


class ScenePipeline:
    """Segment a scene, classify each region, aggregate the result."""

    def __init__(
        self,
        classifier: RegionClassifier,
        segmenter: Segmenter | None = None,
        min_confidence: float = 0.5,
        batch_size: int = 32,
    ):
        """
        Args:
            classifier: A loaded predictor.
            segmenter: Region proposer. Defaults to a 6x8 grid.
            min_confidence: Regions below this are recorded as rejected rather
                than counted. On a closed-set classifier this is the only thing
                standing between "empty ground" and a confident wrong label.
            batch_size: Regions classified per forward pass.
        """
        self.classifier = classifier
        self.segmenter = segmenter or GridSegmenter()
        self.min_confidence = min_confidence
        self.batch_size = batch_size

    def _resize(self, tile: np.ndarray) -> np.ndarray:
        """Resize one crop to the classifier's input size, with Pillow."""
        from PIL import Image

        height, width = self.classifier.image_size
        image = Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8), mode="RGB")
        return np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)

    def analyse(self, image: np.ndarray) -> SceneResult:
        """Run the full pipeline over one ``(H, W, 3)`` RGB array."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape}")

        regions = self.segmenter.segment(image)
        result = SceneResult(
            class_names=list(self.classifier.class_names),
            image_size=(image.shape[0], image.shape[1]),
        )
        if not regions:
            logger.warning("segmenter proposed no regions")
            return result

        logger.info("classifying %d region(s)", len(regions))
        for start in range(0, len(regions), self.batch_size):
            chunk = regions[start : start + self.batch_size]
            batch = np.stack([self._resize(r.crop(image)) for r in chunk])
            for region, prediction in zip(chunk, self.classifier.predict_array(batch)):
                entry = (region, prediction.label, prediction.confidence)
                if prediction.confidence >= self.min_confidence:
                    result.detections.append(entry)
                else:
                    result.rejected.append(entry)

        return result

    def analyse_file(self, path: str | Path) -> SceneResult:
        """Load an image from disk and analyse it."""
        from PIL import Image

        with Image.open(path) as img:
            return self.analyse(np.asarray(img.convert("RGB"), dtype=np.float32))
