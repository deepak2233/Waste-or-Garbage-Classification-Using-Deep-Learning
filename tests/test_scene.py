"""Scene segmentation and the two-stage pipeline.

No TensorFlow: the segmenters are numpy, and the pipeline takes any object
satisfying the RegionClassifier protocol.
"""

from __future__ import annotations

import numpy as np
import pytest

from wasteclf.scene import ContentSegmenter, GridSegmenter, Region, ScenePipeline


class FakeClassifier:
    """Returns a fixed label per region, so the pipeline's own logic is what is tested."""

    def __init__(self, labels, confidences=None, image_size=(32, 32)):
        self.class_names = ["cardboard", "glass", "metal"]
        self.image_size = image_size
        self._labels = labels
        self._confidences = confidences or [0.9] * len(labels)
        self._cursor = 0
        self.seen_shapes = []

    def predict_array(self, images, paths=None):
        from wasteclf.inference.types import Prediction

        self.seen_shapes.append(images.shape)
        out = []
        for _ in range(len(images)):
            label = self._labels[self._cursor % len(self._labels)]
            confidence = self._confidences[self._cursor % len(self._confidences)]
            self._cursor += 1
            scores = dict.fromkeys(self.class_names, (1 - confidence) / 2)
            scores[label] = confidence
            out.append(Prediction("", label, confidence, scores))
        return out


@pytest.fixture
def scene():
    """A frame where the left half is flat background and the right half is noisy."""
    rng = np.random.default_rng(0)
    image = np.full((120, 160, 3), 100.0, dtype=np.float32)
    image[:, 80:] = rng.uniform(0, 255, (120, 80, 3))
    return image


# Segmenters ----------------------------------------------------------------


def test_grid_covers_the_frame_exactly():
    image = np.zeros((120, 160, 3), dtype=np.float32)
    regions = GridSegmenter(rows=4, cols=5).segment(image)

    assert len(regions) == 20
    assert sum(r.area for r in regions) == 120 * 160
    # No region may leave the frame.
    assert all(r.x >= 0 and r.y >= 0 for r in regions)
    assert all(r.x + r.width <= 160 and r.y + r.height <= 120 for r in regions)


def test_grid_overlap_stays_inside_the_frame():
    image = np.zeros((100, 100, 3), dtype=np.float32)
    regions = GridSegmenter(rows=4, cols=4, overlap=0.5).segment(image)

    assert all(r.x + r.width <= 100 and r.y + r.height <= 100 for r in regions)
    # Overlapping tiles cover more than the frame area.
    assert sum(r.area for r in regions) > 100 * 100


def test_grid_rejects_nonsense_geometry():
    with pytest.raises(ValueError, match="must be >= 1"):
        GridSegmenter(rows=0, cols=4)
    with pytest.raises(ValueError, match="overlap"):
        GridSegmenter(overlap=1.5)


def test_content_segmenter_drops_flat_background(scene):
    """The whole point: do not spend inference on empty ground."""
    everything = GridSegmenter(4, 4).segment(scene)
    kept = ContentSegmenter(4, 4, min_activity=0.1).segment(scene)

    assert len(kept) < len(everything)
    # Every kept tile must start in the noisy right half.
    assert all(r.x >= 60 for r in kept), [r.box for r in kept]


def test_activity_scores_noise_above_flat():
    flat = np.full((32, 32, 3), 120.0, dtype=np.float32)
    noisy = np.random.default_rng(0).uniform(0, 255, (32, 32, 3)).astype(np.float32)

    assert ContentSegmenter.activity(flat) == pytest.approx(0.0, abs=1e-6)
    assert ContentSegmenter.activity(noisy) > 0.3


def test_keep_top_caps_the_region_count(scene):
    kept = ContentSegmenter(6, 6, min_activity=0.0, keep_top=5).segment(scene)
    assert len(kept) == 5


def test_keep_top_falls_back_when_the_threshold_rejects_everything():
    """A uniformly flat frame must still yield something rather than nothing."""
    flat = np.full((64, 64, 3), 90.0, dtype=np.float32)
    kept = ContentSegmenter(4, 4, min_activity=0.9, keep_top=3).segment(flat)
    assert len(kept) == 3


def test_region_crop_matches_its_box():
    image = np.arange(100 * 100 * 3, dtype=np.float32).reshape(100, 100, 3)
    region = Region(10, 20, 30, 40)
    crop = region.crop(image)

    assert crop.shape == (40, 30, 3)
    assert np.array_equal(crop, image[20:60, 10:40])


# Pipeline ------------------------------------------------------------------


def test_pipeline_classifies_every_region(scene):
    classifier = FakeClassifier(["glass"])
    result = ScenePipeline(classifier, GridSegmenter(3, 4), min_confidence=0.0).analyse(scene)

    assert len(result.detections) == 12
    assert result.rejected == []
    assert result.counts["glass"] == 12


def test_pipeline_rejects_below_threshold(scene):
    classifier = FakeClassifier(["glass", "metal"], confidences=[0.9, 0.2])
    result = ScenePipeline(classifier, GridSegmenter(2, 2), min_confidence=0.5).analyse(scene)

    assert len(result.detections) == 2
    assert len(result.rejected) == 2
    assert all(label == "glass" for _, label, _ in result.detections)


def test_regions_are_resized_to_the_classifier_input(scene):
    classifier = FakeClassifier(["glass"], image_size=(24, 24))
    ScenePipeline(classifier, GridSegmenter(2, 2), min_confidence=0.0).analyse(scene)

    assert classifier.seen_shapes[0][1:3] == (24, 24)


def test_composition_is_area_weighted_not_count_weighted():
    """Two small tiles must not outvote one large one."""
    image = np.zeros((100, 100, 3), dtype=np.float32)

    class Stub:
        class_names = ["cardboard", "glass", "metal"]
        image_size = (16, 16)

        def predict_array(self, images, paths=None):
            from wasteclf.inference.types import Prediction

            return [Prediction("", "glass", 0.9, {"glass": 0.9}) for _ in images]

    pipeline = ScenePipeline(Stub(), GridSegmenter(2, 2), min_confidence=0.0)
    result = pipeline.analyse(image)
    assert result.composition["glass"] == pytest.approx(1.0)
    assert result.coverage == pytest.approx(1.0)


def test_empty_scene_result_is_safe():
    class NoRegions:
        def segment(self, image):
            return []

    classifier = FakeClassifier(["glass"])
    result = ScenePipeline(classifier, NoRegions()).analyse(np.zeros((10, 10, 3), np.float32))

    assert result.detections == []
    assert result.coverage == 0.0
    assert result.composition == dict.fromkeys(classifier.class_names, 0.0)
    assert "nothing above" in result.format_summary()


def test_pipeline_rejects_a_non_rgb_array():
    classifier = FakeClassifier(["glass"])
    with pytest.raises(ValueError, match="H, W, 3"):
        ScenePipeline(classifier).analyse(np.zeros((10, 10), np.float32))


def test_result_serialises(scene):
    classifier = FakeClassifier(["glass"])
    result = ScenePipeline(classifier, GridSegmenter(2, 2), min_confidence=0.0).analyse(scene)
    payload = result.to_dict()

    assert payload["regions_accepted"] == 4
    assert payload["composition"]["glass"] == 1.0
    assert len(payload["detections"]) == 4
    assert {"x", "y", "width", "height", "label", "confidence"} <= set(payload["detections"][0])
