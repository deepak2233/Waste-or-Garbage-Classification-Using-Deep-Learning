"""Input pipeline behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from wasteclf.config import DataConfig
from wasteclf.data.manifest import Split, build_manifest
from wasteclf.data.pipeline import build_datasets, make_dataset

pytestmark = pytest.mark.needs_tf


def test_batches_have_the_right_shape_and_range(synthetic_root):
    manifest = build_manifest(synthetic_root)
    ds = make_dataset(
        manifest.paths(Split.TRAIN),
        manifest.labels(Split.TRAIN),
        manifest.num_classes,
        image_size=(48, 48),
        batch_size=4,
    )
    images, labels = next(iter(ds))
    assert images.shape[1:] == (48, 48, 3)
    assert labels.shape[1] == manifest.num_classes
    # The pipeline emits raw 0-255; normalisation happens inside the model.
    assert float(np.min(images)) >= 0.0 and float(np.max(images)) <= 255.0
    assert float(np.max(images)) > 1.5, "images look pre-scaled to [0,1]"


def test_labels_are_one_hot(synthetic_root):
    manifest = build_manifest(synthetic_root)
    ds = make_dataset(
        manifest.paths(Split.VAL), manifest.labels(Split.VAL), manifest.num_classes, batch_size=4
    )
    _, labels = next(iter(ds))
    assert np.allclose(np.sum(np.asarray(labels), axis=1), 1.0)


def test_evaluation_order_is_deterministic(synthetic_root):
    """Evaluation splits must keep manifest order so predictions pair with labels."""
    manifest = build_manifest(synthetic_root)
    args = (manifest.paths(Split.TEST), manifest.labels(Split.TEST), manifest.num_classes)
    first = np.concatenate([np.argmax(y, axis=1) for _, y in make_dataset(*args, batch_size=4)])
    second = np.concatenate([np.argmax(y, axis=1) for _, y in make_dataset(*args, batch_size=4)])
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.array(manifest.labels(Split.TEST)))


def test_training_split_reshuffles_between_epochs(synthetic_root):
    manifest = build_manifest(synthetic_root)
    ds = make_dataset(
        manifest.paths(Split.TRAIN),
        manifest.labels(Split.TRAIN),
        manifest.num_classes,
        batch_size=4,
        training=True,
        cache=False,
    )
    first = np.concatenate([np.argmax(y, axis=1) for _, y in ds])
    second = np.concatenate([np.argmax(y, axis=1) for _, y in ds])
    assert not np.array_equal(first, second), "training order repeated across epochs"


def test_build_datasets_produces_all_three_splits(synthetic_root):
    manifest = build_manifest(synthetic_root)
    datasets = build_datasets(manifest, DataConfig(image_size=(32, 32), batch_size=4))
    assert set(datasets) == {"train", "val", "test"}


def test_greyscale_and_rgba_are_converted_to_three_channels(tmp_path):
    """cv2.imread(path, -1) kept the alpha channel; decode_image(channels=3) does not."""
    from PIL import Image

    root = tmp_path / "mixed"
    (root / "a").mkdir(parents=True)
    Image.fromarray(np.full((16, 16), 128, np.uint8), mode="L").save(root / "a" / "grey.png")
    Image.fromarray(np.full((16, 16, 4), 200, np.uint8), mode="RGBA").save(root / "a" / "alpha.png")

    manifest = build_manifest(root)
    ds = make_dataset(
        manifest.paths(Split.TRAIN) + manifest.paths(Split.VAL) + manifest.paths(Split.TEST),
        [0, 0],
        1,
        image_size=(16, 16),
        batch_size=2,
    )
    images, _ = next(iter(ds))
    assert images.shape[-1] == 3


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="differ in length"):
        make_dataset(["a.jpg", "b.jpg"], [0], 2)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="zero images"):
        make_dataset([], [], 2)
