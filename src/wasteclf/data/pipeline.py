"""``tf.data`` input pipelines.

The pipeline emits raw float32 RGB in ``[0, 255]``. Backbone normalisation is a
layer inside the model (see :mod:`wasteclf.models.backbones`), so the same
tensor shape and range feeds training, evaluation and serving.

Ordering matters for correctness. ``shuffle`` goes before ``batch`` so batches
are reshuffled each epoch rather than merely reordered. ``cache`` goes before
``shuffle`` so decoding happens once. Augmentation is not in the pipeline at
all; it lives in the model and is inactive outside training.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from wasteclf.config import DataConfig
from wasteclf.data.manifest import DatasetManifest, Split
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


def _decode(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int], num_classes: int):
    """Read, decode and resize one image.

    ``expand_animations=False`` forces a static 3-D shape, which
    ``decode_image`` otherwise refuses to guarantee for GIFs. ``channels=3``
    converts greyscale and RGBA to RGB, so a stray PNG with an alpha channel does
    not produce a 4-channel tensor the backbone cannot consume.
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32)
    image.set_shape((*image_size, 3))
    return image, tf.one_hot(label, num_classes)


def make_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    training: bool = False,
    shuffle_buffer: int = 1000,
    cache: bool = True,
    seed: int = 42,
    skip_corrupt: bool = True,
) -> tf.data.Dataset:
    """Build one ``tf.data.Dataset`` of ``(image, one_hot_label)``.

    Args:
        paths: Absolute image paths.
        labels: Integer class indices, parallel to ``paths``.
        num_classes: Width of the one-hot label.
        image_size: Target ``(height, width)``.
        batch_size: Images per batch.
        training: Shuffle and repeat-free training order when ``True``.
            Evaluation splits keep file order so predictions line up with
            ``manifest.labels(split)``.
        shuffle_buffer: Shuffle buffer size. Capped at the dataset size.
        cache: Cache decoded images in RAM. Turn this off for datasets larger
            than memory.
        seed: Shuffle seed.
        skip_corrupt: Drop images that fail to decode instead of aborting the
            epoch.

    Returns:
        A batched, prefetched dataset.
    """
    if len(paths) != len(labels):
        raise ValueError(f"paths and labels differ in length: {len(paths)} vs {len(labels)}")
    if not paths:
        raise ValueError("cannot build a dataset from zero images")

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(
        lambda p, y: _decode(p, y, image_size, num_classes),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    if skip_corrupt:
        # ignore_errors() sets the dataset cardinality to UNKNOWN, so Keras can
        # no longer derive steps_per_epoch and warns on every epoch boundary.
        # build_manifest(verify_images=True) is the cheaper place to catch this.
        ds = ds.ignore_errors()
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(paths)), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_datasets(
    manifest: DatasetManifest, cfg: DataConfig, seed: int = 42
) -> dict[str, tf.data.Dataset]:
    """Build train, val and test datasets from a manifest.

    Only the training split is shuffled. Val and test keep manifest order, which
    is what lets :mod:`wasteclf.evaluation.metrics` pair predictions with labels
    without re-reading the files.
    """
    datasets: dict[str, tf.data.Dataset] = {}
    for split in Split:
        paths = manifest.paths(split)
        if not paths:
            logger.warning("split %s is empty; skipping", split.value)
            continue
        datasets[split.value] = make_dataset(
            paths=paths,
            labels=manifest.labels(split),
            num_classes=manifest.num_classes,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            training=(split is Split.TRAIN),
            shuffle_buffer=cfg.shuffle_buffer,
            cache=cfg.cache,
            seed=seed,
            skip_corrupt=cfg.skip_corrupt,
        )
        logger.info("%s dataset: %d images", split.value, len(paths))
    return datasets
