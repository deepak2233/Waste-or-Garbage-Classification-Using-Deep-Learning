"""Train-time augmentation as a Keras layer block.

Built from ``keras.layers`` rather than ``ImageDataGenerator``. The layers are
no-ops unless the model is called with ``training=True``, so augmentation cannot
leak into evaluation, and they run on the accelerator as part of the graph
instead of in a Python thread pool.
"""

from __future__ import annotations

import keras
from keras import layers

from wasteclf.config import AugmentConfig


def build_augmentation(cfg: AugmentConfig, seed: int = 42) -> keras.Sequential | None:
    """Return the augmentation block, or ``None`` when it is disabled.

    Every layer here is active only when the model is called with
    ``training=True``. ``model.predict`` and ``model.evaluate`` pass through
    untouched.
    """
    if not cfg.enabled:
        return None

    blocks: list[layers.Layer] = []

    flips = []
    if cfg.horizontal_flip:
        flips.append("horizontal")
    if cfg.vertical_flip:
        flips.append("vertical")
    if flips:
        blocks.append(layers.RandomFlip("_and_".join(flips), seed=seed, name="aug_flip"))

    if cfg.rotation > 0:
        blocks.append(
            layers.RandomRotation(cfg.rotation, fill_mode="reflect", seed=seed, name="aug_rotation")
        )
    if cfg.zoom > 0:
        blocks.append(layers.RandomZoom(cfg.zoom, fill_mode="reflect", seed=seed, name="aug_zoom"))
    if cfg.translation > 0:
        blocks.append(
            layers.RandomTranslation(
                cfg.translation,
                cfg.translation,
                fill_mode="reflect",
                seed=seed,
                name="aug_translate",
            )
        )
    if cfg.contrast > 0:
        blocks.append(layers.RandomContrast(cfg.contrast, seed=seed, name="aug_contrast"))
    if cfg.brightness > 0:
        blocks.append(
            layers.RandomBrightness(
                cfg.brightness, value_range=(0.0, 255.0), seed=seed, name="aug_brightness"
            )
        )

    if not blocks:
        return None
    return keras.Sequential(blocks, name="augmentation")
