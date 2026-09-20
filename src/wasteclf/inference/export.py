"""Deployment exports."""

from __future__ import annotations

from pathlib import Path

import keras
import tensorflow as tf

from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


def export_savedmodel(model: keras.Model, path: str | Path) -> Path:
    """Write a TensorFlow SavedModel for TF Serving.

    Preprocessing is inside the graph, so a serving client posts raw 0-255 RGB
    and cannot apply the wrong normalisation.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(target))
    logger.info("SavedModel written to %s", target)
    return target


def export_tflite(
    model: keras.Model,
    path: str | Path,
    quantize: bool = True,
    representative_images=None,
) -> Path:
    """Convert to TFLite for phone and edge deployment.

    Args:
        model: The trained model.
        path: Destination ``.tflite`` file.
        quantize: Apply default (dynamic-range) float16/int8 optimisation,
            roughly a 4x size reduction for a small accuracy cost.
        representative_images: Iterable of float32 batches. Supplying this
            switches on full integer quantisation, which is what a
            microcontroller target needs. Without it, weights are quantised but
            activations stay float.

    Returns:
        The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_images is not None:

            def representative_dataset():
                for batch in representative_images:
                    yield [tf.cast(batch, tf.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Grad-CAM-free inference graphs still use a handful of ops TFLite maps to
    # its TF kernel fallback; allow them rather than failing the export.
    converter.target_spec.supported_ops = list(
        getattr(converter.target_spec, "supported_ops", [])
    ) or [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    blob = converter.convert()
    target.write_bytes(blob)
    logger.info("TFLite model written to %s (%.1f KB)", target, len(blob) / 1024)
    return target
