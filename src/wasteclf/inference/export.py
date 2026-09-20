"""Deployment exports."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from wasteclf.models.build import inference_model
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


def export_onnx(
    model: keras.Model,
    path: str | Path,
    image_size: tuple[int, int] | None = None,
) -> Path:
    """Convert to ONNX, for serving without TensorFlow.

    onnxruntime plus Pillow is roughly 180 MB installed against 1.2 GB for the
    TensorFlow stack, which is the difference between fitting in a serverless
    function and not.

    The augmentation block is stripped first (see
    :func:`wasteclf.models.build.inference_model`). Leaving it in produces a
    graph containing random-sampling ops that ONNX has no operator for; the
    converter drops them and emits a file onnxruntime rejects as invalid.

    Requires the ``onnx`` extra. Note that ``onnx`` must be pinned below 1.18:
    newer releases need protobuf >= 6, while TensorFlow 2.17 pins protobuf < 5,
    and the two cannot share an environment.

    Args:
        model: The trained model.
        path: Destination ``.onnx`` file.
        image_size: Input ``(height, width)``. Read from the model when omitted.

    Returns:
        The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    slim = inference_model(model)
    height, width = image_size or slim.input_shape[1:3]

    # Keras refuses to export a model it has never called.
    slim(np.zeros((1, height, width, 3), dtype="float32"))

    try:
        slim.export(str(target), format="onnx", verbose=False)
    except ImportError as exc:
        raise ImportError("ONNX export needs the onnx extra: pip install 'wasteclf[onnx]'") from exc

    logger.info("ONNX model written to %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target
