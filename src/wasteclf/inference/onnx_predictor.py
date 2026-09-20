"""Inference through onnxruntime, without TensorFlow.

The full stack is about 1.2 GB installed, which does not fit in a serverless
function. onnxruntime, Pillow and NumPy come to roughly 180 MB, which does.

The trade-off is resizing. Training resized with ``tf.image.resize`` and
bilinear interpolation; here it is Pillow. The two do not agree exactly,
because Pillow's bilinear filter applies a support-scaled kernel when
downscaling and TensorFlow's does not. On real photographs the probabilities
differ in the third decimal and the predicted label is the same, but if you
need the two paths to agree exactly, serve the Keras model instead.

This module imports neither TensorFlow nor anything from
``wasteclf.models``. Keep it that way.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wasteclf.inference.types import Prediction

MODEL_FILENAME = "model.onnx"
LABELS_FILENAME = "labels.json"


class OnnxPredictor:
    """Classifies images using an exported ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Sequence[str],
        image_size: tuple[int, int] | None = None,
        threshold: float = 0.0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "OnnxPredictor needs onnxruntime: pip install 'wasteclf[onnx]'"
            ) from exc

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.class_names = list(class_names)
        self.threshold = threshold

        options = ort.SessionOptions()
        # One thread per core is counterproductive on a shared serverless vCPU,
        # and the spin-wait burns billed time while blocked.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        self.image_size = tuple(image_size) if image_size else self._size_from(spec.shape)

        outputs = self.session.get_outputs()[0].shape
        width = outputs[-1] if isinstance(outputs[-1], int) else None
        if width is not None and width != len(self.class_names):
            raise ValueError(
                f"model outputs {width} classes but labels list {len(self.class_names)}: "
                f"{self.class_names}"
            )

    @staticmethod
    def _size_from(shape: Sequence) -> tuple[int, int]:
        """Read (height, width) from an NHWC input spec, defaulting to 224."""
        if len(shape) == 4 and all(isinstance(d, int) for d in shape[1:3]):
            return int(shape[1]), int(shape[2])
        return (224, 224)

    @classmethod
    def from_dir(cls, model_dir: str | Path, threshold: float = 0.0) -> OnnxPredictor:
        """Load ``model.onnx`` and ``labels.json`` from a directory."""
        directory = Path(model_dir)
        labels_path = directory / LABELS_FILENAME
        if not labels_path.exists():
            raise FileNotFoundError(
                f"{labels_path} not found. Produce it with: "
                f"wasteclf export --run <run> --format onnx --out {directory}"
            )
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        return cls(
            model_path=directory / MODEL_FILENAME,
            class_names=payload["class_names"],
            image_size=tuple(payload["image_size"]) if payload.get("image_size") else None,
            threshold=threshold,
        )

    # Image handling ---------------------------------------------------------

    def load_bytes(self, data: bytes) -> np.ndarray:
        """Decode and resize one image to float32 RGB in ``[0, 255]``.

        Normalisation is inside the exported graph, so nothing is applied here.
        """
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            # convert() before resize: a palette or greyscale image resized in
            # its own mode then converted gives different pixels.
            rgb = img.convert("RGB")
            height, width = self.image_size
            resized = rgb.resize((width, height), Image.BILINEAR)
            return np.asarray(resized, dtype=np.float32)

    def load_image(self, path: str | Path) -> np.ndarray:
        return self.load_bytes(Path(path).read_bytes())

    # Prediction -------------------------------------------------------------

    def predict_array(
        self, images: np.ndarray, paths: Sequence[str] | None = None
    ) -> list[Prediction]:
        """Classify a batch of loaded images, shape ``(n, H, W, 3)``."""
        if images.ndim == 3:
            images = images[None, ...]
        batch = np.ascontiguousarray(images, dtype=np.float32)
        probs = self.session.run(None, {self.input_name: batch})[0]
        names = paths or [""] * len(probs)
        return [
            self._to_prediction(np.asarray(p, dtype=np.float64), str(n))
            for p, n in zip(probs, names)
        ]

    def predict_bytes(self, data: bytes, name: str = "upload") -> Prediction:
        return self.predict_array(self.load_bytes(data)[None, ...], [name])[0]

    def predict(self, paths, batch_size: int = 16) -> list[Prediction]:
        """Classify one path or many."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        items = [str(p) for p in paths]
        if not items:
            return []

        results: list[Prediction] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            results.extend(self.predict_array(np.stack([self.load_image(p) for p in chunk]), chunk))
        return results

    def _to_prediction(self, probabilities: np.ndarray, path: str) -> Prediction:
        order = np.argsort(-probabilities)
        scores = {self.class_names[i]: float(probabilities[i]) for i in order}
        top = int(order[0])
        confidence = float(probabilities[top])
        return Prediction(
            path=path,
            label=self.class_names[top],
            confidence=confidence,
            scores=scores,
            low_confidence=confidence < self.threshold,
        )
