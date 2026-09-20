"""Inference on a trained run.

A run directory carries the model, the label order and the config it was trained
with, so :class:`Predictor` reconstructs the input contract without being told
the image size or the class names. Getting the class order from the run rather
than from a fresh directory listing is what keeps index 3 meaning the same thing
at inference as it did during training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from wasteclf.config import Config

# Imported for its side effect: BackbonePreprocessing is decorated with
# @keras.saving.register_keras_serializable, and the decorator only runs when
# the module is imported. Without this, load_model() in a fresh process fails
# with "Could not locate class 'BackbonePreprocessing'".
from wasteclf.models.backbones import BackbonePreprocessing
from wasteclf.utils.logging import get_logger
from wasteclf.utils.run import RunDirectory

logger = get_logger(__name__)


@dataclass
class Prediction:
    """One image's result."""

    path: str
    label: str
    confidence: float
    #: All class probabilities, highest first.
    scores: dict[str, float]
    #: ``True`` when confidence fell below the caller's threshold. The caller
    #: decides what to do; nothing here silently rewrites the label.
    low_confidence: bool = False

    @property
    def runner_up(self) -> tuple[str, float]:
        items = list(self.scores.items())
        return items[1] if len(items) > 1 else items[0]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "low_confidence": self.low_confidence,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class Predictor:
    """Loads a run and classifies images."""

    def __init__(
        self,
        model: keras.Model,
        class_names: Sequence[str],
        image_size: tuple[int, int] = (224, 224),
        threshold: float = 0.0,
    ):
        self.model = model
        self.class_names = list(class_names)
        self.image_size = tuple(image_size)
        self.threshold = threshold

        output_units = int(model.output_shape[-1])
        if output_units != len(self.class_names):
            raise ValueError(
                f"model outputs {output_units} classes but labels.json lists "
                f"{len(self.class_names)}: {self.class_names}. The run directory is inconsistent."
            )

    @classmethod
    def from_run(cls, run_dir: str | Path, threshold: float = 0.0) -> Predictor:
        """Load a predictor from a run directory."""
        run = RunDirectory.open(run_dir)
        if not run.model_path.exists():
            raise FileNotFoundError(
                f"{run.model_path} not found. The run did not finish, or predates model.keras."
            )
        model = keras.models.load_model(
            run.model_path,
            # Registered via the import above; named here as well so the
            # dependency is explicit rather than an import that looks unused.
            custom_objects={"BackbonePreprocessing": BackbonePreprocessing},
        )
        class_names = run.read_labels()

        image_size = (224, 224)
        if run.config_path.exists():
            image_size = Config.load(run.config_path).data.image_size

        logger.info(
            "loaded %s: %d classes at %dx%d",
            run.model_path,
            len(class_names),
            image_size[0],
            image_size[1],
        )
        return cls(model, class_names, image_size, threshold)

    # Image loading ----------------------------------------------------------

    def load_image(self, path: str | Path) -> np.ndarray:
        """Read one image into the model's input format: float32 RGB 0-255."""
        raw = tf.io.read_file(str(path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size, method="bilinear")
        return tf.cast(image, tf.float32).numpy()

    def load_bytes(self, data: bytes) -> np.ndarray:
        """Read one image from raw bytes, for the HTTP serving path."""
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size, method="bilinear")
        return tf.cast(image, tf.float32).numpy()

    # Prediction -------------------------------------------------------------

    def predict_array(
        self, images: np.ndarray, paths: Sequence[str] | None = None
    ) -> list[Prediction]:
        """Classify a batch of already-loaded images, shape ``(n, H, W, 3)``."""
        if images.ndim == 3:
            images = images[None, ...]
        probs = np.asarray(self.model.predict(images, verbose=0), dtype=np.float64)
        names = paths or [""] * len(probs)
        return [self._to_prediction(p, str(n)) for p, n in zip(probs, names)]

    def predict(
        self, paths: str | Path | Iterable[str | Path], batch_size: int = 32
    ) -> list[Prediction]:
        """Classify one path, or many.

        Images are batched rather than sent one at a time, which on CPU is worth
        roughly an order of magnitude.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        items = [str(p) for p in paths]
        if not items:
            return []

        results: list[Prediction] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = np.stack([self.load_image(p) for p in chunk])
            results.extend(self.predict_array(batch, chunk))
        return results

    def predict_bytes(self, data: bytes, name: str = "upload") -> Prediction:
        """Classify one in-memory image."""
        return self.predict_array(self.load_bytes(data)[None, ...], [name])[0]

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
