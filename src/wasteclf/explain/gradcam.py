"""Grad-CAM.

Answers the question a confusion matrix cannot: when the model calls a picture
"metal", is it looking at the can or at the background? On a small scraped
dataset the honest answer is often the background, and Grad-CAM is the cheapest
way to find that out before anyone deploys the thing.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017 (arXiv:1610.02391).
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from wasteclf.models.build import get_backbone
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Class-activation heatmaps for a model built by :func:`build_model`."""

    def __init__(self, model: keras.Model, layer_name: str | None = None):
        """
        Args:
            model: The trained classifier.
            layer_name: Convolutional layer to explain. Defaults to the last
                4-D output inside the backbone, which is the standard choice:
                deep enough to be semantic, still spatial enough to localise.
        """
        self.model = model
        self.backbone = get_backbone(model)
        self.layer_name = layer_name or self._last_conv_layer()
        self._grad_model = self._build_grad_model()

    def _last_conv_layer(self) -> str:
        for layer in reversed(self.backbone.layers):
            shape = getattr(layer, "output_shape", None)
            if shape is None:
                shape = getattr(getattr(layer, "output", None), "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
        raise ValueError(f"no 4-D convolutional output found in backbone {self.backbone.name!r}")

    def _build_grad_model(self) -> keras.Model:
        """Wire a model that emits both the target feature map and the logits.

        The backbone is a nested model, so a single functional graph cannot tap
        an inner layer directly. This builds a sub-model over the backbone and
        replays the classifier head on top of it.
        """
        target = self.backbone.get_layer(self.layer_name)
        feature_extractor = keras.Model(
            self.backbone.inputs, [target.output, self.backbone.output], name="gradcam_features"
        )

        # Layers applied after the backbone, in graph order.
        backbone_index = next(
            i for i, layer in enumerate(self.model.layers) if layer is self.backbone
        )
        head_layers = self.model.layers[backbone_index + 1 :]
        pre_layers = self.model.layers[1:backbone_index]  # augmentation, preprocessing

        inputs = keras.Input(shape=self.model.input_shape[1:], name="gradcam_input")
        x = inputs
        for layer in pre_layers:
            # training=False keeps augmentation inert, so the heatmap explains
            # the image that was passed in rather than a random crop of it.
            x = layer(x, training=False)
        feature_map, pooled = feature_extractor(x)
        y = pooled
        for layer in head_layers:
            y = layer(y)
        return keras.Model(inputs, [feature_map, y], name="gradcam")

    def heatmap(
        self, image: np.ndarray, class_index: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """Compute a heatmap for one image.

        Args:
            image: Raw RGB in ``[0, 255]``, shape ``(H, W, 3)`` or ``(1, H, W, 3)``.
            class_index: Class to explain. Defaults to the predicted class.

        Returns:
            ``(heatmap, class_index, score)`` where ``heatmap`` is ``(h, w)`` in
            ``[0, 1]`` at the feature map's resolution.
        """
        batch = image[None, ...] if image.ndim == 3 else image
        batch = tf.convert_to_tensor(batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            feature_map, predictions = self._grad_model(batch, training=False)
            tape.watch(feature_map)
            index = int(tf.argmax(predictions[0])) if class_index is None else int(class_index)
            score = predictions[:, index]

        grads = tape.gradient(score, feature_map)
        if grads is None:
            raise RuntimeError(
                f"no gradient flows from the output to layer {self.layer_name!r}; "
                "pick a layer inside the backbone"
            )

        # Channel importance = spatially averaged gradient.
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(feature_map[0] * weights, axis=-1)
        cam = tf.nn.relu(cam)  # only evidence *for* the class

        peak = tf.reduce_max(cam)
        cam = cam / peak if peak > 0 else cam
        return cam.numpy(), index, float(score[0])

    def explain(
        self, image: np.ndarray, class_index: int | None = None, alpha: float = 0.45
    ) -> tuple[np.ndarray, int, float]:
        """Return the heatmap already overlaid on the image."""
        cam, index, score = self.heatmap(image, class_index)
        raw = image[0] if image.ndim == 4 else image
        return overlay_heatmap(raw, cam, alpha=alpha), index, score


def overlay_heatmap(
    image: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: str = "jet"
) -> np.ndarray:
    """Blend a heatmap over an image.

    Args:
        image: RGB in ``[0, 255]``, shape ``(H, W, 3)``.
        cam: Heatmap in ``[0, 1]``, any resolution; resized to match.
        alpha: Heatmap opacity.
        colormap: Any matplotlib colormap name.

    Returns:
        ``uint8`` RGB, shape ``(H, W, 3)``.
    """
    import matplotlib

    height, width = image.shape[:2]
    resized = tf.image.resize(cam[..., None], (height, width), method="bilinear").numpy()[..., 0]

    colours = matplotlib.colormaps[colormap](resized)[..., :3] * 255.0
    blended = np.clip((1 - alpha) * image.astype(np.float64) + alpha * colours, 0, 255)
    return blended.astype(np.uint8)


def save_explanation(overlaid: np.ndarray, path: str | Path, title: str | None = None) -> Path:
    """Write an overlay to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlaid)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target
