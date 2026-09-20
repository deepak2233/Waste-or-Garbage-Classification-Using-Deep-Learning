"""Backbone registry.

Each ImageNet backbone was trained with its own input normalisation, and using
the wrong one costs accuracy silently: the network still trains, it just starts
from features computed on out-of-distribution inputs.

VGG16 and ResNet50 want caffe-style inputs (BGR, ImageNet mean subtracted,
roughly ``[-124, 152]``). MobileNetV2 wants ``[-1, 1]``. EfficientNet wants raw
``[0, 255]`` and normalises internally. Scaling everything to ``[0, 1]`` and
hoping is a quiet way to lose several points of accuracy.

Each entry here carries its backbone's own ``preprocess_input``, and
:class:`BackbonePreprocessing` applies it inside the model, so switching
backbone switches the normalisation with it.

Adding a backbone means adding one entry to :data:`BACKBONES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import keras
from keras import layers

# Application modules are imported lazily inside the spec factories so that
# importing this module stays cheap and does not pull in every architecture.


@dataclass(frozen=True)
class BackboneSpec:
    """How to construct one backbone and normalise its inputs."""

    name: str
    #: ``(constructor, preprocess_input)`` pair from ``keras.applications``.
    loader: Callable[[], tuple[Callable[..., keras.Model], Callable]]
    default_image_size: tuple[int, int]
    #: Roughly how many trainable parameters the backbone carries, for the
    #: help text. Not used in any computation.
    params_millions: float
    notes: str

    def build(
        self,
        input_shape: tuple[int, int, int],
        weights: str | None = "imagenet",
        name: str | None = None,
    ) -> keras.Model:
        """Construct the backbone with its classifier head removed.

        ``name`` is passed through when the installed Keras supports it (3.x
        does, 2.15 does not), so the caller can look the backbone up by a stable
        name after the model is saved and reloaded.
        """
        ctor, _ = self.loader()
        kwargs = {"input_shape": input_shape, "include_top": False, "weights": weights}
        if name is not None:
            try:
                return ctor(name=name, **kwargs)
            except TypeError:
                pass  # older Keras: fall through and let get_backbone() find it structurally
        return ctor(**kwargs)

    def preprocess_fn(self) -> Callable:
        _, fn = self.loader()
        return fn


def _vgg16() -> tuple[Callable, Callable]:
    from keras.applications import vgg16

    return vgg16.VGG16, vgg16.preprocess_input


def _vgg19() -> tuple[Callable, Callable]:
    from keras.applications import vgg19

    return vgg19.VGG19, vgg19.preprocess_input


def _resnet50() -> tuple[Callable, Callable]:
    from keras.applications import resnet50

    return resnet50.ResNet50, resnet50.preprocess_input


def _resnet50v2() -> tuple[Callable, Callable]:
    from keras.applications import resnet_v2

    return resnet_v2.ResNet50V2, resnet_v2.preprocess_input


def _mobilenetv2() -> tuple[Callable, Callable]:
    from keras.applications import mobilenet_v2

    return mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input


def _efficientnetb0() -> tuple[Callable, Callable]:
    from keras.applications import efficientnet

    return efficientnet.EfficientNetB0, efficientnet.preprocess_input


def _efficientnetv2b0() -> tuple[Callable, Callable]:
    from keras.applications import efficientnet_v2

    return efficientnet_v2.EfficientNetV2B0, efficientnet_v2.preprocess_input


def _densenet121() -> tuple[Callable, Callable]:
    from keras.applications import densenet

    return densenet.DenseNet121, densenet.preprocess_input


BACKBONES: dict[str, BackboneSpec] = {
    "vgg16": BackboneSpec(
        name="vgg16",
        loader=_vgg16,
        default_image_size=(224, 224),
        params_millions=14.7,
        notes="Where this project started. Slow and heavy for the accuracy it gives.",
    ),
    "vgg19": BackboneSpec(
        name="vgg19",
        loader=_vgg19,
        default_image_size=(224, 224),
        params_millions=20.0,
        notes="Deeper VGG. Rarely worth the extra cost over vgg16 on this dataset size.",
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        loader=_resnet50,
        default_image_size=(224, 224),
        params_millions=23.6,
        notes="Residual baseline. Freeze BatchNorm when fine-tuning on a small dataset.",
    ),
    "resnet50v2": BackboneSpec(
        name="resnet50v2",
        loader=_resnet50v2,
        default_image_size=(224, 224),
        params_millions=23.6,
        notes="Pre-activation ResNet. Fine-tunes more stably than v1.",
    ),
    "mobilenetv2": BackboneSpec(
        name="mobilenetv2",
        loader=_mobilenetv2,
        default_image_size=(224, 224),
        params_millions=2.3,
        notes="Use this for the TFLite export path and anything running on a phone.",
    ),
    "efficientnetb0": BackboneSpec(
        name="efficientnetb0",
        loader=_efficientnetb0,
        default_image_size=(224, 224),
        params_millions=4.0,
        notes="Best accuracy per parameter here. Recommended default for new work.",
    ),
    "efficientnetv2b0": BackboneSpec(
        name="efficientnetv2b0",
        loader=_efficientnetv2b0,
        default_image_size=(224, 224),
        params_millions=5.9,
        notes="Trains faster than v1 and tolerates a higher fine-tuning learning rate.",
    ),
    "densenet121": BackboneSpec(
        name="densenet121",
        loader=_densenet121,
        default_image_size=(224, 224),
        params_millions=7.0,
        notes="Strong on texture-driven classes such as paper against cardboard.",
    ),
}


def available_backbones() -> list[str]:
    return sorted(BACKBONES)


def get_backbone_spec(name: str) -> BackboneSpec:
    """Look up a backbone, with a useful error when the name is wrong."""
    key = name.lower().replace("-", "").replace("_", "")
    if key in BACKBONES:
        return BACKBONES[key]
    raise KeyError(f"unknown backbone {name!r}. Available: {', '.join(available_backbones())}")


@keras.saving.register_keras_serializable(package="wasteclf")
class BackbonePreprocessing(layers.Layer):
    """Applies the backbone's own ``preprocess_input`` as a model layer.

    Keeping normalisation inside the model rather than in the input pipeline
    means an exported ``.keras`` or SavedModel artefact is self-contained: the
    serving code feeds raw 0-255 RGB and cannot apply the wrong normalisation,
    because there is no normalisation left for it to get wrong.

    Input is float32 RGB in ``[0, 255]``.
    """

    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self._fn = get_backbone_spec(backbone_name).preprocess_fn()

    def call(self, inputs):
        return self._fn(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config["backbone_name"] = self.backbone_name
        return config
