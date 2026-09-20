"""Backbone registry and model assembly."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from wasteclf.config import AugmentConfig, ModelConfig
from wasteclf.models.backbones import (
    BACKBONES,
    BackbonePreprocessing,
    available_backbones,
    get_backbone_spec,
)
from wasteclf.models.build import build_model, get_backbone, set_finetune_trainable

pytestmark = pytest.mark.needs_tf


def test_registry_is_not_empty():
    assert "vgg16" in available_backbones()
    assert "efficientnetb0" in available_backbones()


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_every_registered_backbone_resolves(name):
    """A registry entry that cannot be imported is worse than no entry."""
    spec = get_backbone_spec(name)
    ctor, preprocess = spec.loader()
    assert callable(ctor) and callable(preprocess)


def test_lookup_is_forgiving_about_separators():
    assert get_backbone_spec("ResNet-50").name == "resnet50"
    assert get_backbone_spec("mobilenet_v2").name == "mobilenetv2"


def test_unknown_backbone_lists_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        get_backbone_spec("inception_v99")


def test_vgg16_preprocessing_is_caffe_style_not_zero_to_one():
    """VGG16 wants mean-subtracted BGR, not [0,1]. Easy thing to get wrong."""
    layer = BackbonePreprocessing("vgg16")
    out = np.asarray(layer(np.full((1, 8, 8, 3), 255.0, dtype=np.float32)))
    # White maps to 255 minus the ImageNet channel means, nowhere near 1.0.
    assert out.max() > 100
    assert np.allclose(
        np.sort(out[0, 0, 0]), sorted([255 - 103.939, 255 - 116.779, 255 - 123.68]), atol=1e-3
    )


def test_preprocessing_layer_survives_serialisation(tmp_path):
    model = keras.Sequential([keras.Input((8, 8, 3)), BackbonePreprocessing("resnet50")])
    path = tmp_path / "m.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    probe = np.random.default_rng(0).uniform(0, 255, (1, 8, 8, 3)).astype(np.float32)
    assert np.allclose(np.asarray(model(probe)), np.asarray(reloaded(probe)))


def test_build_model_has_the_expected_structure():
    model = build_model(
        ModelConfig(backbone="mobilenetv2", weights=None), 7, (64, 64), AugmentConfig()
    )
    names = [layer.name for layer in model.layers]
    assert names[0] == "image"
    # Augmentation precedes normalisation so fill values stay in pixel space.
    assert names.index("augmentation") < names.index("preprocess")
    assert names.index("preprocess") < names.index("backbone")
    assert model.output_shape == (None, 7)


def test_backbone_starts_frozen():
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    assert get_backbone(model).trainable is False


def test_augmentation_can_be_disabled():
    model = build_model(
        ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64), AugmentConfig(enabled=False)
    )
    assert "augmentation" not in [layer.name for layer in model.layers]


def test_augmentation_is_inert_at_inference():
    """Two inference passes must agree; otherwise evaluation is non-deterministic."""
    model = build_model(
        ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64), AugmentConfig()
    )
    probe = np.random.default_rng(0).uniform(0, 255, (2, 64, 64, 3)).astype(np.float32)
    first = np.asarray(model(probe, training=False))
    second = np.asarray(model(probe, training=False))
    assert np.allclose(first, second)


def test_unfreeze_counts_from_the_top():
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    unfrozen = set_finetune_trainable(model, unfreeze_layers=10, freeze_batchnorm=True)
    backbone = get_backbone(model)
    assert 0 < unfrozen <= 10
    # Everything below the boundary stays frozen.
    assert backbone.layers[0].trainable is False


def test_batchnorm_stays_frozen_when_asked():
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    set_finetune_trainable(model, unfreeze_layers=-1, freeze_batchnorm=True)
    bn = [
        layer
        for layer in get_backbone(model).layers
        if isinstance(layer, keras.layers.BatchNormalization)
    ]
    assert bn, "mobilenetv2 should contain BatchNormalization layers"
    assert all(layer.trainable is False for layer in bn)


def test_batchnorm_unfreezes_when_not_asked():
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    set_finetune_trainable(model, unfreeze_layers=-1, freeze_batchnorm=False)
    bn = [
        layer
        for layer in get_backbone(model).layers
        if isinstance(layer, keras.layers.BatchNormalization)
    ]
    assert all(layer.trainable is True for layer in bn)


def test_unfreeze_zero_refreezes():
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    set_finetune_trainable(model, unfreeze_layers=20)
    assert set_finetune_trainable(model, unfreeze_layers=0) == 0
    assert get_backbone(model).trainable is False


def test_flatten_pooling_inflates_the_head():
    """Why avg is the default pooling.

    Flatten feeds every spatial position into the head. On MobileNetV2 at
    64x64 that is a 2x2x1280 map, so the head takes 5,120 inputs against 1,280
    for average pooling. On VGG16 at 224x224 it is 25,088.
    """
    avg = build_model(ModelConfig(backbone="mobilenetv2", weights=None, pooling="avg"), 7, (64, 64))
    flat = build_model(
        ModelConfig(backbone="mobilenetv2", weights=None, pooling="flatten"), 7, (64, 64)
    )

    avg_head = avg.get_layer("head_dense").count_params()
    flat_head = flat.get_layer("head_dense").count_params()
    assert flat_head > avg_head * 3


def test_frozen_random_backbone_emits_constant_features():
    """Why `weights=None` plus a frozen backbone cannot learn.

    An untrained backbone held in inference mode carries BatchNorm statistics
    that were never fitted, which on MobileNetV2 drives every ReLU6 to its
    floor. The pooled features come out identical for every image, so the head
    can only learn the class prior and the loss parks at ln(num_classes).
    wasteclf.training.trainer warns when a config asks for this.
    """
    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 7, (64, 64))
    probe = np.stack([np.full((64, 64, 3), 30 * i + 20, np.float32) for i in range(7)])
    pooled = np.asarray(
        keras.Model(model.inputs, model.get_layer("pool").output)(probe, training=False)
    )

    # Identical output for seven visibly different images.
    assert pooled.std(axis=0).max() == pytest.approx(0.0, abs=1e-6)
