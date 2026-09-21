"""Swin Transformer blocks and the SwinConvNeXt fusion backbone."""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops

from wasteclf.models.swin import (
    PatchMerging,
    SwinBlock,
    swin_backbone,
    window_partition,
    window_reverse,
)

pytestmark = pytest.mark.needs_tf

SMALL = {
    "embed_dim": 24,
    "depths": (2, 2),
    "num_heads": (2, 4),
    "window_size": 4,
    "drop_path": 0.0,
}


# Window mechanics ----------------------------------------------------------


def test_window_partition_round_trips_exactly():
    x = np.random.default_rng(0).normal(size=(2, 8, 8, 5)).astype("float32")
    windows = window_partition(ops.convert_to_tensor(x), 4)
    restored = window_reverse(windows, 4, 8, 8, 5)

    assert tuple(windows.shape) == (8, 16, 5)  # 2 images x 4 windows, 16 tokens each
    assert np.allclose(x, np.asarray(restored))


def test_patch_merging_halves_space_and_doubles_channels():
    layer = PatchMerging(dim=16)
    out = layer(np.zeros((2, 8, 8, 16), dtype="float32"))
    assert tuple(out.shape)[1:] == (4, 4, 32)


# Backbone ------------------------------------------------------------------


def test_swin_tiny_matches_the_published_parameter_count():
    """Swin-T is ~28M. A wrong count means sublayers silently failed to build."""
    model = swin_backbone((224, 224, 3))
    assert model.output_shape == (None, 7, 7, 768)
    assert 25e6 < model.count_params() < 31e6


def test_every_sublayer_is_built_after_construction():
    """compute_output_shape lets Keras skip call(), so build() must be explicit."""
    model = swin_backbone((64, 64, 3), **SMALL)
    blocks = [layer for layer in model.layers if isinstance(layer, SwinBlock)]

    assert blocks
    for block in blocks:
        assert block.built
        assert block.count_params() > 0, f"{block.name} has no parameters"
        assert block.attention.built and block.attention.qkv.built


def test_depths_and_heads_must_agree():
    with pytest.raises(ValueError, match="same length"):
        swin_backbone((64, 64, 3), depths=(2, 2), num_heads=(3,))


def test_alternate_blocks_shift_their_windows():
    """Without the shift, windows never exchange information."""
    model = swin_backbone((64, 64, 3), **SMALL)
    blocks = [layer for layer in model.layers if isinstance(layer, SwinBlock)]
    shifted = [b for b in blocks if b.effective_shift > 0]

    assert shifted, "no block applies a shifted window"
    assert all(b.attention_mask is not None for b in shifted)


def test_window_shrinks_to_fit_a_small_feature_map():
    """A window larger than the feature map would make the reshape fail."""
    model = swin_backbone((32, 32, 3), embed_dim=16, depths=(2, 2), num_heads=(2, 4), window_size=7)
    out = np.asarray(model(np.zeros((1, 32, 32, 3), dtype="float32"), training=False))
    assert np.isfinite(out).all()


def test_inference_is_deterministic():
    model = swin_backbone((64, 64, 3), **SMALL)
    probe = np.random.default_rng(0).uniform(0, 1, (2, 64, 64, 3)).astype("float32")
    assert np.allclose(
        np.asarray(model(probe, training=False)), np.asarray(model(probe, training=False))
    )


def test_stochastic_depth_trains_without_a_shape_error():
    """drop_path > 0 adds a Dropout whose noise shape must match a 4-D tensor."""
    backbone = swin_backbone(
        (32, 32, 3), embed_dim=16, depths=(2,), num_heads=(2,), window_size=4, drop_path=0.3
    )
    model = keras.Sequential(
        [
            backbone,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    x = np.random.default_rng(0).uniform(0, 1, (6, 32, 32, 3)).astype("float32")
    y = keras.utils.to_categorical([0, 1, 2, 0, 1, 2], 3)
    history = model.fit(x, y, epochs=1, batch_size=3, verbose=0)
    assert np.isfinite(history.history["loss"][-1])


@pytest.mark.slow
def test_swin_can_overfit_a_textured_batch():
    """Gradients must reach every block.

    The patterns carry their signal in spatial frequency rather than mean
    colour, because the LayerNorm in the patch embedding normalises each token
    across channels and would erase a flat colour field.
    """
    classes, per_class = 4, 3
    rng = np.random.default_rng(0)
    images, labels = [], []
    for c in range(classes):
        for _ in range(per_class):
            yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
            pattern = np.sin((0.3 + 0.4 * c) * xx + 0.5 * c * yy) * 60 + 128
            pattern = pattern + rng.normal(0, 3, (32, 32))
            images.append(np.stack([pattern, np.roll(pattern, c, 0), np.roll(pattern, c, 1)], -1))
            labels.append(c)

    x = np.stack(images).astype("float32")
    y = keras.utils.to_categorical(labels, classes)

    backbone = swin_backbone(
        (32, 32, 3), embed_dim=24, depths=(2, 2), num_heads=(2, 4), window_size=4
    )
    model = keras.Sequential(
        [
            backbone,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(2e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(x, y, epochs=100, batch_size=len(x), verbose=0)

    assert history.history["accuracy"][-1] == 1.0
    assert history.history["loss"][-1] < 0.1


# Fusion --------------------------------------------------------------------


def test_spatial_attention_preserves_shape_and_bounds_the_mask():
    from wasteclf.models.swinconvnext import SpatialAttention

    layer = SpatialAttention()
    x = np.random.default_rng(0).uniform(0, 1, (2, 8, 8, 16)).astype("float32")
    out = np.asarray(layer(x))

    assert out.shape == x.shape
    # A sigmoid gate can only scale down.
    assert np.all(np.abs(out) <= np.abs(x) + 1e-5)


def test_swinconvnext_fuses_both_branches():
    from wasteclf.models.swinconvnext import build_swinconvnext

    model = build_swinconvnext(
        (64, 64, 3),
        weights=None,
        fusion_dim=32,
        swin_embed_dim=16,
        swin_depths=(2, 2),
        swin_heads=(2, 4),
        swin_window=4,
    )
    names = [layer.name for layer in model.layers]

    assert "convnext_branch" in names
    assert "swin_branch" in names
    assert model.output_shape[-1] == 32


def test_swinconvnext_rejects_an_unknown_variant():
    from wasteclf.models.swinconvnext import build_swinconvnext

    with pytest.raises(ValueError, match="convnext_variant"):
        build_swinconvnext((64, 64, 3), weights=None, convnext_variant="enormous")


def test_swinconvnext_is_reachable_through_the_registry():
    """Config-driven training needs the registry entry to accept branch kwargs."""
    from wasteclf.config import ModelConfig
    from wasteclf.models.build import build_model, get_backbone

    model = build_model(
        ModelConfig(
            backbone="swinconvnext",
            weights=None,
            hidden_units=16,
            backbone_kwargs={
                "fusion_dim": 24,
                "swin_embed_dim": 16,
                "swin_depths": [2],
                "swin_heads": [2],
                "swin_window": 4,
            },
        ),
        num_classes=12,
        image_size=(64, 64),
    )

    assert model.output_shape == (None, 12)
    assert get_backbone(model).name == "backbone"
