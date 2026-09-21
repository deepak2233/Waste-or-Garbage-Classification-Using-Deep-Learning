"""SwinConvNeXt: a two-branch backbone fusing windowed attention with convolution.

Follows Kunwar et al., "SwinConvNeXt: a fused deep learning architecture for
real-time garbage image classification", Scientific Reports 15 (2025),
doi:10.1038/s41598-025-91302-7, which reports 98.97% accuracy on the twelve-class
Garbage Classification V2 benchmark.

The idea is that the two families fail differently on waste imagery. Convolution
is good at the local cues that separate visually similar materials: the fibre of
cardboard against paper, the specular highlight on metal against white glass.
Windowed attention is good at the global layout that says whether the frame
holds a crumpled garment or a heap of organic waste. Running both and fusing
them with spatial attention beats either branch on its own; the paper puts each
branch alone in the high seventies to low eighties.

One honest caveat for this implementation. The ConvNeXt branch loads ImageNet
weights, but there is no public Keras checkpoint for Swin, so that branch starts
from scratch. Expect it to need far more epochs than the convolutional branch,
and do not expect the paper's headline number without pretrained Swin weights.
Set ``swin_weights`` to a local file if you have them.
"""

from __future__ import annotations

import keras
from keras import layers, ops

from wasteclf.models.swin import swin_backbone
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


@keras.saving.register_keras_serializable(package="wasteclf")
class SpatialAttention(layers.Layer):
    """CBAM-style spatial attention.

    Pools across channels to get "how much evidence is at this position", then
    learns a single convolution over that 2-channel summary to produce a soft
    mask. Average pooling carries the general response and max pooling the
    strongest one; using both is what the CBAM ablation found necessary.

    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    (arXiv:1807.06521).
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1, kernel_size, padding="same", activation="sigmoid", use_bias=False, name="attn_conv"
        )

    def build(self, input_shape):
        b, h, w, _ = input_shape
        self.conv.build((b, h, w, 2))
        super().build(input_shape)

    def call(self, inputs):
        average = ops.mean(inputs, axis=-1, keepdims=True)
        maximum = ops.max(inputs, axis=-1, keepdims=True)
        attention = self.conv(ops.concatenate([average, maximum], axis=-1))
        return inputs * attention

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


def _align(x, target_height: int, target_width: int, name: str):
    """Match a branch's spatial size to the other one.

    The two branches downsample by the same factor at 224x224, but not at every
    input size, so this resizes rather than assuming.
    """
    if x.shape[1] == target_height and x.shape[2] == target_width:
        return x
    return layers.Resizing(target_height, target_width, name=name)(x)


def build_swinconvnext(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
    fusion_dim: int = 256,
    swin_embed_dim: int = 96,
    swin_depths: tuple[int, ...] = (2, 2, 6, 2),
    swin_heads: tuple[int, ...] = (3, 6, 12, 24),
    swin_window: int = 7,
    swin_drop_path: float = 0.1,
    convnext_variant: str = "tiny",
    name: str = "swinconvnext",
) -> keras.Model:
    """Build the fused backbone, returning a 4-D feature map.

    Both branches take raw RGB in ``[0, 255]``: ConvNeXt normalises inside its
    own stem, and the Swin branch gets an explicit rescaling here. That keeps the
    contract identical to every other backbone in the registry.

    Args:
        input_shape: ``(height, width, 3)``.
        weights: ``imagenet`` loads pretrained ConvNeXt weights. The Swin branch
            is always randomly initialised; no public Keras checkpoint exists.
        fusion_dim: Channels each branch is projected to before fusing.
        swin_embed_dim: Swin patch-embedding width.
        swin_depths: Swin blocks per stage.
        swin_heads: Swin attention heads per stage.
        swin_window: Swin attention window in tokens.
        swin_drop_path: Maximum stochastic-depth rate for the Swin branch.
        convnext_variant: ``tiny``, ``small`` or ``base``.
        name: Model name.

    Returns:
        A model producing ``(batch, h, w, fusion_dim)``.
    """
    from keras.applications import convnext

    variants = {
        "tiny": convnext.ConvNeXtTiny,
        "small": convnext.ConvNeXtSmall,
        "base": convnext.ConvNeXtBase,
    }
    if convnext_variant not in variants:
        raise ValueError(
            f"convnext_variant must be one of {sorted(variants)}, got {convnext_variant!r}"
        )

    inputs = keras.Input(shape=input_shape, name="image")

    # Local branch: pretrained convolution, normalisation inside the stem.
    conv_branch = variants[convnext_variant](
        input_shape=input_shape, include_top=False, weights=weights, name="convnext_branch"
    )
    conv_features = conv_branch(inputs)

    # Global branch: windowed attention, trained from scratch.
    scaled = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="swin_rescale")(inputs)
    swin_branch = swin_backbone(
        input_shape=input_shape,
        embed_dim=swin_embed_dim,
        depths=swin_depths,
        num_heads=swin_heads,
        window_size=swin_window,
        drop_path=swin_drop_path,
        name="swin_branch",
    )
    swin_features = swin_branch(scaled)

    height = min(conv_features.shape[1], swin_features.shape[1])
    width = min(conv_features.shape[2], swin_features.shape[2])
    conv_features = _align(conv_features, height, width, "conv_align")
    swin_features = _align(swin_features, height, width, "swin_align")

    # Project to a shared width so neither branch dominates the concatenation
    # purely by having more channels.
    conv_proj = layers.Conv2D(fusion_dim, 1, name="conv_project")(conv_features)
    conv_proj = layers.LayerNormalization(epsilon=1e-6, name="conv_project_norm")(conv_proj)
    swin_proj = layers.Conv2D(fusion_dim, 1, name="swin_project")(swin_features)
    swin_proj = layers.LayerNormalization(epsilon=1e-6, name="swin_project_norm")(swin_proj)

    conv_proj = SpatialAttention(name="conv_attention")(conv_proj)
    swin_proj = SpatialAttention(name="swin_attention")(swin_proj)

    fused = layers.Concatenate(name="fuse")([conv_proj, swin_proj])
    fused = layers.Conv2D(fusion_dim, 1, name="fuse_project")(fused)
    fused = layers.LayerNormalization(epsilon=1e-6, name="fuse_norm")(fused)
    fused = layers.Activation("gelu", name="fuse_act")(fused)
    outputs = SpatialAttention(name="fuse_attention")(fused)

    model = keras.Model(inputs, outputs, name=name)
    logger.info(
        "built %s: convnext-%s (%s) + swin (scratch), fusion_dim=%d, %.1fM params",
        name,
        convnext_variant,
        weights or "scratch",
        fusion_dim,
        model.count_params() / 1e6,
    )
    return model
