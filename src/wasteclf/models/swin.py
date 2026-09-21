"""Swin Transformer blocks.

Keras does not ship a Swin implementation and the pretrained checkpoints live
on Kaggle Hub, so this is built from the paper: Liu et al., "Swin Transformer:
Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
(arXiv:2103.14030).

Self-attention runs inside fixed windows, which makes cost linear in the number
of pixels rather than quadratic. Alternating blocks shift the window grid by
half a window so information crosses window boundaries; without that the
windows never talk to each other and the receptive field stops growing.

Trained from scratch. On a dataset of this size that is a real disadvantage
against an ImageNet-pretrained convolutional branch, which is exactly why
:mod:`wasteclf.models.swinconvnext` pairs the two rather than using this alone.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="wasteclf")
class PatchEmbedding(layers.Layer):
    """Split the image into non-overlapping patches and project each one.

    A strided convolution does both at once: kernel and stride equal to the
    patch size means every output position sees exactly one patch.
    """

    def __init__(self, patch_size: int = 4, embed_dim: int = 96, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, name="proj"
        )
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

    def build(self, input_shape):
        # Build sublayers here rather than letting them build lazily on first
        # call. Keras uses compute_output_shape() during functional graph
        # construction and never calls call(), so a lazily built sublayer stays
        # unbuilt: the model then reports almost no parameters and the optimizer
        # is handed an incomplete variable list.
        self.projection.build(input_shape)
        b, h, w, _ = input_shape
        self.norm.build((b, h // self.patch_size, w // self.patch_size, self.embed_dim))
        super().build(input_shape)

    def call(self, inputs):
        return self.norm(self.projection(inputs))

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        return (b, h // self.patch_size, w // self.patch_size, self.embed_dim)

    def get_config(self):
        return {**super().get_config(), "patch_size": self.patch_size, "embed_dim": self.embed_dim}


def window_partition(x, window_size: int):
    """(B, H, W, C) -> (B * num_windows, window_size ** 2, C)."""
    b, h, w, c = ops.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    x = ops.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window_size * window_size, c))


def window_reverse(windows, window_size: int, height: int, width: int, channels: int):
    """Inverse of :func:`window_partition`."""
    x = ops.reshape(
        windows,
        (-1, height // window_size, width // window_size, window_size, window_size, channels),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, height, width, channels))


@keras.saving.register_keras_serializable(package="wasteclf")
class WindowAttention(layers.Layer):
    """Multi-head self-attention within a window, with relative position bias.

    The bias is learned per (head, relative offset) pair. Absolute position
    embeddings would be wrong here: a window carries no notion of where it sits
    in the image, so only the offsets between tokens inside it are meaningful.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=True, name="qkv")
        self.projection = layers.Dense(dim, name="proj")
        self.attention_dropout = layers.Dropout(dropout)
        self.projection_dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        size = self.window_size
        # (2W-1)^2 distinct offsets between any two positions in the window.
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * size - 1) ** 2, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table",
        )

        coords = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"))
        coords = coords.reshape(2, -1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.transpose(1, 2, 0)
        relative[:, :, 0] += size - 1  # shift to start from 0
        relative[:, :, 1] += size - 1
        relative[:, :, 0] *= 2 * size - 1
        index = relative.sum(-1).astype("int32")

        self.relative_position_index = self.add_weight(
            shape=index.shape,
            initializer=keras.initializers.Constant(index),
            trainable=False,
            dtype="int32",
            name="relative_position_index",
        )

        self.qkv.build(input_shape)
        self.projection.build((input_shape[0], input_shape[1], self.dim))
        super().build(input_shape)

    # The argument is deliberately not called `mask`: Keras reserves that name
    # for its own masking protocol and warns that this layer discards it.
    def call(self, inputs, attention_mask=None, training=None):
        tokens = inputs.shape[1]
        channels = inputs.shape[2]
        head_dim = channels // self.num_heads

        qkv = self.qkv(inputs)
        qkv = ops.reshape(qkv, (-1, tokens, 3, self.num_heads, head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = ops.matmul(query * self.scale, ops.transpose(key, (0, 1, 3, 2)))

        bias = ops.take(
            self.relative_position_bias_table,
            ops.reshape(self.relative_position_index, (-1,)),
            axis=0,
        )
        bias = ops.reshape(bias, (tokens, tokens, self.num_heads))
        bias = ops.transpose(bias, (2, 0, 1))
        attention = attention + ops.expand_dims(bias, 0)

        if attention_mask is not None:
            windows = attention_mask.shape[0]
            attention = ops.reshape(attention, (-1, windows, self.num_heads, tokens, tokens))
            attention = attention + ops.expand_dims(ops.expand_dims(attention_mask, 1), 0)
            attention = ops.reshape(attention, (-1, self.num_heads, tokens, tokens))

        attention = ops.softmax(attention, axis=-1)
        attention = self.attention_dropout(attention, training=training)

        out = ops.matmul(attention, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, tokens, channels))
        return self.projection_dropout(self.projection(out), training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        }


def _shift_mask(height: int, width: int, window_size: int, shift: int) -> np.ndarray:
    """Attention mask that stops rolled-in edges attending to each other.

    The cyclic shift wraps opposite edges of the image into the same window.
    Those tokens are not neighbours, so the mask drives their attention logits
    to a large negative number before the softmax.
    """
    regions = np.zeros((1, height, width, 1), dtype="float32")
    slices = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    count = 0
    for h in slices:
        for w in slices:
            regions[:, h, w, :] = count
            count += 1

    windows = regions.reshape(
        1, height // window_size, window_size, width // window_size, window_size, 1
    )
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
    mask = windows[:, None, :] - windows[:, :, None]
    return np.where(mask != 0, -100.0, 0.0).astype("float32")


def fit_window(window_size: int, height: int, width: int) -> int:
    """Largest window no bigger than ``window_size`` that tiles the feature map.

    Windows must divide the map exactly, because partitioning reshapes rather
    than pads. Swin's own 7x7 window is chosen to divide 56, 28, 14 and 7, which
    is why the reference configuration starts from 224x224. At other input sizes
    a stage can land on a resolution 7 does not divide, so shrink rather than
    fail: an 8x8 map gets a 4x4 window.
    """
    limit = min(window_size, height, width)
    for size in range(limit, 0, -1):
        if height % size == 0 and width % size == 0:
            return size
    return 1


@keras.saving.register_keras_serializable(package="wasteclf")
class SwinBlock(layers.Layer):
    """One Swin block: windowed attention, then an MLP, both with residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.drop_path_rate = drop_path

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        # The attention layer is created in build(), not here: its relative
        # position table is sized from the window, and the window that actually
        # tiles this feature map is only known once the input shape is.
        self.attention: WindowAttention | None = None
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.mlp_hidden = layers.Dense(int(dim * mlp_ratio), activation="gelu", name="mlp_fc1")
        self.mlp_out = layers.Dense(dim, name="mlp_fc2")
        self.mlp_dropout = layers.Dropout(dropout)
        # Stochastic depth: drops the whole residual branch for a sample. The
        # noise shape must match the rank of the tensor it is applied to, which
        # here is (batch, height, width, channels); broadcasting the last three
        # dimensions is what makes the drop per-sample rather than per-unit.
        self.drop_path = (
            layers.Dropout(drop_path, noise_shape=(None, 1, 1, 1)) if drop_path > 0 else None
        )

    def build(self, input_shape):
        self.height, self.width = input_shape[1], input_shape[2]

        self.effective_window = fit_window(self.window_size, self.height, self.width)
        # One window covering the whole map leaves nothing to shift into, so the
        # block degrades to plain windowed attention.
        if self.effective_window >= min(self.height, self.width):
            self.effective_shift = 0
        else:
            # Keep the shift inside the window: a larger roll moves content past
            # one window and the mask no longer describes what overlaps what.
            self.effective_shift = min(self.shift_size, self.effective_window // 2)

        if self.effective_shift > 0:
            mask = _shift_mask(self.height, self.width, self.effective_window, self.effective_shift)
            self.attention_mask = self.add_weight(
                shape=mask.shape,
                initializer=keras.initializers.Constant(mask),
                trainable=False,
                name="attention_mask",
            )
        else:
            self.attention_mask = None

        # Same reason as PatchEmbedding: build the sublayers rather than
        # relying on a call() that functional construction never makes.
        batch, _, _, channels = input_shape
        self.attention = WindowAttention(
            self.dim, self.effective_window, self.num_heads, self.dropout_rate, name="attn"
        )
        self.norm1.build(input_shape)
        self.attention.build((None, self.effective_window**2, channels))
        self.norm2.build(input_shape)
        self.mlp_hidden.build(input_shape)
        self.mlp_out.build((batch, self.height, self.width, int(self.dim * self.mlp_ratio)))
        super().build(input_shape)

    def call(self, inputs, training=None):
        height, width, channels = self.height, self.width, self.dim
        window = self.effective_window
        shift = self.effective_shift

        shortcut = inputs
        x = self.norm1(inputs)

        if shift > 0:
            x = ops.roll(x, shift=(-shift, -shift), axis=(1, 2))

        windows = window_partition(x, window)
        attended = self.attention(windows, attention_mask=self.attention_mask, training=training)
        x = window_reverse(attended, window, height, width, channels)

        if shift > 0:
            x = ops.roll(x, shift=(shift, shift), axis=(1, 2))

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)
        x = shortcut + x

        residual = x
        y = self.norm2(x)
        y = self.mlp_dropout(self.mlp_hidden(y), training=training)
        y = self.mlp_dropout(self.mlp_out(y), training=training)
        if self.drop_path is not None:
            y = self.drop_path(y, training=training)
        return residual + y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout_rate,
            "drop_path": self.drop_path_rate,
        }


@keras.saving.register_keras_serializable(package="wasteclf")
class PatchMerging(layers.Layer):
    """Halve the spatial resolution and double the channels.

    Concatenates each 2x2 neighbourhood (4C) then projects to 2C, so the stage
    transition keeps the parameter count in check rather than growing it 4x.
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")

    def build(self, input_shape):
        b, h, w, _ = input_shape
        merged = (b, h // 2, w // 2, 4 * self.dim)
        self.norm.build(merged)
        self.reduction.build(merged)
        super().build(input_shape)

    def call(self, inputs):
        height, width = inputs.shape[1], inputs.shape[2]
        x0 = inputs[:, 0::2, 0::2, :]
        x1 = inputs[:, 1::2, 0::2, :]
        x2 = inputs[:, 0::2, 1::2, :]
        x3 = inputs[:, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = ops.reshape(x, (-1, height // 2, width // 2, 4 * self.dim))
        return self.reduction(self.norm(x))

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        return (b, h // 2, w // 2, 2 * self.dim)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


def swin_backbone(
    input_shape: tuple[int, int, int],
    embed_dim: int = 96,
    depths: tuple[int, ...] = (2, 2, 6, 2),
    num_heads: tuple[int, ...] = (3, 6, 12, 24),
    window_size: int = 7,
    patch_size: int = 4,
    dropout: float = 0.0,
    drop_path: float = 0.1,
    name: str = "swin",
) -> keras.Model:
    """Build a Swin backbone returning a 4-D feature map.

    Defaults match Swin-T from the paper: 28M parameters at 224x224.

    Args:
        input_shape: ``(height, width, 3)``.
        embed_dim: Channels after patch embedding; doubles per stage.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Attention window, in tokens.
        patch_size: Pixels per patch on the input.
        dropout: Dropout inside attention and the MLP.
        drop_path: Maximum stochastic-depth rate, ramped linearly over blocks.

    Returns:
        A model whose output is ``(batch, h, w, embed_dim * 2 ** (len(depths) - 1))``.
    """
    if len(depths) != len(num_heads):
        raise ValueError(f"depths and num_heads must be the same length: {depths} vs {num_heads}")

    inputs = keras.Input(shape=input_shape)
    x = PatchEmbedding(patch_size, embed_dim, name="patch_embed")(inputs)

    total_blocks = sum(depths)
    rates = np.linspace(0.0, drop_path, total_blocks) if total_blocks else []
    block_index = 0

    for stage, (depth, heads) in enumerate(zip(depths, num_heads)):
        dim = embed_dim * (2**stage)
        for block in range(depth):
            x = SwinBlock(
                dim=dim,
                num_heads=heads,
                window_size=window_size,
                # Alternate: even blocks use plain windows, odd blocks shift.
                shift_size=0 if block % 2 == 0 else window_size // 2,
                dropout=dropout,
                drop_path=float(rates[block_index]) if total_blocks else 0.0,
                name=f"stage{stage}_block{block}",
            )(x)
            block_index += 1
        if stage < len(depths) - 1:
            x = PatchMerging(dim, name=f"stage{stage}_merge")(x)

    x = layers.LayerNormalization(epsilon=1e-5, name="norm")(x)
    return keras.Model(inputs, x, name=name)
