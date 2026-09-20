"""Model assembly and fine-tuning control."""

from __future__ import annotations

import keras
from keras import layers

from wasteclf.config import AugmentConfig, ModelConfig
from wasteclf.data.augment import build_augmentation
from wasteclf.models.backbones import BackbonePreprocessing, get_backbone_spec
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


def build_model(
    model_cfg: ModelConfig,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    augment_cfg: AugmentConfig | None = None,
    seed: int = 42,
) -> keras.Model:
    """Assemble the full classifier.

    Layer order::

        input (uint8/float 0-255 RGB)
          -> augmentation      (training only, or absent)
          -> backbone preprocessing
          -> frozen backbone
          -> pooling
          -> dense head
          -> softmax

    Augmentation sits before preprocessing so rotation fill values and
    brightness shifts operate in pixel space, where "reflect padding" and
    "0-255" mean what they say. Rotating after mean subtraction fills the
    corners with a value that is not a colour.

    Args:
        model_cfg: Backbone, pooling and head settings.
        num_classes: Number of output classes.
        image_size: Input ``(height, width)``.
        augment_cfg: Augmentation settings, or ``None`` to skip augmentation.
        seed: Seed for augmentation layers.

    Returns:
        An uncompiled :class:`keras.Model` with the backbone frozen.
    """
    spec = get_backbone_spec(model_cfg.backbone)
    input_shape = (*image_size, 3)

    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    if augment_cfg is not None:
        augmentation = build_augmentation(augment_cfg, seed=seed)
        if augmentation is not None:
            x = augmentation(x)

    x = BackbonePreprocessing(spec.name, name="preprocess")(x)

    backbone = spec.build(input_shape=input_shape, weights=model_cfg.weights, name="backbone")
    # Stage one always trains the head alone. set_finetune_trainable() reverses
    # this for stage two.
    backbone.trainable = False
    x = backbone(x, training=False)

    if model_cfg.pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="pool")(x)
    elif model_cfg.pooling == "max":
        x = layers.GlobalMaxPooling2D(name="pool")(x)
    else:
        # Flatten keeps the spatial layout but explodes the head: on VGG16 at
        # 224x224 that is 25,088 features into the first Dense layer against 512
        # for average pooling. Only worth it when position carries signal.
        x = layers.Flatten(name="pool")(x)

    regulariser = keras.regularizers.l2(model_cfg.l2) if model_cfg.l2 > 0 else None

    if model_cfg.hidden_units > 0:
        x = layers.Dense(
            model_cfg.hidden_units,
            activation="relu",
            kernel_regularizer=regulariser,
            name="head_dense",
        )(x)
        x = layers.BatchNormalization(name="head_bn")(x)

    if model_cfg.dropout > 0:
        x = layers.Dropout(model_cfg.dropout, seed=seed, name="head_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regulariser,
        dtype="float32",  # keep the softmax in fp32 under mixed precision
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name=f"wasteclf_{spec.name}")
    logger.info(
        "built %s: %d classes, %s pooling, %.1fM total params",
        spec.name,
        num_classes,
        model_cfg.pooling,
        model.count_params() / 1e6,
    )
    return model


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the backbone submodel.

    Looks the layer up by name first. Older Keras releases ignore the ``name``
    argument to ``keras.applications``, so the fallback finds it structurally:
    the backbone is the only nested Functional model in the graph, and the
    augmentation block is a ``Sequential``.
    """
    for layer in model.layers:
        if layer.name == "backbone":
            return layer

    nested = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and not isinstance(layer, keras.Sequential)
    ]
    if len(nested) == 1:
        return nested[0]

    raise ValueError(
        f"cannot identify the backbone inside model {model.name!r}: found "
        f"{len(nested)} candidate submodels. Was it built by build_model()?"
    )


def inference_model(model: keras.Model) -> keras.Model:
    """Return the model with the augmentation block removed.

    Augmentation layers are already inert at inference, so this changes no
    prediction. It matters for export: tracing them produces
    ``StatelessRandomUniformV2`` and ``ImageProjectiveTransformV3`` nodes, which
    have no ONNX equivalent. The converter drops them and writes a file that
    onnxruntime then refuses to load as an invalid graph.

    The original layer objects are reused, so the returned model shares weights
    with ``model`` rather than copying them.
    """
    if not any(layer.name == "augmentation" for layer in model.layers):
        return model

    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs
    for layer in model.layers[1:]:
        if layer.name == "augmentation":
            continue
        x = layer(x)

    stripped = keras.Model(inputs, x, name=f"{model.name}_inference")
    logger.info("built inference graph without augmentation (%d layers)", len(stripped.layers))
    return stripped


def set_finetune_trainable(
    model: keras.Model, unfreeze_layers: int, freeze_batchnorm: bool = True
) -> int:
    """Unfreeze the top ``unfreeze_layers`` layers of the backbone.

    Args:
        model: A model from :func:`build_model`.
        unfreeze_layers: How many layers to unfreeze, counted from the output
            end. ``0`` leaves the backbone frozen. ``-1`` unfreezes all of it.
        freeze_batchnorm: Keep BatchNormalization layers frozen even inside the
            unfrozen region. On a dataset of a few thousand images the batch
            statistics are noisy enough that updating them degrades the
            pretrained features, and the damage shows up as validation accuracy
            that collapses in the first fine-tuning epoch.

    Returns:
        The number of layers actually made trainable.
    """
    backbone = get_backbone(model)

    if unfreeze_layers == 0:
        backbone.trainable = False
        logger.info("backbone stays frozen")
        return 0

    backbone.trainable = True
    all_layers = backbone.layers
    # Negative means "unfreeze everything", so the boundary sits at layer 0.
    boundary = 0 if unfreeze_layers < 0 else max(0, len(all_layers) - unfreeze_layers)

    unfrozen = 0
    frozen_bn = 0
    for i, layer in enumerate(all_layers):
        if i < boundary:
            layer.trainable = False
            continue
        if freeze_batchnorm and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
            continue
        layer.trainable = True
        unfrozen += 1

    trainable_params = sum(int(w.numpy().size) for w in backbone.trainable_weights)
    logger.info(
        "unfroze %d/%d backbone layers (%d BatchNorm kept frozen, %.2fM trainable params)",
        unfrozen,
        len(all_layers),
        frozen_bn,
        trainable_params / 1e6,
    )
    return unfrozen
