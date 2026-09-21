"""Typed configuration objects and YAML loading.

Every knob that changes a result lives here, gets written into the run
directory, and is read back at inference time. If a number cannot be traced to
the settings that produced it, it is not a result.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically wrong."""


@dataclass
class DataConfig:
    """Where the images are and how they are split."""

    root: str = "data/raw"
    #: Fractions of the *whole* dataset. Must sum to 1.0 within 1e-6.
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    #: Seed for the stratified split. Fixed by default so the split is stable
    #: across runs; change it deliberately when you want a different partition.
    split_seed: int = 42
    shuffle_buffer: int = 1000
    cache: bool = True
    #: Open every image during the scan and drop the unreadable ones. Costs one
    #: pass over the data and reports exactly which files are bad, which is the
    #: right place to find out rather than 40 minutes into a training run.
    verify_images: bool = True
    #: Swallow decode errors inside the tf.data graph instead. This makes the
    #: dataset cardinality unknown, so Keras cannot compute steps_per_epoch and
    #: warns once per epoch. Prefer ``verify_images``; reach for this only when
    #: the dataset is too large to pre-verify.
    skip_corrupt: bool = False

    def validate(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"train/val/test splits must sum to 1.0, got {total:.6f} "
                f"({self.train_split} + {self.val_split} + {self.test_split})"
            )
        for name in ("train_split", "val_split", "test_split"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must be strictly between 0 and 1, got {value}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        h, w = self.image_size
        if h < 32 or w < 32:
            raise ConfigError(f"image_size must be at least 32x32, got {self.image_size}")


@dataclass
class AugmentConfig:
    """Train-time augmentation.

    Applied to the training split only. Augmenting val or test measures accuracy
    on randomly distorted images instead of on the held-out set.
    """

    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: float = 0.15  # fraction of 2*pi
    zoom: float = 0.15
    translation: float = 0.1
    contrast: float = 0.1
    brightness: float = 0.0


@dataclass
class ModelConfig:
    """Backbone choice and classifier head."""

    backbone: str = "vgg16"
    #: ``imagenet`` or ``null`` for random init.
    weights: str | None = "imagenet"
    pooling: str = "avg"  # avg | max | flatten
    hidden_units: int = 128
    dropout: float = 0.5
    #: L2 penalty on the head's dense layers. 0.0 disables it.
    l2: float = 0.0
    label_smoothing: float = 0.0
    #: Extra keyword arguments handed to the backbone constructor. Only some
    #: backbones take any; swinconvnext uses these to size its two branches
    #: (fusion_dim, swin_depths, swin_heads, swin_window, convnext_variant).
    backbone_kwargs: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.pooling not in {"avg", "max", "flatten"}:
            raise ConfigError(f"pooling must be avg, max or flatten, got {self.pooling!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")


@dataclass
class StageConfig:
    """One training stage.

    Training runs in two stages. Stage one trains the head with the backbone
    frozen. Stage two unfreezes the top of the backbone at a much lower learning
    rate. Running stage two from the start destroys the pretrained features,
    because the randomly initialised head emits large gradients on the first
    batches.
    """

    epochs: int = 20
    learning_rate: float = 1e-3
    #: Number of backbone layers to unfreeze from the top. 0 freezes the whole
    #: backbone. ``-1`` unfreezes everything.
    unfreeze_layers: int = 0
    #: Keep BatchNormalization layers in inference mode when fine-tuning. This
    #: matters for ResNet and EfficientNet; leaving BN trainable on a small
    #: dataset wrecks the running statistics.
    freeze_batchnorm: bool = True

    def validate(self) -> None:
        if self.epochs < 0:
            raise ConfigError(f"epochs must be >= 0, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.unfreeze_layers < -1:
            raise ConfigError(
                f"unfreeze_layers must be -1 (all) or >= 0, got {self.unfreeze_layers}"
            )


@dataclass
class TrainConfig:
    """Optimisation, callbacks and class balancing."""

    warmup: StageConfig = field(default_factory=lambda: StageConfig(epochs=20, learning_rate=1e-3))
    finetune: StageConfig = field(
        default_factory=lambda: StageConfig(epochs=25, learning_rate=1e-5, unfreeze_layers=8)
    )
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    #: Weight the loss by inverse class frequency. The dataset is imbalanced
    #: (compost and trash are the small classes), so an unweighted model can
    #: score well on accuracy while never predicting them.
    class_weights: bool = True
    monitor: str = "val_loss"
    mixed_precision: bool = False

    def validate(self) -> None:
        self.warmup.validate()
        self.finetune.validate()
        if self.monitor not in {"val_loss", "val_accuracy", "val_macro_f1"}:
            raise ConfigError(f"unsupported monitor metric {self.monitor!r}")


@dataclass
class Config:
    """Root configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    output_dir: str = "runs"
    run_name: str | None = None

    def validate(self) -> Config:
        self.data.validate()
        self.model.validate()
        self.train.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Config:
        return _build(cls, raw)

    @classmethod
    def load(
        cls, path: str | Path | None = None, overrides: Mapping[str, Any] | None = None
    ) -> Config:
        """Load a YAML config, apply dotted-key overrides, and validate.

        ``overrides`` keys use dots for nesting, e.g. ``{"train.warmup.epochs": 3}``.
        """
        raw: dict[str, Any] = {}
        if path is not None:
            text = Path(path).read_text(encoding="utf-8")
            loaded = yaml.safe_load(text) or {}
            if not isinstance(loaded, dict):
                raise ConfigError(
                    f"{path} must contain a YAML mapping, got {type(loaded).__name__}"
                )
            raw = loaded
        if overrides:
            for dotted, value in overrides.items():
                _set_nested(raw, dotted, value)
        return cls.from_dict(raw).validate()


def _set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cursor = target
    for key in keys[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[keys[-1]] = value


def _build(cls: type, raw: Mapping[str, Any]) -> Any:
    """Recursively construct a nested dataclass from a mapping.

    Unknown keys are an error rather than a silent no-op: a typo in a config
    file should not quietly train a different model than you asked for.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(fields))}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in raw:
            continue
        value = raw[name]
        if dataclasses.is_dataclass(f.type) or (
            isinstance(f.type, str) and f.type in _DATACLASS_BY_NAME
        ):
            nested = _DATACLASS_BY_NAME[f.type] if isinstance(f.type, str) else f.type
            if not isinstance(value, Mapping):
                raise ConfigError(f"{name} must be a mapping, got {type(value).__name__}")
            kwargs[name] = _build(nested, value)
        elif name == "image_size" and isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ConfigError(f"image_size must have exactly 2 entries, got {len(value)}")
            kwargs[name] = (int(value[0]), int(value[1]))
        else:
            kwargs[name] = value
    return cls(**kwargs)


_DATACLASS_BY_NAME: dict[str, type] = {
    "DataConfig": DataConfig,
    "AugmentConfig": AugmentConfig,
    "ModelConfig": ModelConfig,
    "TrainConfig": TrainConfig,
    "StageConfig": StageConfig,
}
