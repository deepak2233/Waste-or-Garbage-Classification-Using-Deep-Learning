"""Configuration loading, overrides and validation."""

from __future__ import annotations

import pytest
import yaml

from wasteclf.config import Config, ConfigError


def test_defaults_are_valid():
    cfg = Config().validate()
    assert cfg.model.backbone == "vgg16"
    assert cfg.data.train_split + cfg.data.val_split + cfg.data.test_split == pytest.approx(1.0)


def test_dotted_overrides_reach_nested_dataclasses():
    cfg = Config.load(None, {"train.finetune.epochs": 3, "model.backbone": "resnet50"})
    assert cfg.train.finetune.epochs == 3
    assert cfg.model.backbone == "resnet50"
    # Untouched siblings keep their defaults.
    assert cfg.train.warmup.epochs == 20


def test_image_size_accepts_a_yaml_list():
    cfg = Config.load(None, {"data.image_size": [160, 192]})
    assert cfg.data.image_size == (160, 192)


def test_yaml_round_trip(tmp_path):
    original = Config.load(None, {"model.backbone": "mobilenetv2", "seed": 7})
    path = original.save(tmp_path / "cfg.yaml")
    reloaded = Config.load(path)
    assert reloaded.to_dict() == original.to_dict()


def test_splits_must_sum_to_one():
    with pytest.raises(ConfigError, match="sum to 1.0"):
        Config.load(None, {"data.train_split": 0.8, "data.val_split": 0.3})


def test_unknown_key_is_rejected_rather_than_ignored():
    # A typo must not silently train a different model than intended.
    with pytest.raises(ConfigError, match="unknown key"):
        Config.load(None, {"model.backbon": "vgg16"})


def test_bad_pooling_is_rejected():
    with pytest.raises(ConfigError, match="pooling"):
        Config.load(None, {"model.pooling": "global"})


def test_bad_monitor_is_rejected():
    with pytest.raises(ConfigError, match="monitor"):
        Config.load(None, {"train.monitor": "val_f1_score"})


@pytest.mark.parametrize(
    "name", ["base", "vgg16", "resnet50", "mobilenetv2", "efficientnetb0", "smoke"]
)
def test_shipped_configs_load(name):
    """Every config in configs/ must parse and validate."""
    cfg = Config.load(f"configs/{name}.yaml")
    assert cfg.model.backbone
    assert cfg.data.batch_size >= 1


def test_config_file_must_be_a_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(["a", "list"]), encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        Config.load(path)
