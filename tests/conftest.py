"""Shared fixtures.

Fixtures build a synthetic dataset on the fly, so the suite runs on a clean
checkout with no download and no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """A small imbalanced dataset: 7 classes, counts from 4 to 20."""
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    counts = {
        "cardboard": 20,
        "compost": 4,
        "glass": 18,
        "metal": 16,
        "paper": 20,
        "plastic": 15,
        "trash": 5,
    }
    for index, (name, count) in enumerate(counts.items()):
        folder = root / name
        folder.mkdir()
        base = np.array([40 * index % 255, 90, 200 - 20 * index], dtype=np.float64)
        for i in range(count):
            pixels = np.clip(base + rng.normal(0, 15, (32, 32, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(folder / f"{name}_{i:02d}.jpg")
    return root


@pytest.fixture(scope="session")
def class_counts() -> dict[str, int]:
    return {
        "cardboard": 20,
        "compost": 4,
        "glass": 18,
        "metal": 16,
        "paper": 20,
        "plastic": 15,
        "trash": 5,
    }


@pytest.fixture
def messy_root(tmp_path) -> Path:
    """A dataset with the rubbish a real download contains."""
    root = tmp_path / "messy"
    (root / "glass").mkdir(parents=True)
    (root / "metal").mkdir(parents=True)

    rng = np.random.default_rng(1)
    for i in range(6):
        pixels = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(root / "glass" / f"g{i}.jpg")
        Image.fromarray(pixels, mode="RGB").save(root / "metal" / f"m{i}.png")

    (root / "glass" / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / "metal" / "empty.jpg").write_bytes(b"")
    (root / "glass" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return root


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_root):
    """Train one minimal model and share it across the slow test modules.

    Imports live in the body so that collecting the fast tests does not pay for
    importing TensorFlow.

    ``model.weights=None`` keeps this offline and quick. The model therefore
    learns almost nothing, which is fine: these tests assert on plumbing,
    artefacts and shapes, not on accuracy.
    """
    from wasteclf.config import Config
    from wasteclf.data.manifest import build_manifest
    from wasteclf.data.pipeline import build_datasets
    from wasteclf.training.trainer import train

    cfg = Config.load(
        None,
        {
            "data.root": str(synthetic_root),
            "data.image_size": [32, 32],
            "data.batch_size": 4,
            "data.cache": False,
            "model.backbone": "mobilenetv2",
            "model.weights": None,
            "model.hidden_units": 8,
            "train.warmup.epochs": 1,
            "train.finetune.epochs": 1,
            "train.finetune.unfreeze_layers": 4,
            "train.monitor": "val_loss",
            "output_dir": str(tmp_path_factory.mktemp("runs")),
            "run_name": "e2e",
            "seed": 0,
        },
    )
    manifest = build_manifest(
        cfg.data.root,
        cfg.data.train_split,
        cfg.data.val_split,
        cfg.data.test_split,
        seed=cfg.data.split_seed,
    )
    datasets = build_datasets(manifest, cfg.data, seed=cfg.seed)
    result = train(cfg, manifest, datasets)
    return result, manifest, datasets
