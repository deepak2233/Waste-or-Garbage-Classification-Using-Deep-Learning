"""Dataset scanning, splitting and class weights."""

from __future__ import annotations

import pytest

from wasteclf.data.manifest import DatasetManifest, Split, build_manifest, class_weights


def test_scan_finds_every_class(synthetic_root, class_counts):
    manifest = build_manifest(synthetic_root)
    assert manifest.class_names == sorted(class_counts)
    assert len(manifest.records) == sum(class_counts.values())


def test_splits_are_disjoint_and_cover_everything(synthetic_root, class_counts):
    manifest = build_manifest(synthetic_root)
    by_split = {s: set(manifest.paths(s)) for s in Split}

    assert not by_split[Split.TRAIN] & by_split[Split.VAL]
    assert not by_split[Split.TRAIN] & by_split[Split.TEST]
    assert not by_split[Split.VAL] & by_split[Split.TEST]
    assert sum(len(v) for v in by_split.values()) == sum(class_counts.values())


def test_every_class_appears_in_every_split(synthetic_root):
    """The smallest class has 4 images; it must still reach val and test."""
    manifest = build_manifest(synthetic_root)
    for split in Split:
        counts = manifest.counts(split)
        assert all(n > 0 for n in counts.values()), f"{split.value} is missing a class: {counts}"


def test_split_is_deterministic_for_a_fixed_seed(synthetic_root):
    first = build_manifest(synthetic_root, seed=42)
    second = build_manifest(synthetic_root, seed=42)
    assert first.records == second.records


def test_changing_the_seed_changes_the_split(synthetic_root):
    first = build_manifest(synthetic_root, seed=1)
    second = build_manifest(synthetic_root, seed=2)
    assert first.records != second.records


def test_split_proportions_are_roughly_honoured(synthetic_root, class_counts):
    manifest = build_manifest(synthetic_root, 0.7, 0.15, 0.15)
    total = sum(class_counts.values())
    train = sum(1 for _, _, s in manifest.records if s is Split.TRAIN)
    # Per-class rounding on small classes moves this a few images either way.
    assert 0.60 <= train / total <= 0.78


def test_manifest_round_trips_through_csv(synthetic_root, tmp_path):
    original = build_manifest(synthetic_root)
    path = original.save(tmp_path / "manifest.csv")
    reloaded = DatasetManifest.load(path, synthetic_root)
    assert reloaded.class_names == original.class_names
    assert reloaded.records == original.records


def test_non_images_and_empty_files_are_rejected(messy_root):
    manifest = build_manifest(messy_root, verify_images=True)
    reasons = {reason for _, reason in manifest.rejected}
    assert any("extension" in r for r in reasons)
    assert "empty file" in reasons
    assert "unreadable image" in reasons
    # The 12 good images survive.
    assert len(manifest.records) == 12


def test_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        build_manifest("does/not/exist")


def test_empty_root_raises(tmp_path):
    with pytest.raises(ValueError, match="no class subdirectories"):
        build_manifest(tmp_path)


def test_class_weights_favour_rare_classes():
    labels = [0] * 90 + [1] * 10
    weights = class_weights(labels, 2)
    assert weights[1] > weights[0]
    # Normalised to mean 1.0 so the effective learning rate does not change.
    assert sum(weights.values()) / 2 == pytest.approx(1.0)


def test_class_weights_handle_absent_classes():
    weights = class_weights([0, 0, 1], num_classes=4)
    assert weights[2] > 0 and weights[3] > 0


def test_class_weights_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        class_weights([], 3)
