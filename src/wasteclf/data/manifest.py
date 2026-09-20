"""Dataset scanning and deterministic stratified splitting.

Three disjoint splits: train fits the weights, val drives early stopping and
checkpointing, test is read once at the end. Keeping val and test separate is
what stops the reported number being measured on data the model was selected
against.

The exact file assignment goes to a CSV so a later evaluation runs on the same
held-out images rather than a fresh random split.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from wasteclf.constants import IMAGE_EXTENSIONS
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class DatasetManifest:
    """File paths, labels and split assignment for one dataset."""

    root: Path
    class_names: list[str]
    #: ``(relative_path, class_index, split)`` for every usable image.
    records: list[tuple[str, int, Split]] = field(default_factory=list)
    #: Files skipped during the scan, with the reason.
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def paths(self, split: Split) -> list[str]:
        return [str(self.root / rel) for rel, _, s in self.records if s is split]

    def labels(self, split: Split) -> list[int]:
        return [idx for _, idx, s in self.records if s is split]

    def counts(self, split: Split | None = None) -> dict[str, int]:
        """Images per class, for one split or the whole dataset."""
        out = dict.fromkeys(self.class_names, 0)
        for _, idx, s in self.records:
            if split is None or s is split:
                out[self.class_names[idx]] += 1
        return out

    def summary(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "class_names": self.class_names,
            "total_images": len(self.records),
            "rejected": len(self.rejected),
            "per_split": {s.value: sum(1 for _, _, x in self.records if x is s) for s in Split},
            "per_class": self.counts(),
            "per_class_train": self.counts(Split.TRAIN),
        }

    def save(self, path: str | Path) -> Path:
        """Write the manifest to CSV so the split can be audited and reused."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["relative_path", "class_index", "class_name", "split"])
            for rel, idx, split in self.records:
                writer.writerow([rel, idx, self.class_names[idx], split.value])
        logger.info("manifest written to %s (%d images)", target, len(self.records))
        return target

    @classmethod
    def load(cls, path: str | Path, root: str | Path) -> DatasetManifest:
        """Reload a saved manifest, so evaluation reuses the training split."""
        with Path(path).open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"manifest {path} is empty")
        names: dict[int, str] = {}
        records = []
        for row in rows:
            idx = int(row["class_index"])
            names[idx] = row["class_name"]
            records.append((row["relative_path"], idx, Split(row["split"])))
        class_names = [names[i] for i in sorted(names)]
        return cls(root=Path(root), class_names=class_names, records=records)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _stable_key(relative_path: str) -> str:
    """Hash of the relative path.

    Used to order files before splitting so the split does not depend on the
    order the filesystem happens to return directory entries in. ``os.listdir``
    ordering differs between ext4, APFS and a zip extraction, which is enough to
    silently change which images land in the test set.
    """
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def _allocate(n: int, val_split: float, test_split: float) -> tuple[int, int]:
    """Split ``n`` images into (train, val) counts; train takes the remainder.

    A class with very few images must still land in val and test, or its
    per-class precision and recall are undefined and the macro average silently
    drops it. So val and test get at least one image each whenever the class has
    three or more, taking from train. Below three there is nothing meaningful to
    measure, and everything goes to train.
    """
    if n < 3:
        return n, 0

    n_val = max(1, int(round(n * val_split)))
    n_test = max(1, int(round(n * test_split)))

    # Never starve train. Shrink the larger of val/test until one image remains.
    while n_val + n_test > n - 1:
        if n_test > n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    return n - n_val - n_test, n_val


def build_manifest(
    root: str | Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    class_names: Sequence[str] | None = None,
    verify_images: bool = False,
) -> DatasetManifest:
    """Scan ``root`` and assign every image to exactly one split.

    ``root`` holds one directory per class::

        data/raw/cardboard/*.jpg
        data/raw/glass/*.jpg

    The split is stratified per class, so a class holding 3% of the data holds
    roughly 3% of each split. With small classes this matters: a uniform random
    split can leave a rare class with two test images, and accuracy on two
    images is noise.

    Args:
        root: Dataset root holding one subdirectory per class.
        train_split: Fraction assigned to training.
        val_split: Fraction assigned to validation (model selection).
        test_split: Fraction assigned to test (reported once, at the end).
        seed: Seed for the per-class shuffle.
        class_names: Force a label order. Defaults to sorted directory names.
        verify_images: Open every file with Pillow and drop unreadable ones.
            Costs a full pass over the data; worth it once after downloading.

    Returns:
        A populated :class:`DatasetManifest`.

    Raises:
        FileNotFoundError: ``root`` does not exist.
        ValueError: No class directories, or no usable images.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"dataset root {root_path} does not exist. "
            "Run `wasteclf fetch-data --help` for how to populate it."
        )

    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {total}")

    discovered = sorted(
        d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not discovered:
        raise ValueError(f"no class subdirectories under {root_path}")

    names = list(class_names) if class_names is not None else discovered
    missing = set(names) - set(discovered)
    if missing:
        raise ValueError(f"requested classes not present under {root_path}: {sorted(missing)}")

    rng = random.Random(seed)
    records: list[tuple[str, int, Split]] = []
    rejected: list[tuple[str, str]] = []

    for idx, name in enumerate(names):
        files: list[str] = []
        for entry in sorted((root_path / name).rglob("*")):
            if not entry.is_file():
                continue
            rel = entry.relative_to(root_path).as_posix()
            if not _is_image(entry):
                rejected.append((rel, f"unsupported extension {entry.suffix!r}"))
                continue
            if entry.stat().st_size == 0:
                rejected.append((rel, "empty file"))
                continue
            if verify_images and not _readable(entry):
                rejected.append((rel, "unreadable image"))
                continue
            files.append(rel)

        if not files:
            logger.warning("class %r has no usable images", name)
            continue

        # Sort by a content-independent stable hash, then shuffle with a fixed
        # seed. Filesystem order never reaches the split decision.
        files.sort(key=_stable_key)
        rng.shuffle(files)

        n_train, n_val = _allocate(len(files), val_split, test_split)

        for i, rel in enumerate(files):
            if i < n_train:
                split = Split.TRAIN
            elif i < n_train + n_val:
                split = Split.VAL
            else:
                split = Split.TEST
            records.append((rel, idx, split))

    if not records:
        raise ValueError(f"no usable images found under {root_path}")

    manifest = DatasetManifest(
        root=root_path, class_names=names, records=records, rejected=rejected
    )
    counts = manifest.summary()["per_split"]
    logger.info(
        "scanned %s: %d images across %d classes (train=%d val=%d test=%d, %d rejected)",
        root_path,
        len(records),
        len(names),
        counts["train"],
        counts["val"],
        counts["test"],
        len(rejected),
    )
    return manifest


def _readable(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 - any decode failure disqualifies the file
        return False


def class_weights(labels: Iterable[int], num_classes: int) -> dict[int, float]:
    """Inverse-frequency weights, normalised to mean 1.0.

    Computed from the training split only. Deriving them from the full dataset
    leaks test-set composition into the loss.

    A class holding 5% of the data gets a weight above 1, so the optimiser stops
    treating "never predict compost" as a cheap way to lower the loss.
    """
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1

    total = sum(counts)
    if total == 0:
        raise ValueError("cannot compute class weights from an empty label set")

    present = sum(1 for c in counts if c > 0)
    weights: dict[int, float] = {}
    for i, count in enumerate(counts):
        # Absent classes get weight 1.0; they contribute no gradient anyway.
        weights[i] = total / (present * count) if count else 1.0

    mean = sum(weights.values()) / num_classes
    return {k: v / mean for k, v in weights.items()}
