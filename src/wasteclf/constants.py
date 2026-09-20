"""Values shared across the package.

The class list is intentionally *not* hardcoded into the model code. It is
derived from the dataset directory at scan time and persisted alongside every
trained model, so a checkpoint always carries the label order it was fit with.
``DEFAULT_CLASSES`` only documents the taxonomy this project was built around.
"""

from __future__ import annotations

from typing import Final

DEFAULT_CLASSES: Final[tuple[str, ...]] = (
    "cardboard",
    "compost",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

#: Extensions accepted when scanning a dataset directory. Anything else is
#: skipped and counted as a rejection so the scan report stays honest.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)

#: File written into every run directory recording how the model was trained.
RUN_CONFIG_FILENAME: Final[str] = "config.yaml"
LABELS_FILENAME: Final[str] = "labels.json"
METRICS_FILENAME: Final[str] = "metrics.json"
HISTORY_FILENAME: Final[str] = "history.csv"
MANIFEST_FILENAME: Final[str] = "manifest.csv"
