"""Run directories.

Every training run gets its own timestamped directory holding the config, the
label order, the data manifest, the checkpoint, the metrics and the plots. This
is what makes a result reproducible six months later: you no longer have to
remember which notebook cell produced which number.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wasteclf.constants import (
    HISTORY_FILENAME,
    LABELS_FILENAME,
    METRICS_FILENAME,
    RUN_CONFIG_FILENAME,
)
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunDirectory:
    """A single training run on disk."""

    path: Path

    @classmethod
    def create(
        cls, output_dir: str | Path, name: str | None = None, backbone: str = "model"
    ) -> RunDirectory:
        """Create a new run directory.

        If ``name`` is omitted the directory is ``<backbone>-<UTC timestamp>``,
        which sorts chronologically and never collides.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        folder = name or f"{backbone}-{stamp}"
        path = Path(output_dir) / folder
        path.mkdir(parents=True, exist_ok=True)
        (path / "plots").mkdir(exist_ok=True)
        logger.info("run directory: %s", path)
        return cls(path=path)

    # Conventional paths -----------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.path / RUN_CONFIG_FILENAME

    @property
    def labels_path(self) -> Path:
        return self.path / LABELS_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.path / METRICS_FILENAME

    @property
    def history_path(self) -> Path:
        return self.path / HISTORY_FILENAME

    @property
    def plots_dir(self) -> Path:
        return self.path / "plots"

    @property
    def model_path(self) -> Path:
        """Keras v3 archive. Self-contained: architecture plus weights."""
        return self.path / "model.keras"

    @property
    def checkpoint_path(self) -> Path:
        """Best-epoch weights, written during training."""
        return self.path / "checkpoint.weights.h5"

    # Writers ----------------------------------------------------------------

    def write_labels(self, class_names: Sequence[str]) -> Path:
        """Persist the label order.

        Index order is the contract between training and inference. Reading it
        back from disk removes the class ``os.listdir`` ordering hazard, where
        the same code gives different label maps on different filesystems.
        """
        payload = {
            "class_names": list(class_names),
            "num_classes": len(class_names),
            "index_to_label": dict(enumerate(class_names)),
        }
        self.labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.labels_path

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        self.metrics_path.write_text(
            json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
        )
        return self.metrics_path

    def write_json(self, filename: str, payload: Any) -> Path:
        target = self.path / filename
        target.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return target

    # Readers ----------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> RunDirectory:
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"run directory not found: {p}")
        return cls(path=p)

    def read_labels(self) -> list[str]:
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"{self.labels_path} is missing; the run is incomplete or was not "
                "produced by this version of wasteclf"
            )
        payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
        return list(payload["class_names"])

    def read_metrics(self) -> dict[str, Any]:
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    """Make NumPy scalars and arrays JSON-serialisable."""
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
