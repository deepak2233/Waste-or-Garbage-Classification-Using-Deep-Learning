"""Logging setup.

One configuration point, called once from the CLI. Library modules call
``get_logger(__name__)`` and never touch handlers, so importing wasteclf from a
notebook does not hijack the root logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Attach a stderr handler, and optionally a file handler, to ``wasteclf``.

    Safe to call more than once; later calls replace the handlers rather than
    stacking them, so re-running a cell does not duplicate every log line.
    """
    global _CONFIGURED
    logger = logging.getLogger("wasteclf")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``wasteclf`` namespace."""
    if not name.startswith("wasteclf"):
        name = f"wasteclf.{name}"
    return logging.getLogger(name)
