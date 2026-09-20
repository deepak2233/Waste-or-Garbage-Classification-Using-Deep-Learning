"""Determinism controls.

Full bit-for-bit reproducibility on GPU costs real throughput, so this sets the
seeds every run needs and leaves op-level determinism behind a flag.
"""

from __future__ import annotations

import os
import random

from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 42, deterministic_ops: bool = False) -> int:
    """Seed Python, NumPy and TensorFlow.

    Args:
        seed: The seed value.
        deterministic_ops: Also enable TensorFlow op determinism. This makes GPU
            training reproducible and measurably slower, so it is off by default.

    Returns:
        The seed, so callers can log or persist it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        logger.warning("numpy not importable; skipping numpy seed")

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic_ops:
            tf.config.experimental.enable_op_determinism()
            logger.info("TensorFlow op determinism enabled (slower)")
    except ImportError:  # pragma: no cover
        logger.warning("tensorflow not importable; skipping tf seed")

    logger.debug("seeded everything with %d", seed)
    return seed
