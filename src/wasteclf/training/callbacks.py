"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from wasteclf.config import TrainConfig
from wasteclf.utils.logging import get_logger

logger = get_logger(__name__)


class MacroF1(keras.callbacks.Callback):
    """Compute macro-averaged F1 on the validation set after each epoch.

    Accuracy is a poor monitor on this dataset. The class counts are uneven, so
    a model that never predicts the two smallest classes can still post a
    respectable accuracy. Macro F1 weights every class equally and drops as soon
    as one is being ignored.

    Writes ``val_macro_f1`` into the epoch logs, which makes it usable as the
    ``monitor`` for EarlyStopping and ModelCheckpoint.
    """

    def __init__(self, val_ds: tf.data.Dataset, num_classes: int, verbose: bool = True):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        for batch_x, batch_y in self.val_ds:
            probs = self.model(batch_x, training=False)
            y_pred.append(np.argmax(np.asarray(probs), axis=1))
            y_true.append(np.argmax(np.asarray(batch_y), axis=1))

        if not y_true:
            return

        truth = np.concatenate(y_true)
        pred = np.concatenate(y_pred)
        score = _macro_f1(truth, pred, self.num_classes)
        logs["val_macro_f1"] = score
        if self.verbose:
            logger.info("epoch %d: val_macro_f1=%.4f", epoch + 1, score)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro F1 without a scikit-learn round trip.

    Classes absent from ``y_true`` are excluded from the average rather than
    scored as 0, which is what ``sklearn`` does with ``zero_division=0`` and
    would otherwise drag the metric down for a reason the model cannot fix.
    """
    scores = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn)
        scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def build_callbacks(
    cfg: TrainConfig,
    checkpoint_path: str | Path,
    history_path: str | Path | None = None,
    val_ds: tf.data.Dataset | None = None,
    num_classes: int | None = None,
    stage: str = "warmup",
) -> list[keras.callbacks.Callback]:
    """Assemble the callback list for one training stage.

    ``restore_best_weights=True`` on EarlyStopping means the model in memory at
    the end of the stage is the best one, not the last one. Without it, stage
    two would start fine-tuning from an overfitted stage-one head.
    """
    monitor = cfg.monitor
    mode = "max" if monitor.endswith(("accuracy", "f1")) else "min"

    callbacks: list[keras.callbacks.Callback] = []

    if monitor == "val_macro_f1":
        if val_ds is None or num_classes is None:
            raise ValueError("monitoring val_macro_f1 requires val_ds and num_classes")
        # Must run before the callbacks that read the metric from the logs.
        callbacks.append(MacroF1(val_ds, num_classes))

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
    )
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        )
    )
    callbacks.append(keras.callbacks.TerminateOnNaN())

    if history_path is not None:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # append=True so stage two extends stage one's log instead of erasing it.
        callbacks.append(keras.callbacks.CSVLogger(str(path), append=(stage != "warmup")))

    return callbacks
