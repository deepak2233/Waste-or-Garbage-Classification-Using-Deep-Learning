"""Two-stage transfer learning.

Stage one trains the classifier head with the backbone frozen. Stage two
unfreezes the top of the backbone and continues at a much lower learning rate.

Running a single stage with the backbone unfrozen from epoch 0 destroys the
pretrained features: the head starts at random, its first gradients are large,
and they propagate straight into convolution filters that took an ImageNet run
to learn. Warming up the head first keeps those gradients small by the time the
backbone is unfrozen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import keras
import tensorflow as tf

from wasteclf.config import Config
from wasteclf.data.manifest import DatasetManifest, Split, class_weights
from wasteclf.models.build import build_model, set_finetune_trainable
from wasteclf.training.callbacks import build_callbacks
from wasteclf.utils.logging import get_logger
from wasteclf.utils.run import RunDirectory

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """What a training run produced."""

    model: keras.Model
    run: RunDirectory
    history: dict[str, list[float]] = field(default_factory=dict)
    stage_epochs: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    class_weights: dict[int, float] | None = None


def _compile(model: keras.Model, learning_rate: float, label_smoothing: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )


def _merge_history(target: dict[str, list[float]], source: dict[str, list]) -> None:
    for key, values in source.items():
        target.setdefault(key, []).extend(float(v) for v in values)


def train(
    cfg: Config,
    manifest: DatasetManifest,
    datasets: dict[str, tf.data.Dataset],
    run: RunDirectory | None = None,
) -> TrainingResult:
    """Run both training stages and persist the artefacts.

    Args:
        cfg: Validated configuration.
        manifest: Dataset manifest, used for the label order and class weights.
        datasets: Mapping with at least ``train``; ``val`` is strongly advised.
        run: Existing run directory, or ``None`` to create one.

    Returns:
        A :class:`TrainingResult` whose ``model`` holds the best weights seen.
    """
    started = time.time()
    run = run or RunDirectory.create(cfg.output_dir, cfg.run_name, cfg.model.backbone)

    cfg.save(run.config_path)
    run.write_labels(manifest.class_names)
    manifest.save(run.path / "manifest.csv")
    run.write_json("dataset_summary.json", manifest.summary())

    if cfg.train.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("mixed precision enabled (float16 compute, float32 softmax)")

    train_ds = datasets.get("train")
    if train_ds is None:
        raise ValueError("datasets must contain a 'train' split")
    val_ds = datasets.get("val")
    if val_ds is None:
        logger.warning(
            "no validation split: early stopping and checkpointing are disabled, "
            "and the final model will be the last epoch rather than the best"
        )

    model = build_model(
        model_cfg=cfg.model,
        num_classes=manifest.num_classes,
        image_size=cfg.data.image_size,
        augment_cfg=cfg.augment,
        seed=cfg.seed,
    )

    weights = None
    if cfg.train.class_weights:
        weights = class_weights(manifest.labels(Split.TRAIN), manifest.num_classes)
        pretty = {manifest.class_names[k]: round(v, 3) for k, v in weights.items()}
        logger.info("class weights: %s", pretty)

    history: dict[str, list[float]] = {}
    stage_epochs: dict[str, int] = {}

    # Stage one: head only -------------------------------------------------
    warmup = cfg.train.warmup
    if warmup.epochs > 0 and cfg.model.weights is None and warmup.unfreeze_layers == 0:
        # A randomly initialised backbone held in inference mode emits a constant
        # feature vector (on MobileNetV2 it is exactly zero, because untrained
        # BatchNorm statistics push every ReLU6 to its floor). The head then has
        # nothing to separate and the loss parks at ln(num_classes).
        logger.warning(
            "weights=None with a frozen backbone: stage 1 cannot learn anything, because "
            "an untrained frozen backbone emits constant features. Set model.weights=imagenet, "
            "or give train.warmup.unfreeze_layers a non-zero value."
        )

    if warmup.epochs > 0:
        logger.info(
            "stage 1/2 warmup: %d epochs at lr=%g, backbone frozen",
            warmup.epochs,
            warmup.learning_rate,
        )
        set_finetune_trainable(model, warmup.unfreeze_layers, warmup.freeze_batchnorm)
        _compile(model, warmup.learning_rate, cfg.model.label_smoothing)
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup.epochs,
            class_weight=weights,
            # tf.data already shuffles the training split; letting Keras
            # shuffle as well is a no-op it warns about.
            shuffle=False,
            callbacks=build_callbacks(
                cfg.train,
                run.checkpoint_path,
                run.history_path,
                val_ds,
                manifest.num_classes,
                stage="warmup",
            )
            if val_ds is not None
            else [],
            verbose=2,
        )
        _merge_history(history, h.history)
        stage_epochs["warmup"] = len(h.history.get("loss", []))

    # Stage two: fine-tune the top of the backbone --------------------------
    finetune = cfg.train.finetune
    if finetune.epochs > 0 and finetune.unfreeze_layers != 0:
        logger.info(
            "stage 2/2 fine-tune: %d epochs at lr=%g, unfreezing %s backbone layers",
            finetune.epochs,
            finetune.learning_rate,
            "all" if finetune.unfreeze_layers < 0 else finetune.unfreeze_layers,
        )
        set_finetune_trainable(model, finetune.unfreeze_layers, finetune.freeze_batchnorm)
        # Recompiling after changing trainability is mandatory: Keras captures
        # the trainable-variable list at compile time, so skipping this trains
        # the head only and reports fine-tuning numbers that never happened.
        _compile(model, finetune.learning_rate, cfg.model.label_smoothing)
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune.epochs,
            class_weight=weights,
            shuffle=False,
            callbacks=build_callbacks(
                cfg.train,
                run.checkpoint_path,
                run.history_path,
                val_ds,
                manifest.num_classes,
                stage="finetune",
            )
            if val_ds is not None
            else [],
            verbose=2,
        )
        _merge_history(history, h.history)
        stage_epochs["finetune"] = len(h.history.get("loss", []))
    elif finetune.epochs > 0:
        logger.info("fine-tune stage skipped: unfreeze_layers is 0")

    model.save(run.model_path)
    logger.info("saved model to %s", run.model_path)

    elapsed = time.time() - started
    run.write_json(
        "training_summary.json",
        {
            "backbone": cfg.model.backbone,
            "stage_epochs": stage_epochs,
            "total_epochs": sum(stage_epochs.values()),
            "seconds": round(elapsed, 1),
            "class_weights": weights,
            "final": {k: v[-1] for k, v in history.items() if v},
        },
    )
    logger.info("training finished in %.1fs (%d epochs)", elapsed, sum(stage_epochs.values()))

    return TrainingResult(
        model=model,
        run=run,
        history=history,
        stage_epochs=stage_epochs,
        seconds=elapsed,
        class_weights=weights,
    )
