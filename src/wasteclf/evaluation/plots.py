"""Diagnostic plots.

Uses the Agg backend so plots render on a headless box (CI, a container, a
remote GPU node) without a display.

Each plot takes an explicit destination path so two runs never collide, and
figures are closed after saving so a long session does not accumulate them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wasteclf.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_DPI = 140


def plot_history(
    history: Mapping[str, Sequence[float]], path: str | Path, stage_boundary: int | None = None
) -> Path:
    """Loss and accuracy against epoch.

    ``stage_boundary`` draws a vertical line where fine-tuning began, which
    makes it obvious whether stage two helped or simply overfitted.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.get("loss", [])) + 1)

    axes[0].plot(epochs, history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("accuracy", []), label="train")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="validation")
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], label="val macro F1", linestyle="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if stage_boundary:
        for ax in axes:
            ax.axvline(stage_boundary + 0.5, color="crimson", linestyle=":", linewidth=1.5)
            ax.annotate(
                "fine-tune",
                xy=(stage_boundary + 0.5, ax.get_ylim()[1]),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color="crimson",
            )

    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", target)
    return target


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: str | Path,
    normalise: bool = True,
) -> Path:
    """Confusion matrix heatmap.

    Row-normalised by default. With imbalanced classes the raw counts are
    dominated by the largest class and the small ones are unreadable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(matrix, dtype=np.float64)
    counts = data.copy()
    if normalise:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    size = max(6.0, 0.9 * len(class_names) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalise else ""))

    threshold = data.max() / 2 if data.max() else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            label = (
                f"{data[i, j]:.2f}\n({int(counts[i, j])})" if normalise else f"{int(counts[i, j])}"
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] > threshold else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", target)
    return target


def plot_per_class_f1(per_class: Mapping[str, Mapping[str, float]], path: str | Path) -> Path:
    """Per-class F1 with support annotated.

    The plot that answers "is the headline accuracy carried by two big classes?".
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    names = list(per_class)
    scores = [float(per_class[n]["f1"]) for n in names]
    supports = [int(per_class[n]["support"]) for n in names]
    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]
    supports = [supports[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(names) + 2)))
    bars = ax.barh(names, scores, color="#4C78A8")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1")
    ax.set_title("Per-class F1")
    ax.grid(axis="x", alpha=0.3)
    for bar, score, support in zip(bars, scores, supports):
        ax.text(
            min(score + 0.02, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}  (n={support})",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    logger.info("wrote %s", target)
    return target


def save_all_plots(
    report,
    plots_dir: str | Path,
    history: Mapping[str, Sequence[float]] | None = None,
    stage_boundary: int | None = None,
) -> list[Path]:
    """Write every diagnostic plot for a run."""
    directory = Path(plots_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        plot_confusion_matrix(
            report.confusion_matrix, report.class_names, directory / "confusion_matrix.png"
        ),
        plot_per_class_f1(report.per_class, directory / "per_class_f1.png"),
    ]
    if history:
        written.append(plot_history(history, directory / "training_history.png", stage_boundary))
    return written
