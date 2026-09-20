"""Evaluation metrics.

A single accuracy figure hides most of what matters on an imbalanced taxonomy.
A model can post 72% accuracy while scoring 0.0 F1 on compost, and the headline
number looks respectable either way.

So this reports per-class precision, recall and F1, the confusion matrix, the
ranked confusion pairs, and a calibration error. Enough to tell whether the
model works or whether two big classes are carrying it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationReport:
    """Everything computed from one set of predictions."""

    split: str
    class_names: list[str]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    top2_accuracy: float
    mean_confidence: float
    expected_calibration_error: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    most_confused: list[dict[str, Any]] = field(default_factory=list)
    support: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "support": self.support,
            "class_names": self.class_names,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "top2_accuracy": round(self.top2_accuracy, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "most_confused": self.most_confused,
        }

    def format_table(self) -> str:
        """Render the per-class table for the terminal."""
        header = f"{'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"
        lines = [header, "-" * len(header)]
        for name in self.class_names:
            row = self.per_class.get(name)
            if row is None:
                continue
            lines.append(
                f"{name:<14}{row['precision']:>10.3f}{row['recall']:>9.3f}"
                f"{row['f1']:>8.3f}{int(row['support']):>9d}"
            )
        lines.append("-" * len(header))
        lines.append(f"{'macro avg':<14}{'':>10}{'':>9}{self.macro_f1:>8.3f}{self.support:>9d}")
        lines.append(f"{'accuracy':<14}{'':>10}{'':>9}{self.accuracy:>8.3f}{self.support:>9d}")
        return "\n".join(lines)


def predict_split(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a dataset and return ``(y_true, probabilities)``.

    Labels are read back from the same dataset rather than from the manifest, so
    the pairing holds even when ``ignore_errors()`` has silently dropped an
    undecodable image from the batch stream.
    """
    truths: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for batch_x, batch_y in dataset:
        probs.append(np.asarray(model(batch_x, training=False), dtype=np.float64))
        truths.append(np.asarray(batch_y))

    if not truths:
        raise ValueError("dataset yielded no batches")

    y_true = np.argmax(np.concatenate(truths), axis=1)
    y_prob = np.concatenate(probs)
    return y_true, y_prob


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins.

    A model with 0.95 mean confidence and 0.70 accuracy has an ECE near 0.25 and
    should not be wired to an automatic sorting decision without a threshold.
    Softmax outputs from a fine-tuned network are usually overconfident, so this
    is worth reading before anyone quotes the accuracy to a stakeholder.
    """
    confidence = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    split: str = "test",
    top_confusions: int = 5,
) -> EvaluationReport:
    """Turn raw predictions into a full report.

    Args:
        y_true: Integer labels, shape ``(n,)``.
        y_prob: Class probabilities, shape ``(n, num_classes)``.
        class_names: Label order matching the probability columns.
        split: Name recorded in the report.
        top_confusions: How many off-diagonal confusion pairs to list.

    Returns:
        A populated :class:`EvaluationReport`.
    """
    names = list(class_names)
    num_classes = len(names)
    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        raise ValueError(
            f"y_prob has shape {y_prob.shape}, expected (n, {num_classes}) to match class_names"
        )
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true has {len(y_true)} rows, y_prob has {len(y_prob)}")

    y_pred = y_prob.argmax(axis=1)
    support = len(y_true)
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    weights: list[int] = []

    for c, name in enumerate(names):
        tp = int(matrix[c, c])
        fn = int(matrix[c, :].sum() - tp)
        fp = int(matrix[:, c].sum() - tp)
        cls_support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / cls_support if cls_support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": cls_support,
        }
        # Classes with no examples in this split are excluded from the averages.
        if cls_support:
            f1s.append(f1)
            recalls.append(recall)
            weights.append(cls_support)

    accuracy = float((y_pred == y_true).mean())
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if f1s else 0.0

    if num_classes >= 2:
        top2 = np.argsort(-y_prob, axis=1)[:, :2]
        top2_accuracy = float(np.mean([t in row for t, row in zip(y_true, top2)]))
    else:
        top2_accuracy = accuracy

    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and matrix[i, j] > 0:
                confusions.append(
                    {
                        "true": names[i],
                        "predicted": names[j],
                        "count": int(matrix[i, j]),
                        "rate": round(float(matrix[i, j] / max(1, matrix[i].sum())), 4),
                    }
                )
    confusions.sort(key=lambda d: d["count"], reverse=True)

    return EvaluationReport(
        split=split,
        class_names=names,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        top2_accuracy=top2_accuracy,
        mean_confidence=float(y_prob.max(axis=1).mean()),
        expected_calibration_error=expected_calibration_error(y_true, y_prob),
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        most_confused=confusions[:top_confusions],
        support=support,
    )
