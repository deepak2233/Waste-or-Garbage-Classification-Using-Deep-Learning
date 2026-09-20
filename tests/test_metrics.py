"""Evaluation metric correctness."""

from __future__ import annotations

import numpy as np
import pytest

from wasteclf.evaluation.metrics import (
    confusion_matrix,
    evaluate,
    expected_calibration_error,
)


def _one_hot_probs(preds: list[int], num_classes: int, confidence: float = 0.9) -> np.ndarray:
    probs = np.full((len(preds), num_classes), (1 - confidence) / (num_classes - 1))
    for i, p in enumerate(preds):
        probs[i, p] = confidence
    return probs


def test_perfect_predictions_score_one():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    report = evaluate(y_true, _one_hot_probs(list(y_true), 3), ["a", "b", "c"])
    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.balanced_accuracy == pytest.approx(1.0)


def test_accuracy_hides_an_ignored_minority_class():
    """Accuracy alone cannot see a class the model never predicts."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = [0] * 100  # class 1 never predicted
    report = evaluate(y_true, _one_hot_probs(y_pred, 2), ["common", "rare"])

    assert report.accuracy == pytest.approx(0.90)
    assert report.per_class["rare"]["f1"] == 0.0
    assert report.per_class["rare"]["recall"] == 0.0
    # Macro F1 and balanced accuracy both expose it.
    assert report.macro_f1 < 0.5
    assert report.balanced_accuracy == pytest.approx(0.5)


def test_confusion_matrix_rows_are_true_classes():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    matrix = confusion_matrix(y_true, y_pred, 2)
    assert matrix.tolist() == [[1, 1], [0, 1]]
    assert matrix.sum() == 3


def test_most_confused_ranks_by_count():
    y_true = np.array([0] * 10 + [1] * 10)
    y_pred = [1] * 10 + [1] * 10  # every class-0 image misread as class 1
    report = evaluate(y_true, _one_hot_probs(y_pred, 2), ["a", "b"])
    assert report.most_confused[0]["true"] == "a"
    assert report.most_confused[0]["predicted"] == "b"
    assert report.most_confused[0]["count"] == 10
    assert report.most_confused[0]["rate"] == pytest.approx(1.0)


def test_top2_accuracy_is_at_least_top1():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 50)
    probs = rng.dirichlet(np.ones(5), 50)
    report = evaluate(y_true, probs, list("abcde"))
    assert report.top2_accuracy >= report.accuracy


def test_absent_classes_do_not_drag_down_the_macro_average():
    """A class with no test examples is excluded rather than scored as zero."""
    y_true = np.array([0, 0, 1, 1])
    report = evaluate(y_true, _one_hot_probs([0, 0, 1, 1], 3), ["a", "b", "never_seen"])
    assert report.per_class["never_seen"]["support"] == 0
    assert report.macro_f1 == pytest.approx(1.0)


def test_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    # 100% confident and 100% correct.
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y_true, probs) == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_catches_overconfidence():
    # 95% confident, 50% correct.
    y_true = np.array([0, 1, 0, 1])
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]])
    assert expected_calibration_error(y_true, probs) == pytest.approx(0.45, abs=0.01)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        evaluate(np.array([0, 1]), np.zeros((2, 5)), ["a", "b"])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="rows"):
        evaluate(np.array([0, 1, 2]), np.zeros((2, 2)), ["a", "b"])


def test_report_serialises_and_formats():
    y_true = np.array([0, 1])
    report = evaluate(y_true, _one_hot_probs([0, 1], 2), ["a", "b"])
    payload = report.to_dict()
    assert payload["accuracy"] == 1.0
    assert "confusion_matrix" in payload
    assert "precision" in report.format_table()
