"""Evaluation metrics and diagnostic plots."""

from wasteclf.evaluation.metrics import EvaluationReport, evaluate, predict_split
from wasteclf.evaluation.plots import (
    plot_confusion_matrix,
    plot_history,
    plot_per_class_f1,
    save_all_plots,
)

__all__ = [
    "EvaluationReport",
    "evaluate",
    "predict_split",
    "plot_confusion_matrix",
    "plot_history",
    "plot_per_class_f1",
    "save_all_plots",
]
