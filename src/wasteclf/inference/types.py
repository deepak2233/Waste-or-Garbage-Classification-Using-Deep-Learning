"""Result types shared by both predictors.

Deliberately free of TensorFlow and onnxruntime imports, so the serverless
function can use these without pulling in either.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediction:
    """One image's result."""

    path: str
    label: str
    confidence: float
    #: All class probabilities, highest first.
    scores: dict[str, float]
    #: ``True`` when confidence fell below the caller's threshold. The caller
    #: decides what to do; nothing here silently rewrites the label.
    low_confidence: bool = False

    @property
    def runner_up(self) -> tuple[str, float]:
        items = list(self.scores.items())
        return items[1] if len(items) > 1 else items[0]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "low_confidence": self.low_confidence,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }
