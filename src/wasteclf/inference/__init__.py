"""Loading a trained run and making predictions."""

from wasteclf.inference.export import export_savedmodel, export_tflite
from wasteclf.inference.predictor import Prediction, Predictor

__all__ = ["Predictor", "Prediction", "export_savedmodel", "export_tflite"]
