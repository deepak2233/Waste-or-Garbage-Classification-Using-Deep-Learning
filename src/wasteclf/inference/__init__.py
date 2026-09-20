"""Loading a trained run and making predictions.

Imports are deferred. ``OnnxPredictor`` must be importable in an environment
that has onnxruntime but not TensorFlow, and a normal ``from .predictor import
Predictor`` at module scope would drag Keras in and break that.
"""

__all__ = [
    "Predictor",
    "Prediction",
    "OnnxPredictor",
    "export_savedmodel",
    "export_tflite",
    "export_onnx",
]

_TF_BACKED = {
    "Predictor": ("wasteclf.inference.predictor", "Predictor"),
    "Prediction": ("wasteclf.inference.predictor", "Prediction"),
    "export_savedmodel": ("wasteclf.inference.export", "export_savedmodel"),
    "export_tflite": ("wasteclf.inference.export", "export_tflite"),
    "export_onnx": ("wasteclf.inference.export", "export_onnx"),
    "OnnxPredictor": ("wasteclf.inference.onnx_predictor", "OnnxPredictor"),
}


def __getattr__(name: str):
    if name in _TF_BACKED:
        module_name, attr = _TF_BACKED[name]
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
