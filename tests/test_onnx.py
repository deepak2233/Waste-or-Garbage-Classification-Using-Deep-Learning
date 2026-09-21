"""ONNX export and the TensorFlow-free serving path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_tf]


def test_inference_model_drops_augmentation_without_changing_outputs():
    """Stripping augmentation must be a pure refactor of the inference graph."""
    import keras  # noqa: F401

    from wasteclf.config import AugmentConfig, ModelConfig
    from wasteclf.models.build import build_model, inference_model

    full = build_model(
        ModelConfig(backbone="mobilenetv2", weights=None), 5, (64, 64), AugmentConfig()
    )
    slim = inference_model(full)

    assert "augmentation" in [layer.name for layer in full.layers]
    assert "augmentation" not in [layer.name for layer in slim.layers]

    probe = np.random.default_rng(0).uniform(0, 255, (3, 64, 64, 3)).astype(np.float32)
    assert np.allclose(
        np.asarray(full(probe, training=False)), np.asarray(slim(probe, training=False)), atol=1e-5
    )


def test_inference_model_is_a_noop_without_augmentation():
    from wasteclf.config import ModelConfig
    from wasteclf.models.build import build_model, inference_model

    model = build_model(ModelConfig(backbone="mobilenetv2", weights=None), 3, (64, 64))
    assert inference_model(model) is model


def test_jpeg_decode_matches_pillow(tmp_path):
    """Regression: the pipeline must use the accurate IDCT.

    tf.io.decode_image defaults to INTEGER_FAST, which disagrees with Pillow by
    up to 4/255 on most pixels. The ONNX serving path decodes with Pillow, so a
    mismatch there moves the softmax by several points for the same file.
    """
    import io

    import tensorflow as tf
    from PIL import Image

    from wasteclf.data.pipeline import _decode_bytes

    rng = np.random.default_rng(0)
    source = tmp_path / "probe.jpg"
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(source, quality=90)
    raw = source.read_bytes()

    from_pipeline = _decode_bytes(tf.constant(raw), (64, 64)).numpy()
    from_pillow = np.asarray(
        Image.open(io.BytesIO(raw)).convert("RGB").resize((64, 64), Image.BILINEAR),
        dtype=np.float32,
    )
    assert np.array_equal(from_pipeline, from_pillow)


@pytest.fixture(scope="module")
def onnx_dir(trained, tmp_path_factory):
    """Export the shared trained model to ONNX once."""
    pytest.importorskip("onnx", reason="install wasteclf[onnx] to run the ONNX tests")
    import json

    from wasteclf.inference.export import export_onnx

    result, manifest, _ = trained
    out = tmp_path_factory.mktemp("onnx")
    export_onnx(result.model, out / "model.onnx", image_size=(32, 32))
    (out / "labels.json").write_text(
        json.dumps({"class_names": manifest.class_names, "image_size": [32, 32]}), encoding="utf-8"
    )
    return out


def test_exported_graph_loads_in_onnxruntime(onnx_dir):
    """A graph containing augmentation ops loads as invalid, so this is the check."""
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1:3] == [32, 32]


def test_onnx_matches_keras_on_identical_arrays(onnx_dir, trained):
    pytest.importorskip("onnxruntime")
    from wasteclf.inference.onnx_predictor import OnnxPredictor

    result, _, _ = trained
    predictor = OnnxPredictor.from_dir(onnx_dir)

    probe = np.random.default_rng(1).uniform(0, 255, (4, 32, 32, 3)).astype(np.float32)
    keras_out = np.asarray(result.model(probe, training=False))
    onnx_out = np.stack(
        [[p.scores[c] for c in predictor.class_names] for p in predictor.predict_array(probe)]
    )
    assert np.allclose(keras_out, onnx_out, atol=1e-4)


def test_onnx_predictor_agrees_with_keras_on_real_files(onnx_dir, trained, synthetic_root):
    """End to end, including the decode, which is where the two paths could drift."""
    pytest.importorskip("onnxruntime")
    from wasteclf.inference.onnx_predictor import OnnxPredictor
    from wasteclf.inference.predictor import Predictor

    result, _, _ = trained
    files = [str(p) for p in sorted(synthetic_root.rglob("*.jpg"))[:8]]

    keras_preds = Predictor.from_run(result.run.path).predict(files)
    onnx_preds = OnnxPredictor.from_dir(onnx_dir).predict(files)

    for k, o in zip(keras_preds, onnx_preds):
        assert k.label == o.label
        assert k.confidence == pytest.approx(o.confidence, abs=1e-3)


def test_labels_and_model_width_must_agree(onnx_dir):
    pytest.importorskip("onnxruntime")
    from wasteclf.inference.onnx_predictor import OnnxPredictor

    with pytest.raises(ValueError, match="classes but labels"):
        OnnxPredictor(onnx_dir / "model.onnx", class_names=["only", "two"])


def test_missing_model_names_the_export_command(tmp_path):
    from wasteclf.inference.onnx_predictor import OnnxPredictor

    with pytest.raises(FileNotFoundError, match="wasteclf export"):
        OnnxPredictor.from_dir(tmp_path)


def test_onnx_predictor_imports_without_tensorflow():
    """The serverless function depends on this: importing it must not pull in TF."""
    script = (
        "import sys\n"
        "from wasteclf.inference.onnx_predictor import OnnxPredictor\n"
        "assert 'tensorflow' not in sys.modules, 'tensorflow was imported'\n"
        "assert 'keras' not in sys.modules, 'keras was imported'\n"
        "print('clean')\n"
    )
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PATH": "/usr/bin:/bin",
    }
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert "clean" in done.stdout


def test_scene_pipeline_runs_on_the_onnx_predictor(onnx_dir, synthetic_root):
    """The two-stage pipeline must work on the runtime that fits in a function.

    ScenePipeline only needs class_names, image_size and predict_array, so both
    predictors satisfy it. This is what lets scene analysis deploy serverless.
    """
    pytest.importorskip("onnxruntime")
    from PIL import Image

    from wasteclf.inference.onnx_predictor import OnnxPredictor
    from wasteclf.scene import ContentSegmenter, ScenePipeline

    # A scene built from real dataset crops on a flat background.
    rng = np.random.default_rng(0)
    scene = np.full((96, 128, 3), 110.0, dtype=np.float32)
    for i, path in enumerate(sorted(synthetic_root.rglob("*.jpg"))[:4]):
        tile = np.asarray(Image.open(path).convert("RGB").resize((32, 32)), dtype=np.float32)
        y, x = (i // 2) * 40, (i % 2) * 60
        scene[y : y + 32, x : x + 32] = tile
    scene += rng.normal(0, 2, scene.shape)

    predictor = OnnxPredictor.from_dir(onnx_dir)
    result = ScenePipeline(
        predictor, ContentSegmenter(3, 4, min_activity=0.02), min_confidence=0.0
    ).analyse(np.clip(scene, 0, 255))

    assert result.detections
    assert set(result.class_names) == set(predictor.class_names)
    assert sum(result.composition.values()) == pytest.approx(1.0, abs=1e-6)
