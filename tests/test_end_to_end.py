"""Full pipeline: scan, train, save, reload, predict, explain, export.

Marked ``slow`` because it trains a real, tiny model. These assert on the
artefacts a run leaves behind rather than on the functions in isolation, which
is where the awkward bugs live.

    pytest -m slow
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wasteclf.config import Config
from wasteclf.data.manifest import Split
from wasteclf.evaluation.metrics import evaluate, predict_split
from wasteclf.evaluation.plots import save_all_plots
from wasteclf.inference.predictor import Predictor
from wasteclf.utils.run import RunDirectory

pytestmark = [pytest.mark.slow, pytest.mark.needs_tf]


def test_both_stages_ran(trained):
    result, _, _ = trained
    assert result.stage_epochs["warmup"] == 1
    assert result.stage_epochs["finetune"] == 1
    assert result.history["loss"], "no loss recorded"


def test_run_directory_holds_everything_needed_to_reproduce(trained):
    result, _, _ = trained
    run = result.run
    for path in (run.config_path, run.labels_path, run.model_path, run.path / "manifest.csv"):
        assert path.exists(), f"missing {path}"
    assert (run.path / "training_summary.json").exists()


def test_saved_labels_match_the_model_output_width(trained):
    result, manifest, _ = trained
    payload = json.loads(result.run.labels_path.read_text())
    assert payload["class_names"] == manifest.class_names
    assert payload["num_classes"] == result.model.output_shape[-1]


def test_class_weights_were_applied_and_favour_the_rare_class(trained):
    result, manifest, _ = trained
    weights = result.class_weights
    assert weights is not None
    compost = manifest.class_names.index("compost")  # 4 images
    cardboard = manifest.class_names.index("cardboard")  # 20 images
    assert weights[compost] > weights[cardboard]


def test_reloaded_model_predicts_identically(trained):
    """Catches serialisation bugs in the custom preprocessing layer."""
    result, _, datasets = trained
    predictor = Predictor.from_run(result.run.path)

    batch = next(iter(datasets["test"]))[0]
    original = np.asarray(result.model(batch, training=False))
    reloaded = np.asarray(predictor.model(batch, training=False))
    assert np.allclose(original, reloaded, atol=1e-5)


def test_predictor_infers_image_size_and_labels_from_the_run(trained, synthetic_root):
    result, manifest, _ = trained
    predictor = Predictor.from_run(result.run.path)
    assert predictor.class_names == manifest.class_names
    assert predictor.image_size == (32, 32)


def test_predict_on_files_returns_calibrated_probabilities(trained, synthetic_root):
    result, manifest, _ = trained
    predictor = Predictor.from_run(result.run.path, threshold=0.99)
    paths = sorted(str(p) for p in (synthetic_root / "glass").glob("*.jpg"))[:5]

    predictions = predictor.predict(paths)
    assert len(predictions) == 5
    for prediction in predictions:
        assert prediction.label in manifest.class_names
        assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-4)
        # scores are sorted, so the first entry is the reported label
        assert next(iter(prediction.scores)) == prediction.label
        assert prediction.low_confidence == (prediction.confidence < 0.99)


def test_predict_from_bytes_matches_predict_from_path(trained, synthetic_root):
    result, _, _ = trained
    predictor = Predictor.from_run(result.run.path)
    path = sorted((synthetic_root / "metal").glob("*.jpg"))[0]

    from_path = predictor.predict(str(path))[0]
    from_bytes = predictor.predict_bytes(path.read_bytes(), name=path.name)
    assert from_path.label == from_bytes.label
    assert from_path.confidence == pytest.approx(from_bytes.confidence, abs=1e-5)


def test_evaluation_report_covers_every_class(trained, manifest_classes=None):
    result, manifest, datasets = trained
    y_true, y_prob = predict_split(result.model, datasets["test"])
    report = evaluate(y_true, y_prob, manifest.class_names, split="test")

    assert report.support == len(manifest.paths(Split.TEST))
    assert set(report.per_class) == set(manifest.class_names)
    assert len(report.confusion_matrix) == manifest.num_classes
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.expected_calibration_error <= 1.0


def test_plots_are_written_and_are_not_empty(trained):
    """Each plot lands at its own path under the run directory, with content.

    A plotting path that writes a valid but empty PNG is the kind of failure a
    smoke test waves through, so assert a floor on the file size.
    """
    result, manifest, datasets = trained
    y_true, y_prob = predict_split(result.model, datasets["test"])
    report = evaluate(y_true, y_prob, manifest.class_names)

    written = save_all_plots(report, result.run.plots_dir, result.history, stage_boundary=1)
    assert len(written) == 3
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 5_000, f"{path.name} looks blank ({path.stat().st_size} bytes)"


def test_gradcam_localises_and_matches_the_prediction(trained, synthetic_root):
    from wasteclf.explain.gradcam import GradCAM

    result, _, _ = trained
    predictor = Predictor.from_run(result.run.path)
    cam = GradCAM(predictor.model)

    image = predictor.load_image(sorted((synthetic_root / "paper").glob("*.jpg"))[0])
    heatmap, index, score = cam.heatmap(image)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6
    assert 0 <= index < len(predictor.class_names)
    # The explained class is the predicted one when no class is forced.
    assert predictor.class_names[index] == predictor.predict_array(image[None])[0].label
    assert 0.0 <= score <= 1.0


def test_gradcam_overlay_matches_the_input_resolution(trained, synthetic_root):
    from wasteclf.explain.gradcam import GradCAM

    result, _, _ = trained
    predictor = Predictor.from_run(result.run.path)
    image = predictor.load_image(sorted((synthetic_root / "trash").glob("*.jpg"))[0])

    overlaid, _, _ = GradCAM(predictor.model).explain(image)
    assert overlaid.shape == (32, 32, 3)
    assert overlaid.dtype == np.uint8


def test_savedmodel_export_loads_back(trained, tmp_path):
    from wasteclf.inference.export import export_savedmodel

    result, _, datasets = trained
    target = export_savedmodel(result.model, tmp_path / "savedmodel")
    assert target.exists()

    import tensorflow as tf

    restored = tf.saved_model.load(str(target))
    batch = next(iter(datasets["test"]))[0]
    served = np.asarray(restored.serve(batch))
    assert np.allclose(served, np.asarray(result.model(batch, training=False)), atol=1e-4)


def test_run_directory_reopens_from_disk(trained):
    result, manifest, _ = trained
    reopened = RunDirectory.open(result.run.path)
    assert reopened.read_labels() == manifest.class_names
    cfg = Config.load(reopened.config_path)
    assert cfg.model.backbone == "mobilenetv2"


def test_model_loads_in_a_fresh_process(trained):
    """Regression: the custom preprocessing layer must register on import.

    Inside one pytest process the layer is already registered, because building
    the model imported the module. A `wasteclf predict` invocation is a fresh
    interpreter that only imports the predictor, and it used to fail there with
    "Could not locate class 'BackbonePreprocessing'".
    """
    import subprocess
    import sys
    from pathlib import Path

    result, _, _ = trained
    script = (
        "from wasteclf.inference.predictor import Predictor\n"
        f"p = Predictor.from_run({str(result.run.path)!r})\n"
        "print(len(p.class_names))\n"
    )
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PATH": "/usr/bin:/bin",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().endswith("7")
