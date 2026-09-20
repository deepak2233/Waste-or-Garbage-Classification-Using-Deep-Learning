"""Command line surface.

These cover argument wiring and exit codes. The heavy paths are exercised in
test_end_to_end.py.
"""

from __future__ import annotations

import json

import pytest

from wasteclf.cli import _parse_override, build_parser, main


def test_override_parsing_infers_types():
    assert _parse_override("train.warmup.epochs=3") == ("train.warmup.epochs", 3)
    assert _parse_override("data.cache=false") == ("data.cache", False)
    assert _parse_override("model.dropout=0.25") == ("model.dropout", 0.25)
    assert _parse_override("model.weights=null") == ("model.weights", None)
    # A bare string stays a string rather than failing JSON parsing.
    assert _parse_override("run_name=my-run") == ("run_name", "my-run")
    assert _parse_override("data.image_size=[160,160]") == ("data.image_size", [160, 160])


def test_override_without_equals_is_rejected():
    with pytest.raises(Exception, match="key=value"):
        _parse_override("train.epochs")


def test_every_subcommand_is_registered():
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    assert set(action.choices) == {
        "backbones",
        "scan",
        "train",
        "evaluate",
        "predict",
        "explain",
        "export",
        "serve",
    }


def test_backbones_lists_the_registry(capsys):
    assert main(["backbones"]) == 0
    out = capsys.readouterr().out
    assert "vgg16" in out
    assert "efficientnetb0" in out


def test_scan_reports_the_split_as_json(synthetic_root, capsys):
    assert main(["scan", "--data-root", str(synthetic_root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_images"] == 98
    assert set(payload["per_split"]) == {"train", "val", "test"}
    assert len(payload["class_names"]) == 7


def test_scan_writes_a_manifest(synthetic_root, tmp_path, capsys):
    target = tmp_path / "manifest.csv"
    assert main(["scan", "--data-root", str(synthetic_root), "--out", str(target)]) == 0
    assert target.exists()
    assert "relative_path,class_index,class_name,split" in target.read_text()


def test_missing_data_root_exits_with_a_message_not_a_traceback(capsys, tmp_path):
    code = main(["scan", "--data-root", str(tmp_path / "nope")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_bad_override_exits_cleanly(synthetic_root, capsys):
    code = main(["scan", "--data-root", str(synthetic_root), "--set", "data.train_split=2.0"])
    assert code == 2
    assert "sum to 1.0" in capsys.readouterr().err


def test_evaluate_requires_an_existing_run(capsys, tmp_path):
    assert main(["evaluate", "--run", str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
