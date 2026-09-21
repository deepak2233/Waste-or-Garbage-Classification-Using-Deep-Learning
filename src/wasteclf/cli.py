"""Command line interface.

``wasteclf <command>``. Every command is a thin wrapper over the library, so
anything reachable from the terminal is also reachable from a notebook or a
script without shelling out.

Uses argparse rather than click or typer to keep the runtime dependency list to
things already needed for training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from wasteclf import __version__
from wasteclf.config import Config, ConfigError
from wasteclf.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# Shared argument helpers ----------------------------------------------------


def _parse_override(item: str) -> tuple[str, object]:
    """Parse ``--set key.path=value`` into a typed pair.

    Values are parsed as JSON where possible, so ``epochs=3`` becomes an int,
    ``cache=false`` a bool and ``run_name=demo`` a string.
    """
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {item!r}")
    key, _, raw = item.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def _load_config(args: argparse.Namespace) -> Config:
    overrides = dict(getattr(args, "set", None) or [])
    return Config.load(args.config, overrides)


# Commands -------------------------------------------------------------------


def cmd_backbones(args: argparse.Namespace) -> int:  # noqa: ARG001 - uniform dispatch signature
    from wasteclf.models.backbones import BACKBONES, available_backbones

    print(f"{'backbone':<18}{'params':>9}  notes")
    print("-" * 100)
    for name in available_backbones():
        spec = BACKBONES[name]
        print(f"{name:<18}{spec.params_millions:>7.1f}M  {spec.notes}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from wasteclf.data.manifest import build_manifest

    cfg = _load_config(args)
    root = args.data_root or cfg.data.root
    manifest = build_manifest(
        root=root,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.data.split_seed,
        verify_images=args.verify or cfg.data.verify_images,
    )
    summary = manifest.summary()
    print(json.dumps(summary, indent=2))

    if manifest.rejected:
        print(f"\n{len(manifest.rejected)} file(s) rejected:", file=sys.stderr)
        for path, reason in manifest.rejected[:20]:
            print(f"  {path}: {reason}", file=sys.stderr)
        if len(manifest.rejected) > 20:
            print(f"  ... and {len(manifest.rejected) - 20} more", file=sys.stderr)

    if args.out:
        manifest.save(args.out)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from wasteclf.data.manifest import build_manifest
    from wasteclf.data.pipeline import build_datasets
    from wasteclf.evaluation.metrics import evaluate, predict_split
    from wasteclf.evaluation.plots import save_all_plots
    from wasteclf.training.trainer import train
    from wasteclf.utils.run import RunDirectory
    from wasteclf.utils.seed import seed_everything

    cfg = _load_config(args)
    if args.data_root:
        cfg.data.root = args.data_root
    cfg.validate()

    run = RunDirectory.create(cfg.output_dir, cfg.run_name, cfg.model.backbone)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO, run.path / "train.log")
    seed_everything(cfg.seed, deterministic_ops=args.deterministic)

    manifest = build_manifest(
        root=cfg.data.root,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.data.split_seed,
        verify_images=cfg.data.verify_images,
    )
    if manifest.rejected:
        logger.warning(
            "%d unreadable file(s) excluded; run `wasteclf scan --verify` to list them",
            len(manifest.rejected),
        )
    datasets = build_datasets(manifest, cfg.data, seed=cfg.seed)
    result = train(cfg, manifest, datasets, run=run)

    if "test" in datasets:
        logger.info("evaluating on the held-out test split")
        y_true, y_prob = predict_split(result.model, datasets["test"])
        report = evaluate(y_true, y_prob, manifest.class_names, split="test")
        run.write_metrics(report.to_dict())
        save_all_plots(
            report,
            run.plots_dir,
            result.history,
            stage_boundary=result.stage_epochs.get("warmup"),
        )
        print("\n" + report.format_table())
        print(
            f"\nmacro F1 {report.macro_f1:.3f} | balanced accuracy {report.balanced_accuracy:.3f}"
        )
        print(f"artefacts: {run.path}")
    else:
        logger.warning("no test split; skipping final evaluation")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from wasteclf.data.manifest import DatasetManifest, Split, build_manifest
    from wasteclf.data.pipeline import make_dataset
    from wasteclf.evaluation.metrics import evaluate, predict_split
    from wasteclf.evaluation.plots import save_all_plots
    from wasteclf.inference.predictor import Predictor
    from wasteclf.utils.run import RunDirectory

    setup_logging(logging.INFO)
    run = RunDirectory.open(args.run)
    cfg = Config.load(run.config_path)
    predictor = Predictor.from_run(args.run)

    manifest_path = run.path / "manifest.csv"
    data_root = args.data_root or cfg.data.root
    if manifest_path.exists() and not args.rescan:
        # Reuse the training-time split so the test set is the same images.
        manifest = DatasetManifest.load(manifest_path, data_root)
        logger.info("reusing split from %s", manifest_path)
    else:
        logger.warning("rescanning %s; the split may differ from training", data_root)
        manifest = build_manifest(
            data_root,
            cfg.data.train_split,
            cfg.data.val_split,
            cfg.data.test_split,
            seed=cfg.data.split_seed,
            class_names=predictor.class_names,
        )

    split = Split(args.split)
    paths, labels = manifest.paths(split), manifest.labels(split)
    if not paths:
        print(f"split {args.split!r} is empty", file=sys.stderr)
        return 1

    dataset = make_dataset(
        paths,
        labels,
        manifest.num_classes,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        training=False,
        cache=False,
    )
    y_true, y_prob = predict_split(predictor.model, dataset)
    report = evaluate(y_true, y_prob, manifest.class_names, split=args.split)

    print(report.format_table())
    print(
        f"\nmacro F1 {report.macro_f1:.3f} | balanced accuracy {report.balanced_accuracy:.3f} "
        f"| top-2 {report.top2_accuracy:.3f} | ECE {report.expected_calibration_error:.3f}"
    )
    if report.most_confused:
        print("\nmost confused pairs:")
        for pair in report.most_confused:
            print(
                f"  {pair['true']:>12} -> {pair['predicted']:<12} {pair['count']:>4}  ({pair['rate']:.0%})"
            )

    out = Path(args.out) if args.out else run.path / f"metrics_{args.split}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    save_all_plots(report, run.plots_dir / args.split)
    print(f"\nwritten to {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from wasteclf.constants import IMAGE_EXTENSIONS
    from wasteclf.inference.predictor import Predictor

    setup_logging(logging.WARNING if args.json else logging.INFO)
    predictor = Predictor.from_run(args.run, threshold=args.threshold)

    targets: list[str] = []
    for item in args.images:
        path = Path(item)
        if path.is_dir():
            targets.extend(
                str(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            targets.append(str(path))

    if not targets:
        print("no images found", file=sys.stderr)
        return 1

    results = predictor.predict(targets, batch_size=args.batch_size)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            flag = "  [below threshold]" if r.low_confidence else ""
            runner, runner_score = r.runner_up
            print(f"{Path(r.path).name:<40} {r.label:<12} {r.confidence:.3f}{flag}")
            if args.top2:
                print(f"{'':<40} {runner:<12} {runner_score:.3f}  (runner-up)")
    return 0


def cmd_scene(args: argparse.Namespace) -> int:
    """Segment a waste-area photo into regions and classify each one."""
    import numpy as np
    from PIL import Image

    from wasteclf.scene import ContentSegmenter, GridSegmenter, ScenePipeline

    setup_logging(logging.WARNING if args.json else logging.INFO)

    if args.onnx:
        from wasteclf.inference.onnx_predictor import OnnxPredictor

        classifier = OnnxPredictor.from_dir(args.run)
    else:
        from wasteclf.inference.predictor import Predictor

        classifier = Predictor.from_run(args.run)

    if args.segmenter == "grid":
        segmenter = GridSegmenter(args.rows, args.cols, args.overlap)
    else:
        segmenter = ContentSegmenter(
            args.rows, args.cols, args.overlap, args.min_activity, args.keep_top
        )

    pipeline = ScenePipeline(classifier, segmenter, args.min_confidence, args.batch_size)

    results = {}
    for item in args.images:
        result = pipeline.analyse_file(item)
        results[str(item)] = result.to_dict()
        if not args.json:
            print(f"\n{Path(item).name}")
            print(result.format_summary())

        if args.overlay:
            out = Path(args.overlay)
            out.mkdir(parents=True, exist_ok=True)
            target = out / f"{Path(item).stem}_scene.png"
            _draw_overlay(np.asarray(Image.open(item).convert("RGB")), result, target)
            if not args.json:
                print(f"overlay: {target}")

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def _draw_overlay(image, result, path: Path) -> Path:
    """Draw accepted regions and their labels onto the scene."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    labels = sorted({label for _, label, _ in result.detections})
    colours = dict(zip(labels, plt.cm.tab20.colors))

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image.astype("uint8"))
    ax.axis("off")

    for region, label, confidence in result.detections:
        colour = colours.get(label, "white")
        ax.add_patch(
            patches.Rectangle(
                (region.x, region.y),
                region.width,
                region.height,
                linewidth=1.4,
                edgecolor=colour,
                facecolor=colour,
                alpha=0.22,
            )
        )
        ax.text(
            region.x + 3,
            region.y + 13,
            f"{label} {confidence:.2f}",
            fontsize=6,
            color="white",
            bbox={"facecolor": colour, "alpha": 0.75, "pad": 1, "edgecolor": "none"},
        )

    ax.set_title(
        f"{len(result.detections)} regions | "
        + ", ".join(f"{k} {v:.0%}" for k, v in list(result.composition.items())[:4]),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def cmd_explain(args: argparse.Namespace) -> int:
    from wasteclf.explain.gradcam import GradCAM, save_explanation
    from wasteclf.inference.predictor import Predictor

    setup_logging(logging.INFO)
    predictor = Predictor.from_run(args.run)
    cam = GradCAM(predictor.model, layer_name=args.layer)
    logger.info("explaining layer %s", cam.layer_name)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in args.images:
        image = predictor.load_image(item)
        overlaid, index, score = cam.explain(image, class_index=args.class_index)
        label = predictor.class_names[index]
        target = out_dir / f"{Path(item).stem}_gradcam_{label}.png"
        save_explanation(overlaid, target, title=f"{label} ({score:.2f})")
        print(f"{Path(item).name} -> {label} ({score:.3f})  {target}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from wasteclf.inference.export import export_onnx, export_savedmodel, export_tflite
    from wasteclf.inference.predictor import Predictor
    from wasteclf.utils.run import RunDirectory

    setup_logging(logging.INFO)
    run = RunDirectory.open(args.run)
    predictor = Predictor.from_run(args.run)
    out = Path(args.out) if args.out else run.path
    out.mkdir(parents=True, exist_ok=True)

    wants = {args.format} if args.format != "all" else {"savedmodel", "tflite", "onnx"}

    if "savedmodel" in wants:
        export_savedmodel(predictor.model, out / "savedmodel")
    if "tflite" in wants:
        export_tflite(predictor.model, out / "model.tflite", quantize=not args.no_quantize)
    if "onnx" in wants:
        export_onnx(predictor.model, out / "model.onnx", image_size=predictor.image_size)
        # The ONNX serving path has no run directory to read, so the label order
        # and input size travel with the model.
        labels = out / "labels.json"
        labels.write_text(
            json.dumps(
                {
                    "class_names": predictor.class_names,
                    "num_classes": len(predictor.class_names),
                    "image_size": list(predictor.image_size),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out / 'model.onnx'} and {labels}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "serving needs the optional extra: pip install 'wasteclf[serve]'",
            file=sys.stderr,
        )
        return 1

    from wasteclf.serving.app import create_app

    setup_logging(logging.INFO)
    app = create_app(args.run, threshold=args.threshold)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# Parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wasteclf",
        description="Waste image classification: scan, train, evaluate, explain, serve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  wasteclf scan --data-root data/raw\n"
            "  wasteclf train -c configs/garbage12-fast.yaml --data-root data/raw\n"
            "  wasteclf train -c configs/base.yaml --set train.warmup.epochs=2 --set model.backbone=mobilenetv2\n"
            "  wasteclf evaluate --run runs/efficientnetb0-20260920-101500 --split test\n"
            "  wasteclf predict --run runs/latest data/samples/ --json\n"
            "  wasteclf explain --run runs/latest image.jpg --out explanations/\n"
            "  wasteclf scene --run runs/latest dump.jpg --overlay scenes/\n"
            "  wasteclf export --run runs/latest --format onnx --out api/model\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"wasteclf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", help="path to a YAML config file")
        p.add_argument(
            "--set",
            type=_parse_override,
            action="append",
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set train.warmup.epochs=3 (repeatable)",
        )

    p_backbones = sub.add_parser("backbones", help="list the available backbones")
    p_backbones.set_defaults(func=cmd_backbones)

    p_scan = sub.add_parser("scan", help="scan a dataset directory and report the split")
    add_config_args(p_scan)
    p_scan.add_argument("--data-root", help="dataset root (overrides the config)")
    p_scan.add_argument("--out", help="write the manifest CSV here")
    p_scan.add_argument(
        "--verify", action="store_true", help="open every image to find corrupt files"
    )
    p_scan.set_defaults(func=cmd_scan)

    p_train = sub.add_parser("train", help="train a model and evaluate it on the test split")
    add_config_args(p_train)
    p_train.add_argument("--data-root", help="dataset root (overrides the config)")
    p_train.add_argument(
        "--deterministic", action="store_true", help="enable op determinism (slower)"
    )
    p_train.add_argument("-v", "--verbose", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a finished run")
    p_eval.add_argument("--run", required=True, help="run directory")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--data-root", help="dataset root (overrides the run config)")
    p_eval.add_argument(
        "--rescan", action="store_true", help="rebuild the split instead of reusing manifest.csv"
    )
    p_eval.add_argument("--out", help="metrics JSON destination")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="classify images or a directory of images")
    p_pred.add_argument("--run", required=True, help="run directory")
    p_pred.add_argument("images", nargs="+", help="image paths or directories")
    p_pred.add_argument(
        "--threshold", type=float, default=0.0, help="flag predictions below this confidence"
    )
    p_pred.add_argument("--batch-size", type=int, default=32)
    p_pred.add_argument("--top2", action="store_true", help="also show the runner-up class")
    p_pred.add_argument("--json", action="store_true", help="emit JSON")
    p_pred.set_defaults(func=cmd_predict)

    p_explain = sub.add_parser("explain", help="write Grad-CAM overlays")
    p_explain.add_argument("--run", required=True)
    p_explain.add_argument("images", nargs="+")
    p_explain.add_argument("--out", default="explanations", help="output directory")
    p_explain.add_argument(
        "--layer", help="backbone layer to explain (defaults to the last conv layer)"
    )
    p_explain.add_argument(
        "--class-index", type=int, help="explain this class instead of the predicted one"
    )
    p_explain.set_defaults(func=cmd_explain)

    p_scene = sub.add_parser("scene", help="segment a waste-area photo and classify every region")
    p_scene.add_argument(
        "--run", required=True, help="run directory (or ONNX model dir with --onnx)"
    )
    p_scene.add_argument("images", nargs="+", help="scene photographs")
    p_scene.add_argument(
        "--segmenter",
        default="content",
        choices=["grid", "content"],
        help="grid tiles everything; content drops low-variance background tiles",
    )
    p_scene.add_argument("--rows", type=int, default=6)
    p_scene.add_argument("--cols", type=int, default=8)
    p_scene.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="fraction shared between neighbouring tiles, so objects on a boundary are not lost",
    )
    p_scene.add_argument(
        "--min-activity",
        type=float,
        default=0.06,
        help="content segmenter: drop tiles below this variance score",
    )
    p_scene.add_argument(
        "--keep-top", type=int, help="content segmenter: cap the number of tiles kept"
    )
    p_scene.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="regions below this are reported as rejected, not counted",
    )
    p_scene.add_argument("--batch-size", type=int, default=32)
    p_scene.add_argument("--overlay", help="write an annotated image into this directory")
    p_scene.add_argument("--onnx", action="store_true", help="load an ONNX model directory instead")
    p_scene.add_argument("--json", action="store_true")
    p_scene.set_defaults(func=cmd_scene)

    p_export = sub.add_parser("export", help="export SavedModel, TFLite or ONNX")
    p_export.add_argument("--run", required=True)
    p_export.add_argument(
        "--format", default="all", choices=["savedmodel", "tflite", "onnx", "all"]
    )
    p_export.add_argument("--out", help="destination directory (defaults to the run directory)")
    p_export.add_argument("--no-quantize", action="store_true", help="skip TFLite quantisation")
    p_export.set_defaults(func=cmd_export)

    p_serve = sub.add_parser("serve", help="serve the model over HTTP")
    p_serve.add_argument("--run", required=True)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--threshold", type=float, default=0.0)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    setup_logging(logging.INFO)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError, KeyError) as exc:
        # Expected, user-correctable failures: report the message, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
