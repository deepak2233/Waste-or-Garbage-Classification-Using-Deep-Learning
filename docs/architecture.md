# Architecture

## Layout

```
src/wasteclf/
├── config.py          typed config, YAML loading, dotted overrides
├── constants.py       shared literals
├── cli.py             argparse entry point, one function per subcommand
├── data/
│   ├── manifest.py    directory scan, stratified split, class weights
│   ├── pipeline.py    tf.data input pipelines
│   └── augment.py     augmentation as Keras layers
├── models/
│   ├── backbones.py   registry; per-backbone preprocessing layer
│   └── build.py       model assembly, fine-tuning control
├── training/
│   ├── trainer.py     two-stage loop
│   └── callbacks.py   macro-F1 callback, callback assembly
├── evaluation/
│   ├── metrics.py     per-class report, confusion matrix, calibration
│   └── plots.py       diagnostic plots
├── explain/gradcam.py Grad-CAM
├── inference/
│   ├── predictor.py   load a run, classify images
│   └── export.py      SavedModel and TFLite
└── serving/app.py     FastAPI service
```

Dependencies point one way: `cli` → `training` → `models` → `data` → `config`.
Nothing under `data/` imports from `models/`, and nothing under `models/`
imports from `training/`.

## Where preprocessing lives, and why it matters

The input pipeline emits raw float32 RGB in `[0, 255]`. Normalisation is a layer
inside the model:

```
Input(0-255 RGB)
  → augmentation        Keras Random* layers, inert when training=False
  → BackbonePreprocessing   the backbone's own preprocess_input
  → backbone            frozen in stage 1
  → pooling             GlobalAveragePooling2D by default
  → Dense → BatchNorm → Dropout
  → Dense(softmax, dtype=float32)
```

Two things follow from that arrangement.

Augmentation cannot leak into evaluation. Keras `Random*` layers are no-ops
unless `training=True`, so `predict` and `evaluate` see the image as supplied.
There is no flag to remember to set.

Serving cannot apply the wrong normalisation, because there is no normalisation
left for it to apply. A `.keras` file, a SavedModel and a `.tflite` export all
take raw pixels. Switching backbone changes the preprocessing automatically,
since the layer reads it from the registry.

The softmax is pinned to `float32` so mixed precision does not cost probability
resolution.

## Two-stage training

Stage one trains the head with the backbone frozen. Stage two unfreezes the top
`unfreeze_layers` of the backbone at a learning rate typically 100x lower.

Skipping stage one destroys the pretrained features. The head starts random, so
its first gradients are large, and they land on convolution filters that took an
ImageNet run to learn.

`set_finetune_trainable()` is always followed by a recompile. Keras captures the
trainable-variable list when you compile, so changing trainability without
recompiling silently trains the head alone and reports it as fine-tuning.

`freeze_batchnorm=True` keeps BatchNormalization frozen inside the unfrozen
region. On a few thousand images the batch statistics are noisy enough that
updating them degrades the pretrained features, and it shows as validation
accuracy collapsing in the first fine-tuning epoch. It matters most for ResNet
and EfficientNet.

### A trap worth knowing

`model.weights=null` with a frozen backbone cannot learn. An untrained backbone
in inference mode carries BatchNorm statistics that were never fitted, which on
MobileNetV2 drives every ReLU6 to its floor; the pooled features come out
identical for every image. The loss parks at `ln(num_classes)` exactly. The
trainer warns when a config asks for this, and
`test_frozen_random_backbone_emits_constant_features` pins the behaviour.

## Splitting

`build_manifest()` walks the dataset root, sorts each class's files by a SHA-256
hash of the relative path, shuffles with a fixed seed, and assigns splits per
class.

Sorting by hash first means the split does not depend on the order the
filesystem returns directory entries. That order differs between ext4, APFS and
a freshly unzipped archive, which is enough to silently change which images end
up in the test set between two machines running identical code.

Classes with three or more images always get at least one image in validation
and one in test. Without that floor, a rare class can land entirely in training,
and its per-class precision and recall become undefined while the macro average
excludes it without saying so.

The result is written to `manifest.csv`. `wasteclf evaluate` reloads it rather
than re-splitting, so evaluation runs against the same held-out images the
training run never saw.

## Run directories

```
runs/efficientnetb0-20260920-101500/
├── config.yaml              every setting used
├── labels.json              class order, the training/inference contract
├── manifest.csv             the exact split
├── model.keras              architecture and weights
├── checkpoint.weights.h5    best epoch, written during training
├── history.csv              per-epoch metrics, both stages
├── metrics.json             the test report
├── dataset_summary.json     counts per class and split
├── training_summary.json    epochs, duration, class weights
├── train.log
└── plots/
```

`Predictor.from_run()` reads `labels.json` for the class order and `config.yaml`
for the image size, so inference never needs to be told what the model expects.
It raises if the label count and the model's output width disagree.

## Metrics

`val_macro_f1` is the default monitor, not `val_accuracy`. With imbalanced
classes, accuracy rewards ignoring the small ones: a 3-class model that never
predicts the rare class scores 90% accuracy and 0.64 macro F1, with the rare
class at 0.00 F1.

The report also carries expected calibration error, the gap between mean
confidence and accuracy. A fine-tuned softmax is usually overconfident. On a
check run over generated images the model averaged 0.71 confidence against 0.44
accuracy, ECE 0.31. Anything routing on a confidence threshold has to account
for that.

## Testing

102 tests. `pytest` runs the fast ones; `pytest -m slow` adds the end-to-end
training, serving and export tests.

Two are worth singling out, because both cover things a unit test does not see.

`test_model_loads_in_a_fresh_process` spawns a subprocess and loads the saved
model there. Inside one pytest process the custom preprocessing layer is already
registered as a side effect of building a model, so a plain load test passes
while `wasteclf predict` fails in a fresh interpreter. That is why
`predictor.py` imports the module explicitly.

`test_plots_are_written_and_are_not_empty` asserts a floor on the PNG file
size. A plotting path that silently writes a valid but empty canvas is the kind
of failure a smoke test waves through.
