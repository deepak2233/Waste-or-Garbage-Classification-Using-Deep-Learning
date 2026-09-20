# Waste classification

Sorts images of waste into seven categories (cardboard, compost, glass, metal,
paper, plastic and trash) using transfer learning on TensorFlow/Keras.

I started this in 2019 as a couple of notebooks. It worked, but it only ever
worked on my laptop: hardcoded paths, no way to run it twice and get the same
answer, no way to use a trained model without opening Jupyter again. This
version is a proper package. Same idea, engineering that holds up.

The notebooks are still here under [`legacy/`](legacy/).

[![CI](https://github.com/deepak2233/Waste-or-Garbage-Classification-Using-Deep-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/deepak2233/Waste-or-Garbage-Classification-Using-Deep-Learning/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
git clone https://github.com/deepak2233/Waste-or-Garbage-Classification-Using-Deep-Learning.git
cd Waste-or-Garbage-Classification-Using-Deep-Learning
pip install -e ".[all]"
```

## Quick check, no dataset needed

The images are not in the repo. To confirm the install works:

```bash
python scripts/make_synthetic_data.py --out data/synthetic --per-class 40
wasteclf train -c configs/smoke.yaml
```

That generates coloured images, trains a small model on them and writes a
complete run directory. Under a minute on a laptop CPU. The model is useless.
The point is that every stage runs.

## With the real data

```bash
python scripts/fetch_data.py                  # where to download it
python scripts/fetch_data.py --check data/raw # validate what you unpacked
```

One folder per class:

```
data/raw/
├── cardboard/
├── compost/
├── glass/
└── ...
```

Then:

```bash
wasteclf scan  --data-root data/raw
wasteclf train -c configs/efficientnetb0.yaml --data-root data/raw
```

`scan` prints the class counts and the split before you commit to a training
run. Worth doing first — it also flags files that will not decode.

## Commands

```bash
wasteclf backbones                              # what you can train
wasteclf scan      --data-root data/raw --verify
wasteclf train     -c configs/efficientnetb0.yaml
wasteclf evaluate  --run runs/<name> --split test
wasteclf predict   --run runs/<name> path/to/images/ --top2
wasteclf explain   --run runs/<name> image.jpg --out explanations/
wasteclf export    --run runs/<name> --format onnx --out api/model
wasteclf serve     --run runs/<name> --port 8000
```

Anything in a config can be overridden on the command line, so you do not end up
with fifteen near-identical YAML files:

```bash
wasteclf train -c configs/base.yaml \
  --set model.backbone=resnet50v2 \
  --set train.finetune.epochs=40 \
  --set data.batch_size=64
```

## What a run leaves behind

```
runs/efficientnetb0-20260920-101500/
├── config.yaml            every setting used
├── labels.json            class order
├── manifest.csv           which image went to which split
├── model.keras            architecture and weights
├── metrics.json           per-class scores, confusion matrix, calibration
├── history.csv            per-epoch metrics for both stages
├── train.log
└── plots/
    ├── confusion_matrix.png
    ├── per_class_f1.png
    └── training_history.png
```

This is the part I most wanted. Six months later I can open a run directory and
know exactly what produced the number. `wasteclf predict` reads `labels.json`
and `config.yaml` straight from the run, so you never have to remember which
class was index 3 or what input size the thing was trained at.

## Reading the results

`evaluate` gives a per-class table rather than one accuracy figure:

```
class          precision   recall      f1  support
--------------------------------------------------
cardboard          0.188    1.000   0.316        6
compost            0.000    0.000   0.000        2
glass              0.000    0.000   0.000        6
...
--------------------------------------------------
macro avg                           0.045       32
accuracy                            0.188       32
```

That is a real (deliberately under-trained) run. It predicts `cardboard` for
almost everything. Accuracy of 0.188 tells you something is off; the 0.000 rows
tell you what.

The report also includes expected calibration error, the gap between how
confident the model is and how often it is right. Softmax outputs after
fine-tuning are usually overconfident, and if you plan to auto-accept
predictions above some threshold you need to know by how much.

## Backbones

| Backbone | Params | Notes |
| --- | --- | --- |
| `efficientnetb0` | 4.0M | Best accuracy per parameter. Start here. |
| `efficientnetv2b0` | 5.9M | Trains faster, tolerates a higher fine-tuning LR |
| `mobilenetv2` | 2.3M | For TFLite and anything running on a phone |
| `resnet50v2` | 23.6M | Fine-tunes more stably than v1 |
| `densenet121` | 7.0M | Good on paper against cardboard |
| `vgg16` | 14.7M | Where this started. Kept for comparison. |

Adding one is a single entry in `BACKBONES` in
[`src/wasteclf/models/backbones.py`](src/wasteclf/models/backbones.py).

Each entry carries its own `preprocess_input`, and the model applies it
internally. This matters more than it looks: VGG16 expects BGR with the ImageNet
mean subtracted, MobileNetV2 expects `[-1, 1]`, EfficientNet expects raw
`[0, 255]`. Feeding all of them `[0, 1]` costs accuracy and nothing warns you.

## Serving

```bash
wasteclf serve --run runs/<name> --port 8000
curl -F "file=@bottle.jpg" http://localhost:8000/predict
```

```json
{
  "path": "bottle.jpg",
  "label": "plastic",
  "confidence": 0.8731,
  "low_confidence": false,
  "scores": {"plastic": 0.8731, "glass": 0.0642, "metal": 0.0301},
  "latency_ms": 41.2
}
```

`POST /predict/batch` takes up to 64 images and reports per-file failures
without failing the whole request. In Docker:

```bash
docker build -t wasteclf .
docker run -p 8000:8000 -v $(pwd)/runs/<name>:/model:ro wasteclf
```

### Serverless

The TensorFlow stack is about 1.2 GB installed, which does not fit in a
serverless function. Exporting to ONNX drops the runtime to roughly 180 MB,
which does, and inference gets faster because there is no Keras in the way.

```bash
wasteclf export --run runs/<name> --format onnx --out api/model
```

That writes `model.onnx` and `labels.json`. `api/index.py` serves them through
onnxruntime and never imports TensorFlow; `vercel.json` points at it. The model
is gitignored, so deploy the repo and the endpoint reports `no_model` on
`/health` until you export one.

Two things to know before relying on it. The exported graph has the
augmentation layers stripped, because they contain random-sampling ops that
ONNX has no operator for and the resulting file will not load. And the ONNX
path decodes images with Pillow rather than TensorFlow, so the training
pipeline pins `dct_method="INTEGER_ACCURATE"` to make the two decoders produce
identical pixels. Without that pin they disagree by up to 4/255 on most pixels,
which was worth about 0.09 of probability on a test model.

## How the training works

Two stages. First the classifier head trains with the backbone frozen. Then the
top of the backbone unfreezes and training continues at a much lower learning
rate.

Doing it in one stage with everything unfrozen ruins the pretrained filters. The
head starts random, its early gradients are large, and they land on convolution
weights that took an ImageNet run to learn. Warming up the head first keeps
those gradients small by the time the backbone opens up.

BatchNorm stays frozen during fine-tuning by default. On a few thousand images
the batch statistics are too noisy to be worth updating, and when it goes wrong
you see validation accuracy fall off a cliff in the first fine-tuning epoch.

The loss is weighted by inverse class frequency and the default monitor is
`val_macro_f1`, not `val_accuracy`. With classes this uneven, accuracy pays the
model to ignore the small ones: a three-class model that never predicts the rare
class still scores 90% accuracy, with 0.64 macro F1 and 0.00 on the class it
dropped.

More detail in [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
make install-dev
make test        # fast tests
make test-all    # adds end-to-end training, serving, export
make lint
```

111 tests. The slow ones train a small model on generated data, reload it in a
fresh subprocess, run Grad-CAM, export to TFLite and ONNX, check the ONNX path
agrees with Keras, and hit every HTTP endpoint. None of them need the dataset
or a network connection.

## On the old numbers

The notebooks reported 43.03% for VGG16 and 80.8% after fine-tuning. I would not
quote those. That code passed the test directory in as `validation_data`, let
early stopping pick the best epoch against it, and then reported accuracy on the
same directory, so the model was chosen using the data it was scored on. It
also augmented the test set, which made the number move between runs.

I have not retrained on the full dataset yet, so there is no replacement figure
here. `configs/vgg16.yaml` builds the same architecture with a proper three-way
split; one run gives a number worth putting in a table.

## Licence

MIT. See [LICENSE](LICENSE).
