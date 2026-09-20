# The original notebooks

The 2019 version, kept as it was. Nothing here is imported by
`src/wasteclf/`, and none of it runs on a current TensorFlow.

| Path | What it is |
| --- | --- |
| `model/` | The combined VGG16 + ResNet50 notebook and its script dump |
| `cnn-architecture/` | The same work split per backbone |
| `Waste_Garbage_Collection_improve.ipynb` | The fine-tuning run that reached 80.8% |
| `visualization_src/` | Training curves from those runs |
| `requirement.txt` | Pinned dependencies from March 2021 |
| `DataSets/` | Two text files holding the Google Drive link |

## Why it no longer runs

The `.py` files came out of `jupyter nbconvert`, which turned the notebook `cd`
magics into bare syntax:

```python
cd/Users/IRON MAN/Waste or Garbage Classification Using Deep Learning/Datasets/train
```

That is a `SyntaxError`, and it appears five times. Past that,
`model.fit_generator` was removed in TensorFlow 2.6 and `Adam(lr=...)` stopped
accepting `lr`.

The notebooks themselves still read fine. The fine-tuning one is the useful
one: the two-stage recipe in `src/wasteclf/training/trainer.py` follows the
same shape.

## Running the same architecture now

```bash
wasteclf train -c configs/vgg16.yaml --data-root data/raw
```

Same backbone, but with a separate validation split, so the number it reports
will not match the 43.03% in the old README. That figure came from a run where
the validation set and the test set were the same directory.
