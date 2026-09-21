# Scene pipeline and the fused backbone

Version 2 classified one object per image. This adds two things: a taxonomy of
twelve categories instead of seven, and a scene stage that finds objects in a
photograph of a dump area before anything tries to name them.

```
                    ┌─────────────────────┐
   dump photo  ───► │  region proposal    │  grid tiling, or content-filtered
                    └──────────┬──────────┘
                               │  N crops
                    ┌──────────▼──────────┐
                    │  recognition        │  SwinConvNeXt or any registry backbone
                    └──────────┬──────────┘
                               │  N x 12 probabilities
                    ┌──────────▼──────────┐
                    │  aggregation        │  counts, area share, coverage
                    └─────────────────────┘
```

## Why two branches

The literature on this exact dataset converged on hybrids. Kunwar et al. report
that an enhanced Swin Transformer alone reaches about 78% on the twelve-class
benchmark and an enhanced ConvNeXt alone about 80%, while fusing them with
spatial attention reaches 98.97% ([Scientific Reports 15,
2025](https://www.nature.com/articles/s41598-025-91302-7)). Those numbers come
from the abstract. The full text would not open from this network.

The reasoning is that the two families fail differently on waste. Convolution
resolves local material cues: the fibre edge that separates cardboard from
paper, the specular highlight that separates metal from white glass.
Windowed attention resolves global layout: whether the frame holds one crumpled
garment or a heap of organic matter. Three of the twelve classes are glass
distinguished only by colour, and two more are clothes against shoes, so both
kinds of evidence are doing work.

`wasteclf/models/swinconvnext.py` implements that shape:

```
image (0-255)
  ├── ConvNeXt-Tiny ──────► 1x1 conv → LayerNorm → spatial attention ─┐
  └── rescale → Swin-T ───► 1x1 conv → LayerNorm → spatial attention ─┤
                                                                      ▼
                                          concat → 1x1 conv → GELU → spatial attention
```

Both branches project to a shared width before concatenation. Without that,
ConvNeXt's 768 channels would swamp the Swin branch purely by count.

The spatial attention is CBAM's ([Woo et al., ECCV
2018](https://arxiv.org/abs/1807.06521)): pool across channels to get a
per-position summary, learn one convolution over it, gate with a sigmoid.

### The honest caveat

ConvNeXt loads ImageNet weights. Swin does not, because there is no public
Keras checkpoint and the ones that exist live on Kaggle Hub. That branch starts
from scratch on roughly 15,000 images, which is nowhere near enough to train a
transformer from nothing.

So do not expect 98.97% from this implementation as it stands. Either supply
pretrained Swin weights, or run `configs/garbage12-fast.yaml` first, which uses
a single fully-pretrained EfficientNetB0 and will almost certainly beat the
fused model until the Swin branch has weights worth having.

## Swin, implemented from the paper

`wasteclf/models/swin.py` follows [Liu et al., ICCV
2021](https://arxiv.org/abs/2103.14030): patch embedding, windowed multi-head
self-attention with a learned relative position bias, alternating shifted
windows, patch merging between stages. At the default configuration it comes to
27.8 million parameters, which matches the published Swin-T backbone.

Three details cost time and are worth recording.

**Windows must divide the feature map.** Partitioning is a reshape, not a
padded convolution, so a 7x7 window cannot tile an 8x8 map. Swin's own 7 is
chosen because it divides 56, 28, 14 and 7, which is why the reference setup
starts at 224x224. `fit_window()` shrinks the window when a stage lands on a
resolution the nominal size does not divide.

**The attention layer has to be built after the window is known.** Its relative
position table is sized `(2W-1)^2`, so constructing it with the nominal window
and then feeding it effective-window tokens produces a reshape error deep in
the call. It is created in `build()`, not `__init__()`.

**Sublayers must be built explicitly.** Keras uses `compute_output_shape()`
during functional construction and never calls `call()`, so a lazily built
sublayer stays unbuilt. The model then reports 1.8M parameters instead of 27.8M
and hands the optimiser an incomplete variable list.

## The scene stage

Two proposers, neither of which needs training data.

`GridSegmenter` tiles the frame, optionally with overlap so an object
straddling a boundary is not cut in half and missed twice. It is the literal
reading of the architecture diagram.

`ContentSegmenter` tiles the same way, then drops tiles whose mean per-channel
standard deviation falls below a threshold. Waste is cluttered and
high-variance; tarmac, soil and sky are not. On a test scene of fifteen real
waste crops scattered over a flat background, it dropped ten of forty-eight
tiles; those ten contained 0.0% waste by area, against 50.9% for the tiles it
kept, with no false drops.

This matters because the classifier is closed-set. A tile of bare ground still
comes back as one of the twelve classes with a confidence attached, so
something has to decide what is worth classifying. `min_confidence` is the
second line of defence, and rejected regions are reported rather than discarded
so the count stays auditable.

Learned proposers (SAM, YOLO-seg) satisfy the same `Segmenter` protocol and can
be dropped in without touching the pipeline. They are not implemented here
because both need weights this project does not ship.

### Aggregation is by area, not by count

A grid tile is a unit of sampling, not a unit of waste. Two tiles of cardboard
and one of battery does not make the pile one third battery.
`SceneResult.composition` weights each class by pixel area instead.

## Datasets

`scripts/fetch_data.py` handles both.

`garbage12` is Garbage Classification V2, 15,515 images across the twelve
categories, hosted on Kaggle and therefore needing credentials. It is tracked
with Git LFS; see the note in `.gitattributes` about the free tier being smaller
than the dataset.

`trashnet` is the original six-class Stanford CS229 set, 2,527 images in a
public GitHub repository, no credentials needed. It is the faster set to develop
against and it is what the pipeline in this repository has actually been run on.

For the scene stage there is no annotated public dataset in reach here. The
closest is [ZeroWaste](https://arxiv.org/abs/2106.02740), a CVPR 2022 benchmark
for deformable object segmentation on conveyor imagery; its files are on Zenodo.
[AgaMiko's survey](https://github.com/AgaMiko/waste-datasets-review) lists the
rest.

## What has actually been run

On this machine, against TrashNet: the scan, the full two-stage trainer with the
fused backbone, evaluation, and the scene pipeline end to end including the
overlay renderer. One epoch of SwinConvNeXt on 1,769 images took 561 seconds on
four CPU cores and reached 0.273 training accuracy against 0.167 for chance, so
the gradients reach both branches.

No model here has been trained to convergence, and nothing in this repository
has been run against the twelve-class dataset, which needs Kaggle credentials
this environment does not have.
