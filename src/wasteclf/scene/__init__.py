"""Scene-level analysis: segment a waste dump area, then classify each region.

The classifier on its own answers "what is this one object". A dump area holds
many objects at once, so something has to decide *where* to look before
anything can decide *what* it is. That is what this package does.

Free of TensorFlow, so it composes with either predictor.
"""

from wasteclf.scene.pipeline import ScenePipeline, SceneResult
from wasteclf.scene.segmenters import (
    ContentSegmenter,
    GridSegmenter,
    Region,
    Segmenter,
)

__all__ = [
    "ScenePipeline",
    "SceneResult",
    "Segmenter",
    "GridSegmenter",
    "ContentSegmenter",
    "Region",
]
