"""Backbone registry and model assembly."""

from wasteclf.models.backbones import (
    BACKBONES,
    BackboneSpec,
    available_backbones,
    get_backbone_spec,
)
from wasteclf.models.build import build_model, inference_model, set_finetune_trainable

__all__ = [
    "BACKBONES",
    "BackboneSpec",
    "available_backbones",
    "get_backbone_spec",
    "build_model",
    "inference_model",
    "set_finetune_trainable",
]
