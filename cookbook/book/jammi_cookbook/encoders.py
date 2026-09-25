"""The encoder a chapter runs at each scale.

``small`` runs the committed fixture encoders — randomly initialized, a few
hundred kilobytes, seconds on a CPU: they exercise every code path, but their
vectors carry no meaning, so a small run's quality numbers are regression
checks, not findings. ``full`` runs the real encoders the book's findings are
about.
"""

from __future__ import annotations

from . import fixtures
from .scale import Scale

_TEXT = {Scale.SMALL: "tiny_bert", Scale.FULL: "answerdotai/ModernBERT-base"}
# One OpenCLIP checkpoint carries both an image and a text tower.
_IMAGE = {Scale.SMALL: "tiny_open_clip", Scale.FULL: "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"}
_AUDIO = {Scale.SMALL: "htsat_clap_tiny", Scale.FULL: "laion/clap-htsat-fused"}


def _model(models: dict[Scale, str], scale: Scale) -> str:
    model = models[scale]
    return fixtures.model(model) if scale is Scale.SMALL else model


def text(scale: Scale) -> str:
    """The text encoder's model id at ``scale``."""
    return _model(_TEXT, scale)


def image(scale: Scale) -> str:
    """The image (and paired text) encoder's model id at ``scale``."""
    return _model(_IMAGE, scale)


def audio(scale: Scale) -> str:
    """The audio encoder's model id at ``scale``."""
    return _model(_AUDIO, scale)
