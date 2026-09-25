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

# `full`'s text encoder is embedding-trained: a masked-LM backbone's pooled
# states (ModernBERT-base's own) retrieve poorly, and graph propagation has
# nothing to denoise in them.
_TEXT = {Scale.SMALL: "tiny_bert", Scale.FULL: "Alibaba-NLP/gte-modernbert-base"}
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


def training_dtype(scale: Scale) -> str:
    """The backbone precision a chapter fine-tunes at: `bf16` at `full` — the
    tensor-core precision of the sm_80+ GPU a full run needs, at half f32's
    memory — and `f32` at `small`, on a CPU, where `bf16` is refused."""
    return "bf16" if scale is Scale.FULL else "f32"
