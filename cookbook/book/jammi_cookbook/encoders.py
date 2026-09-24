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


def text(scale: Scale) -> str:
    """The text encoder's model id at ``scale``."""
    model = _TEXT[scale]
    return fixtures.model(model) if scale is Scale.SMALL else model
