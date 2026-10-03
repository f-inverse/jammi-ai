"""The cookbook's shared library: it *composes* jammi and *checks* what the
chapters claim and carry — it implements no graph or ML logic of its own.

The dataset loaders live in :mod:`jammi_cookbook.datasets` and are imported
lazily (they pull the optional ``data`` extra); the core claims/rails surface
imports clean without them.
"""

from __future__ import annotations

from . import claims, rails

__all__ = ["claims", "rails"]
