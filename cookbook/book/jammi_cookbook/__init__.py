"""The cookbook's shared library: it *composes* jammi_ai and *enforces* the
contracts and rails — it implements no graph or ML logic of its own.

Importing the package applies the determinism contract (K0 §3) as a side effect,
so a chapter's first line — ``import jammi_cookbook`` — pins the reproducible
regime before any heavy native library is touched.

The dataset loaders live in :mod:`jammi_cookbook.datasets` and are imported
lazily (they pull the optional ``data`` extra); the core contracts/rails surface
imports clean without them.
"""

from __future__ import annotations

from . import contracts, determinism, rails

__all__ = ["contracts", "determinism", "rails"]
