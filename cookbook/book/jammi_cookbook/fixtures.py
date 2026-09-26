"""The cookbook's fixtures: the small corpora, golden sets and tiny model
checkpoints every recipe and chapter runs on.

They ship inside this package, so a fresh install — a Colab runtime, a laptop
with no repository checkout — runs every recipe as it is. In a checkout,
``_fixtures`` is a link to ``cookbook/fixtures``, the tree the engine's own
tests read too, so both read the same bytes.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "_fixtures"


def path(name: str) -> Path:
    """The fixture file or directory ``name`` (e.g. ``"tiny_corpus.parquet"``)."""
    found = _ROOT / name
    if not found.exists():
        raise FileNotFoundError(f"no cookbook fixture named {name!r} under {_ROOT}")
    return found


def url(name: str) -> str:
    """The fixture ``name`` as the ``file://`` URL ``add_source`` registers."""
    return path(name).as_uri()


def model(name: str) -> str:
    """The checkpoint directory ``name`` as a local model id (``local:<path>``)."""
    return f"local:{path(name)}"
