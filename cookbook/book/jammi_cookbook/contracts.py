"""Frozen goldens: the measured verdicts every chapter ends in.

A chapter runs its capability live and checks what it measured against a
frozen golden — ``assert_close("arxiv.tier02.recall_at_10", recall)``. A
golden is a value and the tolerance the check allows, addressed
``"<dataset>.<key>"`` and kept in ``goldens/``:

* ``goldens/<dataset>.json`` — a dataset whose numbers do not depend on the
  scale (a model-free chapter runs the same everywhere);
* ``goldens/<dataset>.<scale>.json`` — one file per scale, for a dataset whose
  numbers do (see :mod:`jammi_cookbook.scale`).

A dataset is one kind or the other, never both.

**Freezing.** A golden is never typed in: it is what a live run measured. With
``JAMMI_COOKBOOK_FREEZE=1`` set, :func:`assert_close` records the observed
value instead of checking it — an existing entry keeps its tolerance, a new one
takes the call's ``tol`` — so re-freezing after a deliberate change is running
the chapter once, at the scale being frozen, and reviewing the diff.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import scale as _scale

GOLDENS = Path(__file__).resolve().parent / "goldens"
FREEZE_ENV = "JAMMI_COOKBOOK_FREEZE"


@dataclass(frozen=True)
class Golden:
    """A frozen metric: the recorded value and the tolerance a check allows."""

    value: float
    tol: float

    def contains(self, observed: float) -> bool:
        return abs(observed - self.value) <= self.tol


def golden_file(dataset: str, scale: _scale.Scale) -> Path:
    """The file holding ``dataset``'s goldens at ``scale``."""
    free = GOLDENS / f"{dataset}.json"
    scaled = GOLDENS / f"{dataset}.{scale}.json"
    if free.exists() and any(GOLDENS.glob(f"{dataset}.*.json")):
        raise ValueError(
            f"goldens for {dataset!r} are both scale-free ({free.name}) and per-scale: "
            "a dataset is one kind or the other"
        )
    return free if free.exists() else scaled


def _split(metric: str) -> tuple[str, str]:
    dataset, _, key = metric.partition(".")
    if not key:
        raise ValueError(f"a metric is '<dataset>.<key>', got {metric!r}")
    return dataset, key


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def golden(metric: str, scale: _scale.Scale | None = None) -> Golden:
    """The frozen ``metric`` at ``scale`` (the running scale when omitted)."""
    dataset, key = _split(metric)
    path = golden_file(dataset, scale or _scale.current())
    entries = _read(path)
    if key not in entries:
        raise KeyError(
            f"no golden {metric!r} in {path.name}: a chapter asserting a metric nobody "
            f"froze is a gap — freeze it with {FREEZE_ENV}=1"
        )
    entry = entries[key]
    return Golden(value=float(entry["value"]), tol=float(entry["tol"]))


def assert_close(metric: str, observed: float, *, tol: float = 0.0) -> float:
    """Check ``observed`` against the frozen ``metric`` at the running scale;
    return ``observed`` so a cell checks and displays at once.

    ``tol`` is the tolerance a first freeze records; once frozen, the golden's
    own tolerance is the one checked.
    """
    if os.environ.get(FREEZE_ENV) == "1":
        _freeze(metric, observed, tol)
        return observed
    g = golden(metric)
    if not g.contains(observed):
        raise AssertionError(
            f"{metric} at scale {_scale.current()}: measured {observed:.6g}, frozen "
            f"{g.value:.6g} ± {g.tol:.3g} (off by {abs(observed - g.value):.3g}). The "
            "engine's behaviour moved; if the move is intended, re-freeze and review the diff."
        )
    return observed


def _freeze(metric: str, observed: float, tol: float) -> None:
    dataset, key = _split(metric)
    path = golden_file(dataset, _scale.current())
    entries = _read(path)
    kept_tol = entries.get(key, {}).get("tol", tol)
    entries[key] = {"value": float(observed), "tol": float(kept_tol)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
