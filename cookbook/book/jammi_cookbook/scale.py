"""The scale a chapter runs at.

Every chapter runs its capability live. What it runs over is a scale:

* ``small`` — the committed fixtures and the tiny fixture encoders, on the CPU.
  Seconds per chapter; what CI renders and what a CPU Colab runtime runs.
* ``full`` — the published datasets and real encoders, on a GPU. The optional
  "at scale" run: the same chapter, the same code, over the data the book's
  findings are about.

The chapter code is identical at both; only the data, the encoders and the
frozen goldens a run is checked against differ. Choose with
``JAMMI_COOKBOOK_SCALE`` (``small`` when unset).
"""

from __future__ import annotations

import os
from enum import Enum

ENV = "JAMMI_COOKBOOK_SCALE"


class Scale(str, Enum):
    """A scale a chapter runs at (a string enum: ``Scale.SMALL == "small"``)."""

    SMALL = "small"
    FULL = "full"

    def __str__(self) -> str:
        return self.value


def current() -> Scale:
    """The scale this process runs at: ``JAMMI_COOKBOOK_SCALE``, else ``small``."""
    value = os.environ.get(ENV, Scale.SMALL.value)
    try:
        return Scale(value)
    except ValueError:
        raise ValueError(
            f"{ENV}={value!r} names no scale; expected one of "
            f"{', '.join(s.value for s in Scale)}"
        ) from None
