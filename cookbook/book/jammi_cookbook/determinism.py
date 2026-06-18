"""The determinism contract (K0 §3), applied on import.

Importing :mod:`jammi_cookbook` pins the process into the reproducible regime the
whole book depends on: single-threaded BLAS/OMP, tokenizer parallelism off, a
fixed dtype, and a recorded seed. Two facts make the book reproducible *across
machines and library versions*, not just across runs on one box:

* **Subset identity is committed, not seeded.** A seed does not reproduce the
  same node selection across library versions, so the selected ``_row_id`` lists
  live in ``data/ids/`` and are the source of truth. :func:`committed_ids` reads
  them; the seed is recorded only for provenance.
* **Metrics are asserted to tolerances, not bit-equality.** BLAS matmul order
  varies, so the committed artifacts are compared against frozen vectors within a
  tolerance (see :mod:`jammi_cookbook.contracts`).
"""

from __future__ import annotations

import os
from pathlib import Path

# The pinned seed. Recorded for provenance only — subset identity comes from the
# committed ID lists, never from replaying this seed (see module docstring).
SEED = 0

# Repo root is two levels up from this file (jammi_cookbook/determinism.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_IDS_DIR = _REPO_ROOT / "data" / "ids"


def _apply_env() -> None:
    """Pin the threading / tokenizer / dtype environment.

    Set before any heavy native library (BLAS, tokenizers, torch) reads these on
    its first use. Importing the cookbook is therefore the first thing a chapter
    does, ahead of importing jammi_ai.
    """
    pinned = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # setdefault, not overwrite: an operator who has deliberately set a value
    # (e.g. the opt-in full-scale run) keeps it; the default regime is otherwise.
    for key, value in pinned.items():
        os.environ.setdefault(key, value)


_apply_env()


def seeded(name: str) -> int:
    """A stable per-use-site seed derived from :data:`SEED` and ``name``.

    Distinct call sites get distinct but fixed seeds, so adding a seeded step does
    not perturb an earlier one. Deterministic across runs and machines: a pure
    function of its inputs.
    """
    # FNV-1a over the name, folded with the global seed. Pure and portable —
    # avoids hash() randomization (PYTHONHASHSEED) entirely.
    h = 0x811C9DC5
    for byte in name.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return (h ^ SEED) & 0x7FFFFFFF


def committed_ids(dataset: str) -> list[str]:
    """Return the committed ``_row_id`` subset for ``dataset``.

    Reads ``data/ids/<dataset>.txt`` — the source of truth for which rows the
    book runs on. Raises if the list is missing rather than silently selecting a
    different subset (which would drift every committed metric).
    """
    path = _IDS_DIR / f"{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"committed id list not found: {path}. The subset for '{dataset}' must "
            f"be committed under data/ids/ — subset identity is never reproduced "
            f"from a seed (see jammi_cookbook.determinism)."
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
