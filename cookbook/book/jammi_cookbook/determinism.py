"""The determinism contract, applied on import.

Importing :mod:`jammi_cookbook` pins the process into the reproducible regime the
whole book depends on: single-threaded BLAS/OMP, tokenizer parallelism off, a
fixed dtype, and a pinned seed. A chapter's claims are relations, never
bit-equalities of a measurement: BLAS matmul order varies across machines, so a
claim states what holds on any of them (see :mod:`jammi_cookbook.claims`).
"""

from __future__ import annotations

import os

from . import scale as _scale

# The pinned seed every :func:`seeded` call folds in.
SEED = 0


def _apply_env() -> None:
    """Pin the threading / tokenizer / dtype environment, and the device the
    running scale runs on.

    Set before any heavy native library (BLAS, tokenizers, torch) reads these on
    its first use. Importing the cookbook is therefore the first thing a chapter
    does, ahead of importing jammi.
    """
    pinned = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # The small scale runs on the CPU (jammi_cookbook.scale), so it measures the
    # same numbers on any host: an encoder on a GPU agrees with the CPU only
    # within the engine's device-parity tolerance, and a recall over
    # sign-quantized vectors turns that into a different query. Every session
    # and spawned server reads the device from this variable.
    if _scale.current() is _scale.Scale.SMALL:
        pinned["JAMMI_GPU__DEVICE"] = "-1"
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
