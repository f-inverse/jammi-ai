"""Seeds for the book's sampling steps.

A chapter's claims are relations that hold on any machine, device and thread
count, never bit-equalities of a measurement (see :mod:`jammi_cookbook.claims`),
so the cookbook leaves the engine to run as it would for any program: on the
reader's GPU when there is one, on every core otherwise. What the book does fix
is the seed of each step that samples — a split, a shuffle, a synthetic draw —
so that step draws the same rows on every run.
"""

from __future__ import annotations

# The seed every :func:`seeded` call folds in.
SEED = 0


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
