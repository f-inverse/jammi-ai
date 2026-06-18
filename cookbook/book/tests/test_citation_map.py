"""The citation map's two rigor contracts, enforced as tests.

The K-bridge thesis is "one recipe = one equation = one canon line". That is only
honest if the map cannot dangle: every reference must resolve to a real
``references.bib`` entry, and every recipe's ``jammi_call`` must be a verb that
actually exists on the pinned engine (the grounded API reference). These tests
fail the build if either contract breaks — so a citation map row can never cite a
paper that is not in the bibliography or a Jammi verb that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

from jammi_cookbook import contracts
from jammi_cookbook.citation_map import CITATION_MAP

_REPO_ROOT = Path(contracts.__file__).resolve().parent.parent
_BIB = _REPO_ROOT / "references.bib"
_API_REFERENCE = _REPO_ROOT / "jammi_cookbook" / "_api_reference.md"


def _bib_keys() -> set[str]:
    """Every @<type>{<key>, … entry key defined in references.bib."""
    text = _BIB.read_text()
    return set(re.findall(r"@\w+\{([^,]+),", text))


def test_every_citation_resolves_to_a_bib_entry():
    """No dangling @cite: every bib_key in the map exists in references.bib."""
    defined = _bib_keys()
    assert defined, "references.bib defines no entries"
    for row in CITATION_MAP:
        for key in row.bib_keys:
            assert key in defined, (
                f"citation map row {row.recipe!r} cites '{key}', which is not a "
                f"references.bib entry (have: {sorted(defined)})"
            )


def test_every_jammi_call_exists_in_the_api_reference():
    """The map cannot cite a verb that does not exist on the pinned engine.

    Every recipe's ``jammi_call`` must appear in the grounded API reference
    (``_api_reference.md``), which ``scripts/check_api_reference.py`` in turn keeps
    in lockstep with the installed ``jammi_ai`` wheel.
    """
    reference = _API_REFERENCE.read_text()
    for row in CITATION_MAP:
        # word-boundary match so `search` does not match `search_by_id`
        pattern = rf"\b{re.escape(row.jammi_call)}\b"
        assert re.search(pattern, reference), (
            f"citation map row {row.recipe!r} runs '{row.jammi_call}', which does "
            f"not appear in the grounded API reference — the map cannot cite a verb "
            f"that does not exist on the pinned engine."
        )


def test_map_covers_every_tier_and_the_repair():
    """The map is complete: a row per tier (01–04) plus the conformal repair."""
    recipes = " ".join(row.recipe for row in CITATION_MAP)
    for tier in ("01 construct", "02 analyze", "03 learn", "04 predict", "04 quantify"):
        assert tier in recipes, f"citation map missing a row for {tier}"


def test_every_row_names_both_axes_and_a_call():
    """Each row pins monograph, canon, and a non-empty Jammi call (no blank cells)."""
    for row in CITATION_MAP:
        assert row.monograph and row.canon and row.jammi_call
        assert row.bib_keys, f"row {row.recipe!r} cites no references"
