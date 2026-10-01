"""The citation map's two rigor contracts, enforced as tests.

The K-bridge thesis is "one recipe = one equation = one canon line". That is only
honest if the map cannot dangle: every reference must resolve to a real
``references.bib`` entry, and every recipe's ``jammi_call`` must be a verb that
actually exists on the pinned engine (the grounded API reference). These tests
fail the build if either contract breaks — so a citation map row can never cite a
paper that is not in the bibliography or a Jammi verb that does not exist.
"""

from __future__ import annotations

from jammi_cookbook.citation_map import CITATION_MAP, bibliography_keys, unresolved


def test_every_citation_and_every_call_resolves():
    """No dangling @cite, and no row cites a verb the pinned engine lacks."""
    assert bibliography_keys(), "references.bib defines no entries"
    assert unresolved() == []


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
