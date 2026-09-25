"""Every frozen golden is one a chapter checks.

A render proves each metric a chapter checks is frozen (an unfrozen one raises);
this proves the converse, statically: a golden no chapter checks any more is a
measurement nobody takes, and is removed rather than kept as a number that
looks verified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jammi_cookbook import contracts

BOOK = Path(__file__).resolve().parents[1]
# The metric a check names: `assert_close(` or `golden(`, then a string or
# f-string literal, possibly on the next line.
_CHECK = re.compile(r"\b(?:assert_close|golden)\(\s*(f?)([\"'])(.+?)\2", re.S)


def _checked() -> list[re.Pattern[str]]:
    """One pattern per metric a chapter or the library checks; an f-string's
    replacement fields match any text."""
    sources = [*BOOK.glob("chapters/**/*.qmd"), *BOOK.glob("jammi_cookbook/*.py")]
    patterns = []
    for source in sources:
        for is_f, _, name in _CHECK.findall(source.read_text()):
            parts = re.split(r"\{[^}]*\}", name) if is_f else [name]
            patterns.append(re.compile(".+".join(map(re.escape, parts))))
    return patterns


def _frozen() -> dict[str, list[str]]:
    """Every frozen metric, by the golden files holding it."""
    frozen: dict[str, list[str]] = {}
    for path in sorted(contracts.GOLDENS.glob("*.json")):
        dataset = path.name.split(".", 1)[0]
        for key in json.loads(path.read_text()):
            frozen.setdefault(f"{dataset}.{key}", []).append(path.name)
    return frozen


def test_the_check_pattern_reads_literals_and_f_strings():
    source = 'contracts.assert_close(\n    f"media.{tower}.change", x)\ngolden("arxiv.tier01.p")'
    found = [(f, n) for f, _, n in _CHECK.findall(source)]
    assert found == [("f", "media.{tower}.change"), ("", "arxiv.tier01.p")]


def test_every_golden_is_checked_by_a_chapter():
    checked = _checked()
    orphans = {
        metric: files
        for metric, files in _frozen().items()
        if not any(p.fullmatch(metric) for p in checked)
    }
    assert not orphans, (
        "goldens no chapter checks — delete them, or restore the check that measured "
        f"them: {orphans}"
    )
