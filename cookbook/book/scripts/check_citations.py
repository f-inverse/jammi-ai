#!/usr/bin/env python3
"""Fail the build on any dangling ``@cite`` in the book.

Quarto/pandoc only *warns* on a citation key it cannot resolve (it exits 0), so a
typo'd or removed reference would render as a broken `[?]` and ship silently. This
gate makes a dangling citation a hard failure: it collects every ``@key`` used in
the chapters (and the home page) and asserts each resolves to a ``references.bib``
entry. Run in CI before ``quarto render``.

Run: ``python scripts/check_citations.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BIB = _ROOT / "references.bib"

# A citation key in pandoc/Quarto markdown: @key, optionally inside [@key; @key2].
# Keys are letters/digits/_:-./ and must start after an @ that is not an email
# (preceded by whitespace, '[', ';', or line start). We also skip code spans.
_CITE = re.compile(r"(?:^|[\s\[;])@([A-Za-z][\w:.#$%&+?<>~/-]*)")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")


def _defined_keys() -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", _BIB.read_text()))


def _used_keys(text: str) -> set[str]:
    # Strip code (fenced + inline) so a literal '@something' in a cell is not a cite.
    text = _CODE_FENCE.sub("", text)
    text = _INLINE_CODE.sub("", text)
    return {m.group(1).rstrip(".,;") for m in _CITE.finditer(text)}


def main() -> int:
    defined = _defined_keys()
    if not defined:
        print("check-citations: references.bib defines no entries.")
        return 1

    sources = sorted(_ROOT.glob("chapters/**/*.qmd")) + [_ROOT / "index.qmd"]
    dangling: dict[str, set[str]] = {}
    used_total: set[str] = set()
    for src in sources:
        if not src.exists():
            continue
        used = _used_keys(src.read_text())
        used_total |= used
        missing = used - defined
        if missing:
            dangling[str(src.relative_to(_ROOT))] = missing

    if dangling:
        print("check-citations: dangling @cite(s) — not in references.bib:")
        for src, keys in dangling.items():
            print(f"  {src}: {sorted(keys)}")
        return 1

    print(f"check-citations: clean — {len(used_total)} citation key(s) used, "
          f"all resolve to references.bib ({len(defined)} entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
