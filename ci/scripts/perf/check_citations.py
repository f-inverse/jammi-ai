#!/usr/bin/env python3
"""Re-resolve every `finetune_step.rs:<n>` / `grad_oracle.rs:<n>` PATH:LINE
citation under `crates/jammi-bench/**` and `ci/scripts/perf/**` against the
actual file content AT HEAD — advisory (i), round-2 audit fix on PR #372.

WHY THIS EXISTS: round 1 of this fix round "corrected" a stale citation in
`finetune_step.rs` (originally naming Rust source lines 253 through 264,
"fixed" to name line 290) by eyeballing the diff — but the code had ALREADY
moved again by the time that commit landed, so the "fixed" citation (line
290) was ALSO stale (the real line was 299) the moment it was committed.
`p1_softmax_scale_fold_ab.json` carried this SAME stale line-290 citation in
TWO fields, and `torch_finetune_step.py`/`README.md` independently carried a
DIFFERENT stale citation naming Rust source lines 112 and 233 together
(the second of those two line numbers pointed at unrelated LoRA-builder
code, not the VRAM baseline capture it was meant to cite) that neither round
noticed by eye either. A
citation that is only ever checked "by eye" at commit time is exactly the
kind of claim this repo's own `implementer-acceptance-clause` ("resolvable
citations") exists to stop being trusted on prose alone — this script is the
mechanical re-check, run every CI, not a one-time manual pass.

CONVENTION THIS SCRIPT ENFORCES: every citation in scope must be immediately
preceded by a backtick-quoted CODE IDENTIFIER (allowing only
whitespace/commas/parens/an apostrophe-s/the literal phrase "at the time of
writing" between the identifier's closing backtick and the citation itself —
see `_find_adjacent_identifier`). For example, this crate's own
`finetune_step.rs` module already carries a self-citation of exactly this
resolvable shape, right next to its own `peak.saturating_sub` expression —
see that function's test-suite doc for the literal text this script accepts
as a positive control. A BARE citation with no adjacent identifier (naming
two Rust source lines together with no code quoted next to either one — the
shape this fix replaces) is a HARD FAIL, not a lint warning — it is exactly
the shape that let a citation go stale
unnoticed, since there is nothing in the text itself a script (or a human)
can mechanically check it against. For each citation found, this script:

    1. Resolves `<file>.rs` to its one known location under
       `crates/jammi-bench/src/` (this script only knows about
       `finetune_step.rs`/`grad_oracle.rs` — the two files this crate's own
       fix-round dispatches have named; extending `_KNOWN_FILES` is a one-line
       change if a THIRD `.rs` file starts being cited this way).
    2. Checks the cited LINE NUMBER is in-bounds for that file's CURRENT
       (HEAD) line count.
    3. Checks the adjacent backtick-quoted identifier's text (whitespace-
       normalized) is a SUBSTRING of that exact line's CURRENT content
       (also whitespace-normalized) — this is what actually catches
       "the line number is in-bounds but points at different code now",
       which an in-bounds-only check (or eyeballing the diff) would miss.

Run: `python3 ci/scripts/perf/check_citations.py`
Hermetic: reads only files in the working tree; no network, no build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The only `.rs` files this script currently knows how to resolve citations
# against — both live under `crates/jammi-bench/src/`. Add a new mapping
# here (not a heuristic search) the day a THIRD file starts being cited this
# way, so a typo'd filename fails loudly (KeyError-shaped) rather than
# silently resolving to the wrong file.
_KNOWN_FILES = {
    "finetune_step.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "finetune_step.rs",
    "grad_oracle.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "grad_oracle.rs",
    # round-4 audit fold-in on PR #372: the determinant tables in
    # `grad_oracle.rs`/`ab_merge.py` cite dozens of `.py:<n>` lines in the
    # torch reference scripts — those citations were NEVER mechanically
    # re-checked (this script only knew about the two `.rs` files above),
    # which is exactly how the `.py` line-drift this round's own audit
    # caught went unnoticed.
    "torch_grad_oracle.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_grad_oracle.py",
    "torch_finetune_step.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_finetune_step.py",
}

# The two roots this advisory names, both searched recursively for every
# file (not just `.py`/`.json` -- a `.md` doc citation is just as resolvable
# and just as capable of going stale, see README.md's own citation this
# round fixed).
_SEARCH_ROOTS = (
    REPO_ROOT / "crates" / "jammi-bench",
    REPO_ROOT / "ci" / "scripts" / "perf",
)

# A known filename, a colon, and a line number -- optionally wrapped in a
# matched pair of backticks around the whole citation (both forms are used
# across this repo's own docs; see this script's own module doc for an
# example of each). Deliberately does NOT match the legacy
# comma-continuation shorthand this round's own fix retired (naming a
# second Rust source line by writing only a bare `,:<n>` after the first) --
# that shape now simply fails to match at all past the first number, which
# is intentional: an isolated `,:<n>` fragment, unattached to a
# `<filename>:` prefix, is not a citation this script recognizes at all, so
# a leftover comma-shorthand citation degrades to "only the first number is
# checked" rather than being silently mis-parsed as one unit.
def _citation_re() -> re.Pattern:
    """Built fresh from the CURRENT `_KNOWN_FILES` on every call (never
    compiled once at import time against whatever `_KNOWN_FILES` happened to
    be at that moment) -- `test_check_citations.py` monkeypatches
    `_KNOWN_FILES` per test to isolate one fixture at a time, and a
    module-load-time-frozen pattern would silently keep matching only the
    ORIGINAL file list regardless of that patch.
    """
    return re.compile(
        r"`?(?P<file>" + "|".join(re.escape(f) for f in _KNOWN_FILES) + r"):(?P<line>\d+)`?"
    )

# A backtick-quoted span -- candidate identifiers.
_IDENT_RE = re.compile(r"`([^`\n]{1,200})`")

# What is allowed to sit BETWEEN an identifier's closing backtick and the
# citation that names it: whitespace, commas, parens, an apostrophe-`s`, and
# the literal phrase "at the time of writing" (the one hedge phrase this
# repo's own self-citation convention uses, see `finetune_step.rs`'s own
# `vram_sampler_finish_reports_true_delta_not_floored_by_a_baseline_at_the_peak`
# test docstring). Anything else in the gap means the identifier just
# preceding is NOT actually attached to this citation.
_CONNECTOR_RE = re.compile(
    r"^(?:'s)?[\s,\(]*(?:at the time of writing[\s,\(]*)?$", re.IGNORECASE
)

_SEARCH_WINDOW = 300


# A wrapped Rust doc-comment (`/// ...`) or line comment (`// `/`# `)
# continuation onto the next source line -- collapsed to a single space
# before the connector check below, so a citation whose adjacent identifier
# sits one wrapped comment-line above it (e.g. `finetune_step.rs`'s own
# self-citation at the time of writing this script, split across two `///`
# lines) is not penalized for the comment SYNTAX carrying it; only the
# actual prose content of the gap is checked.
_COMMENT_CONTINUATION_RE = re.compile(r"\n[ \t]*(?:///|//|#)?[ \t]*")


def _find_adjacent_identifier(text: str, citation_start: int) -> str | None:
    window = text[max(0, citation_start - _SEARCH_WINDOW) : citation_start]
    idents = list(_IDENT_RE.finditer(window))
    if not idents:
        return None
    last = idents[-1]
    gap = _COMMENT_CONTINUATION_RE.sub(" ", window[last.end():])
    if _CONNECTOR_RE.match(gap):
        return last.group(1)
    return None


def _normalize(s: str) -> str:
    return " ".join(s.split())


class Violation:
    def __init__(self, source_path: Path, line_no: int, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        try:
            rel = self.source_path.relative_to(REPO_ROOT)
        except ValueError:
            # A path outside REPO_ROOT (e.g. a test fixture in a throwaway
            # tempdir) -- print it absolute rather than raising, so a test
            # driving `main()`/`check_file()` against a fixture never trips
            # on formatting a violation it deliberately provoked.
            rel = self.source_path
        return f"{rel}:{self.line_no}: {self.message}"


def _iter_source_files():
    seen = set()
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".json", ".md", ".rs", ".sh"):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def check_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations

    for m in _citation_re().finditer(text):
        cited_file = m.group("file")
        cited_line = int(m.group("line"))
        source_line_no = text.count("\n", 0, m.start()) + 1

        target_path = _KNOWN_FILES[cited_file]
        if not target_path.exists():
            violations.append(
                Violation(path, source_line_no, f"cites {cited_file} but {target_path} does not exist")
            )
            continue
        target_lines = target_path.read_text().splitlines()
        if cited_line < 1 or cited_line > len(target_lines):
            violations.append(
                Violation(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} but that file only has {len(target_lines)} lines",
                )
            )
            continue

        ident = _find_adjacent_identifier(text, m.start())
        if ident is None:
            violations.append(
                Violation(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} with no resolvable adjacent backtick-quoted "
                    "identifier -- a bare PATH:LINE citation cannot be mechanically re-checked "
                    "(this is exactly the shape that went stale unnoticed on PR #372); rewrite as "
                    f"`` `some_identifier`, {cited_file}:{cited_line} `` naming what is actually at "
                    "that line",
                )
            )
            continue

        cited_line_text = target_lines[cited_line - 1]
        if _normalize(ident) not in _normalize(cited_line_text):
            violations.append(
                Violation(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line is "
                    f"currently {cited_line_text.strip()!r} -- the citation is STALE (the code moved "
                    "since this was written); re-resolve it against the file at HEAD",
                )
            )
    return violations


def main() -> int:
    all_violations: list[Violation] = []
    checked = 0
    for path in _iter_source_files():
        checked += 1
        all_violations.extend(check_file(path))

    if all_violations:
        print("check-citations: FAIL", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(f"check-citations: {checked} file(s) scanned, all PATH:LINE citations resolve at HEAD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
