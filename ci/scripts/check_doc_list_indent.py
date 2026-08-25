#!/usr/bin/env python3
"""Flag clippy's `doc_lazy_continuation` shape without needing to compile the
cuda-gated code that hides it from local clippy.

The lint fires when a `///`/`//!` doc line looks like a markdown list item
(`- `, `+ `, `* `, or `1. `) and the very next doc line is non-blank, is not
itself a list item, and starts flush-left (no indentation under the item
text) — i.e. clippy treats it as a "lazy continuation" of the list item
rather than ordinary prose, which is almost never what the author meant.

This script is a *report* generator, not a formatter: it does not rewrite
anything. Two invocation modes:

    check_doc_list_indent.py <path-glob> [<path-glob> ...]
        Scan the given files (or directories, recursively) for the pattern
        and print one "file:line: kind" hit per line to stdout. Exits 1 if
        any hit is found, 0 otherwise.

    check_doc_list_indent.py --self-test
        Run the bundled red/green fixtures through the detector and assert
        the expected hits. Exits 0 on success, 1 on failure. Does not touch
        the filesystem outside of temp files it cleans up itself.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

DOC_LINE_RE = re.compile(r"^\s*(///|//!)(.*)$")
# A markdown list item: "- text", "+ text", "* text", or "1. text".
LIST_ITEM_RE = re.compile(r"^(?P<indent>[-+*]\s+|\d+\.\s+)")
# A run of one or more backticks — a code-span delimiter. Toggled per run
# (not per character) so `` `foo` `` and `foo` both count as one open/close
# pair, matching how CommonMark treats backtick runs as a single delimiter.
BACKTICK_RUN_RE = re.compile(r"`+")


def _doc_content(line: str) -> str | None:
    """Return the text after the `///`/`//!` marker, or None if `line` is
    not a doc-comment line at all."""
    m = DOC_LINE_RE.match(line)
    if m is None:
        return None
    content = m.group(2)
    # Doc comments conventionally have exactly one space after the marker;
    # strip at most one so we measure *continuation* indentation, not the
    # marker's own spacing.
    if content.startswith(" "):
        content = content[1:]
    return content


def find_lazy_continuations(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return `(item_line_no, continuation_line_no, item_text)` triples,
    1-indexed, for every lazy-continuation hit in `lines`.

    Mirrors clippy's `doc_lazy_continuation`: once a doc line looks like a
    markdown list item, every following doc line up to the next blank line,
    the next list item, or the first *properly indented* line is treated as
    part of that item's body. Each such follow-on line that starts flush
    left (no indentation under the item text) is its own hit — clippy warns
    once per flush-left line, not just the first one after the item.

    A leading `+`/`-`/`*` is NOT a real list marker if it falls inside an
    inline code span left open by the previous line (e.g. a sentence
    wrapped as `` `scores\n + mask` `` across two doc lines): CommonMark's
    block parser only sees a fresh block start at a position that is not
    already inside open inline content, and empirically real `rustc`/
    `clippy` does not fire in that shape either (verified against a
    standalone crate reproduction). We track backtick-run parity across
    lines within one paragraph and suppress list/continuation detection on
    any line whose start is inside a code span opened earlier.
    """
    hits: list[tuple[int, int, str]] = []
    pending_item: tuple[int, str] | None = None  # (line_no, item_text)
    in_code_span = False  # backtick-run parity carried across doc lines

    for idx, raw in enumerate(lines):
        line_no = idx + 1
        content = _doc_content(raw)

        if content is None:
            # Non-doc line (code, blank source line, etc.) ends any doc
            # block and so ends any pending list-item context.
            pending_item = None
            in_code_span = False
            continue

        stripped = content.strip()

        if stripped == "":
            # A blank doc line is a paragraph break; clippy does not treat
            # what follows as a continuation of the prior list item, and a
            # well-formed doc comment does not leave a code span open
            # across a paragraph break either.
            pending_item = None
            in_code_span = False
            continue

        was_open = in_code_span
        for _ in BACKTICK_RUN_RE.finditer(content):
            in_code_span = not in_code_span

        if was_open:
            # This line's leading characters are inside a code span opened
            # on a prior line — not a fresh block-level position, so it can
            # neither start nor continue-flag a list item.
            pending_item = None
            continue

        is_list_item = LIST_ITEM_RE.match(content) is not None

        if is_list_item:
            # A fresh list item — either the first one, or the next item in
            # an intentional list. Either way it starts a new run.
            pending_item = (line_no, stripped)
            continue

        if pending_item is not None:
            if content == content.lstrip():
                # Flush left: lazy continuation, flag it, and stay in the
                # run — clippy keeps warning on each subsequent flush-left
                # line until the run ends.
                hits.append((pending_item[0], line_no, pending_item[1]))
            else:
                # Properly indented under the item text: author explicitly
                # continued the item body. This ends the flagged run.
                pending_item = None

    return hits


def scan_file(path: Path) -> list[tuple[int, int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return find_lazy_continuations(text.splitlines())


def _iter_target_files(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.rs")))
        elif p.exists():
            out.append(p)
        else:
            # Allow glob-style args passed by the shell already expanded;
            # if it truly doesn't exist, surface that as an error.
            print(f"error: no such file or directory: {arg}", file=sys.stderr)
            sys.exit(2)
    return out


_RED_FIXTURE = """\
/// Hand-composed eager backward reference on `device`: RopeFused + matmul
/// + SoftmaxLastDimFused + matmul, run under `Var`/`backward()` so
/// candle's own autograd (not this op's `bwd`) produces `dqkv` — the
/// independent reference is compared against. Returns `(out, dqkv)`.
fn example() {}
"""

_GREEN_FIXTURE = """\
/// Hand-composed eager backward reference on `device`: RopeFused, matmul,
/// SoftmaxLastDimFused, then matmul again, run under `Var`/`backward()` so
/// candle's own autograd (not this op's `bwd`) produces `dqkv`.
fn example() {}
"""

_GREEN_REAL_LIST_FIXTURE = """\
/// Steps:
/// - RopeFused rotates `q`/`k` in place.
///   Continuation indented under the item text: not a hit.
/// - SoftmaxLastDimFused normalizes the last axis.
/// - matmul contracts against `v`.
fn example() {}
"""

# A leading "+ " that is actually inside an inline code span opened on the
# previous line (a sentence wrapped mid code-span) — verified against real
# `clippy` (a standalone crate reproduction) to NOT fire doc_lazy_continuation.
_GREEN_OPEN_CODE_SPAN_FIXTURE = """\
/// A row is fully masked iff its OWN mask values (not `scores
/// + mask`) contain no exact `0.0`, i.e. `max_i mask[i] < 0.0`. This is
/// checked on `mask` ALONE, before ever reading `scores`.
fn example() {}
"""


def _self_test() -> int:
    failures: list[str] = []

    red_hits = find_lazy_continuations(_RED_FIXTURE.splitlines())
    expected_red = [(2, 3), (2, 4)]
    if [(h[0], h[1]) for h in red_hits] != expected_red:
        failures.append(
            f"red fixture: expected hits at {expected_red}, got "
            f"{[(h[0], h[1]) for h in red_hits]}: {red_hits}"
        )

    green_hits = find_lazy_continuations(_GREEN_FIXTURE.splitlines())
    if green_hits:
        failures.append(f"green fixture: expected 0 hits, got {green_hits}")

    green_list_hits = find_lazy_continuations(_GREEN_REAL_LIST_FIXTURE.splitlines())
    if green_list_hits:
        failures.append(
            f"green real-list fixture: expected 0 hits, got {green_list_hits}"
        )

    green_span_hits = find_lazy_continuations(
        _GREEN_OPEN_CODE_SPAN_FIXTURE.splitlines()
    )
    if green_span_hits:
        failures.append(
            f"green open-code-span fixture: expected 0 hits, got {green_span_hits}"
        )

    # Exercise the file-scanning path too, via a real temp file, so the I/O
    # plumbing (not just the pure detector) is under test.
    with tempfile.TemporaryDirectory() as td:
        red_path = Path(td) / "red.rs"
        red_path.write_text(_RED_FIXTURE, encoding="utf-8")
        file_hits = scan_file(red_path)
        if len(file_hits) != 2:
            failures.append(
                f"red fixture (via scan_file): expected 2 hits, got {file_hits}"
            )

    if failures:
        for f in failures:
            print(f"SELF-TEST FAIL: {f}", file=sys.stderr)
        return 1

    print("self-test OK: red fixture -> 2 hits, green fixtures -> 0 hits")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    if argv[0] == "--self-test":
        return _self_test()

    files = _iter_target_files(argv)
    any_hits = False
    for path in files:
        for item_line, cont_line, item_text in scan_file(path):
            any_hits = True
            print(
                f"{path}:{item_line}: doc list item {item_text!r} has a "
                f"flush-left (lazy) continuation at line {cont_line}"
            )

    return 1 if any_hits else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
