#!/usr/bin/env python3
"""A cheap syntactic tripwire for backend-specific SQL tokens in `jammi-db`.

This is a *first-pass* lint, not the closure mechanism for backend parity: it
catches an obvious, hand-typed backend-specific token before it ever reaches
review. The actual enforcement of Postgres/SQLite behavioral parity is the
required-in-CI Postgres it-matrix (the parameterized `crates/jammi-db/tests/it`
suite run against both backends) — a query that avoids every token below can
still diverge in behavior between backends, and only the it-matrix catches
that. This script only stops the cheap, obvious regression: someone hand-typing
a token that is valid SQL on exactly one backend and silently wrong (or a hard
error) on the other.

Forbidden tokens, each keyed to the backend it is exclusive to:

  - SQLite-only: `rowid` (the implicit pseudo-column — has no Postgres
    analog), `AUTOINCREMENT`, `PRAGMA`, `strftime(`, `glob(`.
  - Postgres-only (bidirectional parity — this crate must never hand-write a
    Postgres-only construct either): `ctid` (the physical-row identifier —
    not creation-ordered and has no SQLite analog).

Scope is `crates/jammi-db/src/**/*.rs` — the one crate that hand-writes SQL
against both backends. `//`/`///` line comments are stripped per line before
matching (the same exact-for-this-tree assumption `check_doc_parity.py` makes:
none of the scanned source carries `//` inside a string literal), so prose
that *discusses* these tokens — e.g. a doc comment explaining why a query does
NOT use `rowid` — does not trip the gate. This is scanning-by-comment-removal,
not a SQL parser: an over-approximation in the other direction (a real SQL
string that legitimately contains one of these substrings as literal data, not
syntax) is not currently a shape this crate has, so it is not handled — a
future false positive there is an acceptable, visible cost of a cheap lint.

Matching is word-bounded (`\\b…\\b` / `\\b…\\(`) so `_rowid` (a legitimate
quoted column identifier in `store/mutable/definition.rs`, not the SQLite
pseudo-column) does not trip the `rowid` check: a leading underscore is a word
character, so there is no `\\b` boundary between it and the following letters.

Run: `python3 ci/scripts/check_sqlite_isms.py`
Self-test (asserts the lint actually bites, and that it does not false-positive
on `_rowid`): `python3 ci/scripts/check_sqlite_isms.py --self-test`
Hermetic: reads only files in the working tree (or an in-memory fragment under
`--self-test`); no network, no build.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "crates" / "jammi-db" / "src"


@dataclass(frozen=True)
class Token:
    name: str
    pattern: re.Pattern[str]
    backend: str  # which backend this token is exclusive to


TOKENS = [
    Token("rowid", re.compile(r"(?i)\browid\b"), "SQLite-only"),
    Token("AUTOINCREMENT", re.compile(r"(?i)\bautoincrement\b"), "SQLite-only"),
    Token("PRAGMA", re.compile(r"(?i)\bpragma\b"), "SQLite-only"),
    Token("strftime(", re.compile(r"(?i)\bstrftime\s*\("), "SQLite-only"),
    Token("glob(", re.compile(r"(?i)\bglob\s*\("), "SQLite-only"),
    Token("ctid", re.compile(r"(?i)\bctid\b"), "Postgres-only"),
]


def strip_line_comments(text: str) -> str:
    """Drop `//`/`///` line comments so prose discussing a token isn't flagged.

    Exact for this tree today: none of the scanned source carries `//` inside a
    string literal (the same assumption `check_doc_parity.py`'s
    `_strip_rust_comments` makes for the enums it parses).
    """
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    line: str
    token: Token


def scan_text(text: str, file_label: str) -> list[Finding]:
    """Scan (comment-stripped) `text` for every forbidden token; report hits."""
    stripped = strip_line_comments(text)
    findings: list[Finding] = []
    for line_no, line in enumerate(stripped.splitlines(), start=1):
        for token in TOKENS:
            if token.pattern.search(line):
                findings.append(Finding(file_label, line_no, line.strip(), token))
    return findings


def scan_tree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not root.is_dir():
        raise SystemExit(f"check_sqlite_isms: scan root not found: {root}")
    for path in sorted(root.rglob("*.rs")):
        text = path.read_text(errors="ignore")
        rel = path.relative_to(REPO_ROOT)
        findings.extend(scan_text(text, str(rel)))
    return findings


def report(findings: list[Finding]) -> int:
    if not findings:
        print(
            "sqlite-isms: OK — no backend-specific SQL token found in "
            f"{SCAN_ROOT.relative_to(REPO_ROOT)}."
        )
        return 0

    print("sqlite-isms: FAIL", file=sys.stderr)
    for f in findings:
        print(
            f"  {f.file}:{f.line_no}: {f.backend} token `{f.token.name}` — {f.line}",
            file=sys.stderr,
        )
    print(
        "\nsqlite-isms: a backend-specific SQL token was hand-written in "
        "crates/jammi-db/src. This is a syntactic first-pass tripwire only — "
        "it does not certify parity; it just stops the obvious cheap regression. "
        "See the docstring of this script.",
        file=sys.stderr,
    )
    return 1


# --- self-test -----------------------------------------------------------

ROWID_FRAGMENT = """
pub(super) const BAD_QUERY: &str = r#"
SELECT rowid, name FROM topics ORDER BY rowid DESC
"#;
"""

UNDERSCORE_ROWID_FRAGMENT = """
let s = schema_with(&[("a", DataType::Int64), ("_rowid", DataType::Int64)]);
// a leading-underscore column named `_rowid` is not the SQLite pseudo-column
"""


def self_test() -> int:
    failures: list[str] = []

    hits = scan_text(ROWID_FRAGMENT, "<self-test: rowid fragment>")
    rowid_hits = [f for f in hits if f.token.name == "rowid"]
    if not rowid_hits:
        failures.append("self-test FAILED: a `rowid`-bearing fragment was not flagged")

    hits = scan_text(UNDERSCORE_ROWID_FRAGMENT, "<self-test: _rowid fragment>")
    if hits:
        failures.append(
            "self-test FAILED: `_rowid` (a legitimate identifier) was flagged: "
            + "; ".join(f"{f.token.name} at line {f.line_no}" for f in hits)
        )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("sqlite-isms self-test: FAIL", file=sys.stderr)
        return 1

    print("sqlite-isms self-test: OK — rowid fragment flagged, _rowid fragment clean.")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    return report(scan_tree(SCAN_ROOT))


if __name__ == "__main__":
    sys.exit(main())
