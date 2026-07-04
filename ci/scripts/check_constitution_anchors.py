#!/usr/bin/env python3
"""Resolve every TYPED code anchor in the engine constitution, fail-closed.

`docs/swarm/CONSTITUTION.md` is an index: each invariant row points at the code
anchor that realises it. An anchor that no longer resolves means the constitution
has silently drifted from the code it governs — this gate makes that drift a red
CI check instead of a human catch.

Anchors are **typed** so the check stays real for every invariant class (a bare
`path:symbol` went vacuous for boundary invariants, whose enforcement is a gate,
not a symbol). Each anchor names its resolver kind:

  - `rust_symbol:<path>:<Symbol>`  — `<Symbol>` still appears in `<path>` (the
    same "does the documented name still exist in the code" check `doc-parity`
    runs for the `ProducingDescriptor` enum).
  - `gate_script:<path>`           — the gate script `<path>` exists AND is
    referenced in `.github/workflows/swarm.yml` (a gate that is not wired into
    the swarm workflow does not enforce anything, so an unwired gate is a
    finding). The sentinel `gate_script:UNENFORCED` marks a boundary invariant
    that has NO wired mechanical gate — an honest hole in the enforcement
    surface, which this gate flags rather than hides.
  - `doc_heading:<path>#<heading>` — `<heading>` appears in `<path>`.
  - `discipline:<rationale>`        — a CONSCIOUS declaration that this invariant
    has no fail-closed mechanical gate and is enforced by an auditor agent / review
    (e.g. B4 one-binary, B6 workspace-atomicity — no clean gate exists; forcing a
    weak one would be theater). Resolves iff it carries a non-empty rationale. It is
    not a loophole: declaring it edits the constitution, tripping CONSTITUTION_TOUCHED
    (human-amend-only), so every "no gate here" is a reviewed decision.

Two completeness rules on top of per-anchor resolution:

  1. **Every boundary invariant (B*) must anchor to a `gate_script` OR a
     `discipline:<rationale>`.** Boundary invariants are the engine-not-platform
     line; their enforcement is either a fail-closed gate or an explicit,
     rationale-bearing discipline declaration. A boundary row resting on only a
     `rust_symbol`/`doc_heading` is a silent gap and fails.
  2. **`gate_script:UNENFORCED` on a boundary invariant is a non-zero finding** —
     a gate was claimed but does not exist. (Use `discipline:` to honestly declare
     "no gate exists"; use `UNENFORCED` never — it is the unresolved-hole sentinel.)
     This turns the anchor check into a completeness check over the enforcement
     surface (ARCHITECTURE §8).

Any unresolvable anchor, any UNENFORCED boundary invariant, or any parse failure
is a non-zero exit naming it. A silent pass on an uncomputable constitution would
defeat the gate, so it fails closed.

Run: `python3 ci/scripts/check_constitution_anchors.py`
Hermetic: reads only files in the working tree; no network, no build.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTITUTION = REPO_ROOT / "docs" / "swarm" / "CONSTITUTION.md"
SWARM_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "swarm.yml"

# A typed anchor token: `<kind>:<value>`, where <value> runs to the next
# whitespace, `;`, `|` (table cell wall) or backtick. Paths and rust symbols keep
# their internal `/` and `:`; only the leading kind is split off here.
ANCHOR_RE = re.compile(r"(rust_symbol|gate_script|doc_heading|discipline):([^\s;|`]+)")

# A constitution invariant row: a Markdown table row whose first bold cell is an
# invariant id (B* = boundary, K* = correctness).
ROW_ID_RE = re.compile(r"\*\*([BK]\d+)\*\*")

UNENFORCED = "UNENFORCED"


class AnchorError(Exception):
    """The constitution could not be parsed — the gate fails closed."""


@dataclass(frozen=True)
class Anchor:
    kind: str
    value: str


@dataclass(frozen=True)
class Row:
    invariant_id: str
    is_boundary: bool
    anchors: list[Anchor]


def parse_rows(text: str) -> list[Row]:
    """Parse invariant rows (id + typed anchors) from the constitution tables."""
    rows: list[Row] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        id_match = ROW_ID_RE.search(line)
        if id_match is None:
            continue
        invariant_id = id_match.group(1)
        anchors = [Anchor(k, v) for k, v in ANCHOR_RE.findall(line)]
        rows.append(
            Row(
                invariant_id=invariant_id,
                is_boundary=invariant_id.startswith("B"),
                anchors=anchors,
            )
        )
    if not rows:
        raise AnchorError(
            f"no invariant rows (`**B*/K***`) found in {CONSTITUTION.name} — "
            "the table shape changed or the file is empty"
        )
    return rows


def _read(path: Path) -> str:
    return path.read_text()


def resolve_rust_symbol(value: str) -> str | None:
    """Return a failure message if `<path>:<Symbol>` does not resolve, else None."""
    if ":" not in value:
        return f"malformed rust_symbol anchor `{value}` — expected `<path>:<Symbol>`"
    rel_path, symbol = value.rsplit(":", 1)
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return f"rust_symbol anchor file not found: `{rel_path}`"
    if symbol not in _read(path):
        return f"rust_symbol `{symbol}` no longer appears in `{rel_path}`"
    return None


def resolve_gate_script(value: str, swarm_text: str) -> str | None:
    """Return a failure message if the gate script is not a wired script, else None.

    `UNENFORCED` is handled by the caller (it is a boundary-completeness finding,
    not a per-anchor resolution failure) and never reaches here.
    """
    path = REPO_ROOT / value
    if not path.is_file():
        return f"gate_script `{value}` does not exist"
    if value not in swarm_text:
        return (
            f"gate_script `{value}` exists but is NOT referenced in "
            f"{SWARM_WORKFLOW.name} — an unwired gate enforces nothing"
        )
    return None


def resolve_doc_heading(value: str) -> str | None:
    """Return a failure message if `<path>#<heading>` does not resolve, else None."""
    if "#" not in value:
        return f"malformed doc_heading anchor `{value}` — expected `<path>#<heading>`"
    rel_path, heading = value.split("#", 1)
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return f"doc_heading anchor file not found: `{rel_path}`"
    if heading not in _read(path):
        return f"doc_heading `{heading}` not found in `{rel_path}`"
    return None


def check_row(row: Row, swarm_text: str) -> list[str]:
    """Resolve one row's anchors + completeness rules; return failure messages."""
    failures: list[str] = []

    if not row.anchors:
        failures.append(
            f"{row.invariant_id}: no typed anchor — every invariant must carry at "
            "least one `rust_symbol:`/`gate_script:`/`doc_heading:` anchor"
        )
        return failures

    kinds = {a.kind for a in row.anchors}

    # Completeness rule 1: a boundary invariant must anchor to a fail-closed
    # `gate_script` OR be CONSCIOUSLY declared `discipline:<rationale>`. It may not
    # rest on a rust_symbol/doc_heading alone — that would be a silent enforcement
    # gap. `discipline:` is not a loophole: declaring it edits the constitution,
    # which trips CONSTITUTION_TOUCHED (human-amend-only), so accepting that an
    # invariant has no mechanical gate is always a reviewed decision.
    if row.is_boundary and "gate_script" not in kinds and "discipline" not in kinds:
        failures.append(
            f"{row.invariant_id}: boundary invariant has no `gate_script` anchor and "
            "is not consciously declared `discipline:<rationale>` — its enforcement "
            "must be a fail-closed gate or an explicit, rationale-bearing declaration"
        )

    for anchor in row.anchors:
        if anchor.kind == "gate_script" and anchor.value == UNENFORCED:
            # Completeness rule 2: an unenforced boundary invariant is a finding.
            failures.append(
                f"{row.invariant_id}: gate_script:UNENFORCED — this boundary "
                "invariant has NO wired mechanical gate. The lead must wire one, "
                "or consciously accept the gap (do not require this check in "
                "branch protection until it is closed)."
            )
            continue

        if anchor.kind == "rust_symbol":
            failure = resolve_rust_symbol(anchor.value)
        elif anchor.kind == "gate_script":
            failure = resolve_gate_script(anchor.value, swarm_text)
        elif anchor.kind == "doc_heading":
            failure = resolve_doc_heading(anchor.value)
        elif anchor.kind == "discipline":
            # A conscious, human-gated declaration that this invariant has no
            # mechanical gate (it is enforced by an auditor agent / review).
            # It resolves iff it carries a non-empty rationale slug.
            failure = (
                None
                if anchor.value.strip()
                else "discipline anchor needs a rationale (`discipline:<why-no-gate>`)"
            )
        else:  # unreachable: ANCHOR_RE only matches the four kinds
            failure = f"unknown anchor kind `{anchor.kind}`"

        if failure is not None:
            failures.append(f"{row.invariant_id}: {failure}")

    return failures


def main() -> int:
    try:
        text = _read(CONSTITUTION)
    except OSError as exc:
        print(f"constitution-anchors: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    try:
        swarm_text = _read(SWARM_WORKFLOW)
    except OSError as exc:
        print(
            f"constitution-anchors: FAIL (uncomputable) — cannot read "
            f"{SWARM_WORKFLOW}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        rows = parse_rows(text)
    except AnchorError as exc:
        print(f"constitution-anchors: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    all_failures: list[str] = []
    for row in rows:
        all_failures.extend(check_row(row, swarm_text))

    if all_failures:
        print("constitution-anchors: FAIL", file=sys.stderr)
        for failure in all_failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            f"\nconstitution-anchors: {len(all_failures)} anchor(s) unresolved or "
            "unenforced across "
            f"{len(rows)} invariant(s). See above.",
            file=sys.stderr,
        )
        return 1

    print(
        f"constitution-anchors: OK — all anchors across {len(rows)} invariant(s) "
        "resolve; every boundary invariant is wired to a gate or consciously "
        "declared `discipline:`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
