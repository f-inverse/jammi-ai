#!/usr/bin/env python3
"""Swarm ownership bijection gate — hermetic, static, no network, no build.

Asserts the engine swarm's domain manifests form a TOTAL PARTITION of every
tracked source path under the covered roots:

  (1) TOTAL COVERAGE   — every enumerated file matches exactly one owner's
                         `owns:` globs. An unowned file is a P0.
  (2) NO DOUBLE-OWNER  — a file matched by two owners is a P0, EXCEPT the
                         explicit shared-declaration class (`lib.rs`,
                         `Cargo.toml`, `error.rs` under `crates/`) which is
                         owned by `docs-ci` and may also be matched by the one
                         domain agent whose crate it lives in.

The owner set is read from the `owns:` frontmatter of every `.claude/agents/*.md`
card. This encodes the principle (ownership is a strict partition; an unowned
path is a P0 — LESSONS §T), not any single instance: it walks the filesystem and
proves totality over whatever tree exists today.

Fails closed: any parse error, any unowned path, or any illegal double-ownership
exits non-zero and names the offending path(s).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"

# Roots whose tracked files the swarm must partition. Tracked repo-root files
# (paths with no '/') are also in scope. Everything else (e.g. examples/,
# tests/, deploy/, .cargo/) is out of the covered surface by design.
COVERED_DIRS = ("crates", "cookbook", "docs", "ci", ".github", ".claude", ".jammi")

# The shared-declaration class: module roots, manifests, and error taxonomies
# under crates/ are lead/`docs-ci`-owned and exempt from domain write-ownership
# (ARCHITECTURE §2.2, §6). A domain glob will also match them; that specific
# overlap is the one permitted double-match.
SHARED_BASENAMES = frozenset({"lib.rs", "Cargo.toml", "error.rs"})
SHARED_OWNER = "docs-ci"


class GateError(Exception):
    """Any fail-closed condition (parse failure, unowned, double-owned)."""


# --------------------------------------------------------------------------- #
# glob matching: `**` spans path segments (incl. zero); `*` stays within one.
# --------------------------------------------------------------------------- #
def _glob_to_regex(glob: str) -> str:
    import re

    out: list[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == "*":
            if glob[i : i + 2] == "**":
                i += 2
                if glob[i : i + 1] == "/":
                    i += 1
                    out.append("(?:.*/)?")  # `**/` = zero or more segments
                else:
                    out.append(".*")  # trailing `**` = anything
            else:
                out.append("[^/]*")
                i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "^" + "".join(out) + "$"


def _compile(glob: str):
    import re

    return re.compile(_glob_to_regex(glob))


# --------------------------------------------------------------------------- #
# read `owns:` from every agent card's YAML frontmatter (no yaml dep)
# --------------------------------------------------------------------------- #
def _parse_owns(card: Path) -> list[str] | None:
    """Return the `owns:` glob list, or None if the card declares no ownership.

    Supports both flow (`owns: [a, b]`) and block list frontmatter forms.
    Raises GateError on a malformed frontmatter block.
    """
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise GateError(f"{card}: missing YAML frontmatter (no leading '---')")
    end = text.find("\n---", 3)
    if end == -1:
        raise GateError(f"{card}: unterminated YAML frontmatter")
    fm = text[3:end]

    lines = fm.splitlines()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("owns:"):
            continue
        rest = stripped[len("owns:") :].strip()
        if rest.startswith("[") and rest.endswith("]"):
            body = rest[1:-1].strip()
            if not body:
                raise GateError(f"{card}: `owns:` is an empty list")
            return [_unquote(tok.strip()) for tok in body.split(",") if tok.strip()]
        if rest == "":
            # block form: following `  - glob` lines
            globs: list[str] = []
            for follow in lines[idx + 1 :]:
                fs = follow.strip()
                if fs.startswith("- "):
                    globs.append(_unquote(fs[2:].strip()))
                elif fs == "":
                    continue
                else:
                    break
            if not globs:
                raise GateError(f"{card}: `owns:` block list is empty")
            return globs
        raise GateError(f"{card}: unparseable `owns:` value: {rest!r}")
    return None


def _unquote(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #
def _tracked_files() -> list[str]:
    res = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [p for p in res.stdout.split("\0") if p]


def _in_scope(path: str) -> bool:
    if "/" not in path:
        return True  # tracked repo-root file
    return path.split("/", 1)[0] in COVERED_DIRS


def _is_shared(path: str) -> bool:
    return path.startswith("crates/") and path.rsplit("/", 1)[-1] in SHARED_BASENAMES


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    try:
        owners = _load_owners()
        files = [p for p in _tracked_files() if _in_scope(p)]
        assign, unowned, double = _partition(files, owners)
    except GateError as exc:
        print(f"FAIL (fail-closed): {exc}", file=sys.stderr)
        return 1

    ok = True
    if unowned:
        ok = False
        print(f"P0 — {len(unowned)} UNOWNED path(s) (no owner's globs match):")
        for p in unowned:
            print(f"    {p}")
    if double:
        ok = False
        print(f"P0 — {len(double)} DOUBLE-OWNED path(s) (matched by >1 owner):")
        for p, ms in double:
            print(f"    {p}  <-  {', '.join(ms)}")

    # coverage summary
    print("\nSwarm ownership coverage:")
    counts: dict[str, int] = {name: 0 for name in owners}
    for _p, owner in assign.items():
        counts[owner] = counts.get(owner, 0) + 1
    for name in sorted(counts):
        print(f"    {name:<12} {counts[name]:>4} file(s)")
    print(f"    {'TOTAL':<12} {len(files):>4} file(s) in scope")

    if ok:
        print("\nPASS — total coverage, exactly-one owner per path (shared class → docs-ci).")
        return 0
    print("\nFAIL — ownership is not a total partition (see P0 paths above).")
    return 1


def _load_owners() -> dict[str, list]:
    if not AGENTS_DIR.is_dir():
        raise GateError(f"agents dir not found: {AGENTS_DIR}")
    owners: dict[str, list] = {}
    for card in sorted(AGENTS_DIR.glob("*.md")):
        globs = _parse_owns(card)
        if globs is None:
            continue  # a card that owns nothing (e.g. a read-only verifier)
        owners[card.stem] = [_compile(g) for g in globs]
    if SHARED_OWNER not in owners:
        raise GateError(f"no '{SHARED_OWNER}' owner declares `owns:` (needed for shared class)")
    return owners


def _partition(files: list[str], owners: dict[str, list]):
    assign: dict[str, str] = {}
    unowned: list[str] = []
    double: list[tuple[str, list[str]]] = []

    for f in files:
        matched = [name for name, regs in owners.items() if any(r.match(f) for r in regs)]
        if not matched:
            unowned.append(f)
            continue
        if _is_shared(f):
            if SHARED_OWNER not in matched:
                # shared file that docs-ci does not claim: broken shared class
                double.append((f, matched + [f"(missing {SHARED_OWNER})"]))
                continue
            domain = sorted(set(m for m in matched if m != SHARED_OWNER))
            if len(domain) > 1:
                double.append((f, matched))  # two domain agents both claim it
                continue
            assign[f] = SHARED_OWNER
        else:
            if len(matched) > 1:
                double.append((f, matched))
                continue
            assign[f] = matched[0]

    return assign, unowned, double


if __name__ == "__main__":
    sys.exit(main())
