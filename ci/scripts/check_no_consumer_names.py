#!/usr/bin/env python3
"""The mechanical half of the discipline test: flag platform pull leaking into the engine.

Jammi is an engine of generic primitives that **names no consumer** and owns
*mechanism*, not *governance* (LESSONS L24 / the model-catalog boundary: the
registry mechanism — list/describe/delete — is open-core; lifecycle governance —
promote/retire/approve — is platform-owned, lives in the consumer's repo). This
gate is the cheap, generic backstop *behind* the LLM `discipline-test-auditor`; it
asserts a class, never a bug-signature, and it hardcodes **no consumer name** —
that would itself name a consumer inside the engine repo.

It has no proper names of its own. It uses only GENERIC patterns:

  1. **Governance-verb-stem tripwire (diff-scoped).** In the change under review
     (`git diff <base>...HEAD` over `crates/`), any NEW `pub` identifier whose
     leading stem is a governance verb (promote / retire / register / approve /
     gate / stage / transition) is surfaced. Governance is platform-owned, so a
     newly-public governance verb on the engine's API surface is a smell to
     confirm. Scoped to the diff because these stems already have legitimate
     open-core uses in the tree (source/UDF *registration*; benchmark SLO
     *gates*) — the tripwire questions what a change *adds*, and a human confirms
     it is mechanism, not governance. Advisory.

  2. **Philosophy leak-smell tokens (whole-tree).** The engine runtime tree
     (`crates/**` + workspace config) is grepped for the specific identifiers the
     philosophy calls out as boundary breaks: a raw-embedding `get_vector` verb
     (embeddings are consumed through `search`), an audit/lifecycle wrapper
     `LifecycleTable`, and actor/signature governance hooks `record_actor` /
     `sign_with`. These are specific enough to carry no open-core meaning, so a
     whole-tree hit is a real leak. Advisory.

  3. **Out-of-band denylist (optional, never committed).** If a gitignored local
     file `ci/scripts/.consumer_names.local` exists, its lines are treated as
     literal names to grep for in `crates/**`. This is the *only* way a concrete
     consumer name enters the check, and it is supplied out-of-band per box —
     never committed, because a committed consumer name is itself the bug this
     gate exists to prevent.

Scope is engine RUNTIME paths only (`crates/**` code / config / fixtures, plus the
workspace `Cargo.toml` / `.cargo`) — never `docs/**` or `.claude/**` prose, which
legitimately *names* these anti-patterns to forbid them.

Fail-closed: any finding is a non-zero exit. Every finding is labelled ADVISORY —
a generic tripwire has false positives, so the exit code says "a human must look",
not "this is definitely a violation". Expected result on a clean engine tree: pass.

Run: `python3 ci/scripts/check_no_consumer_names.py`
Hermetic: reads the working tree + `git diff`; no network, no build.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Engine runtime roots. `crates/**` is the engine's code + fixtures; the workspace
# manifest and cargo config are its runtime config. Deliberately excludes docs/
# and .claude/ prose (which name these anti-patterns to forbid them) and the CI
# gate scripts (which name the forbidden tokens as their own patterns).
SCAN_FILE_ROOTS = [REPO_ROOT / "Cargo.toml", REPO_ROOT / ".cargo"]
SCAN_TREE_ROOTS = [REPO_ROOT / "crates"]

SKIP_DIRS = {"target", ".git", "node_modules", "__pycache__"}
# Binary / non-source extensions we never grep.
BINARY_EXT = {
    ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".safetensors", ".onnx",
    ".gz", ".zip", ".tar", ".wav", ".mp3", ".flac", ".pt", ".npy", ".npz", ".ico",
    ".pdf", ".woff", ".woff2", ".ttf",
}

# (1) Governance verbs. A NEW public identifier whose stem is one of these is
# surfaced for confirmation (mechanism vs governance). Generic — names no consumer.
GOVERNANCE_VERBS = (
    "promote", "retire", "register", "approve", "gate", "stage", "transition",
)

# (2) Philosophy leak-smell tokens — specific boundary-break identifiers.
LEAK_SMELL_TOKENS = ("get_vector", "LifecycleTable", "record_actor", "sign_with")

# (3) Optional out-of-band denylist (gitignored, never committed).
LOCAL_DENYLIST = REPO_ROOT / "ci" / "scripts" / ".consumer_names.local"

# A `pub` item declaration and its identifier.
PUB_DECL_RE = re.compile(
    r"pub(?:\s*\([^)]*\))?\s+(?:async\s+)?"
    r"(?:unsafe\s+)?(?:fn|struct|enum|trait|type|const|static|mod)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def is_source_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in BINARY_EXT


def iter_scan_files():
    """Yield every engine-runtime source file in scope."""
    for root in SCAN_FILE_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            for p in root.rglob("*"):
                if p.is_dir() and p.name in SKIP_DIRS:
                    continue
                if is_source_file(p):
                    yield p
    for root in SCAN_TREE_ROOTS:
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            for child in current.iterdir():
                if child.is_dir():
                    if child.name not in SKIP_DIRS:
                        stack.append(child)
                elif is_source_file(child):
                    yield child


def governance_stem(ident: str) -> str | None:
    """Return the governance verb an identifier's stem matches, else None.

    The verb must be a whole leading token — the exact identifier, a snake_case
    head (`verb_…`), a CamelCase head (`Verb` + an uppercase hump), or a digit
    boundary (`stage2`). So `retire` matches `retire_model` / `RetireModel` but
    NOT `retirement`, and `register` matches `register_query` but not
    `registration`. (A naive case-insensitive char class would treat a lowercase
    continuation as a boundary and mis-flag `retirement`.)
    """
    low = ident.lower()
    for verb in GOVERNANCE_VERBS:
        if low == verb or low.startswith(verb + "_"):
            return verb
        cap = verb.capitalize()
        if ident.startswith(cap) and len(ident) > len(cap) and ident[len(cap)].isupper():
            return verb
        if low.startswith(verb) and len(ident) > len(verb) and ident[len(verb)].isdigit():
            return verb
    return None


def resolve_diff_base() -> str | None:
    """Resolve a git ref to diff against, or None if none is available."""
    candidates = [
        os.environ.get("SWARM_DIFF_BASE"),
        f"origin/{os.environ['GITHUB_BASE_REF']}" if os.environ.get("GITHUB_BASE_REF") else None,
        "origin/main",
        "main",
    ]
    for ref in candidates:
        if not ref:
            continue
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return ref
    return None


def added_crate_lines(base: str) -> list[str]:
    """Added (`+`) lines under `crates/` in `git diff <base>...HEAD`."""
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "crates/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def check_governance_tripwire() -> list[str]:
    """(1) NEW public governance-verb identifiers in the diff (advisory)."""
    base = resolve_diff_base()
    if base is None:
        print(
            "no-consumer-names: no diff base available "
            "(SWARM_DIFF_BASE / origin/<base> / origin/main / main) — "
            "skipping the diff-scoped governance-verb tripwire.",
            file=sys.stderr,
        )
        return []

    findings: list[str] = []
    for line in added_crate_lines(base):
        for ident in PUB_DECL_RE.findall(line):
            verb = governance_stem(ident)
            if verb is not None:
                findings.append(
                    f"ADVISORY: new public identifier `{ident}` has governance-verb "
                    f"stem `{verb}` — confirm it is open-core MECHANISM "
                    "(list/describe/delete/federation), not platform GOVERNANCE "
                    "(governance is platform-owned; LESSONS L24)."
                )
    return findings


def check_leak_smells() -> list[str]:
    """(2) philosophy leak-smell tokens + (3) optional out-of-band denylist."""
    denylist: list[str] = []
    if LOCAL_DENYLIST.is_file():
        denylist = [
            ln.strip()
            for ln in LOCAL_DENYLIST.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    findings: list[str] = []
    for path in iter_scan_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT)
        for token in LEAK_SMELL_TOKENS:
            if token in text:
                findings.append(
                    f"ADVISORY: philosophy leak-smell token `{token}` in `{rel}` — "
                    "a boundary-break identifier the philosophy forbids "
                    "(embeddings via `search`, no raw-vector/lifecycle/actor verb)."
                )
        for name in denylist:
            if re.search(rf"\b{re.escape(name)}\b", text):
                findings.append(
                    f"ADVISORY: out-of-band denylisted name `{name}` in `{rel}` — "
                    "a consumer name must not appear in the engine."
                )
    return findings


def main() -> int:
    findings = check_governance_tripwire() + check_leak_smells()

    if findings:
        print("no-consumer-names: FINDINGS (advisory — a human must confirm)", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\nno-consumer-names: {len(findings)} advisory finding(s). Each is a "
            "generic tripwire, not a proven violation — confirm mechanism-not-"
            "governance and no consumer name, then resolve or annotate.",
            file=sys.stderr,
        )
        return 1

    print(
        "no-consumer-names: OK — no governance-verb leak in the diff, no philosophy "
        "leak-smell token in the engine tree."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
