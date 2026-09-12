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

  4. **The waiver allowlist (`no_consumer_names_allowlist.txt`).** #508
     amendment: a finding a human has already ruled MECHANISM (not governance;
     no consumer name) used to have exactly two exits — rename the identifier to
     dodge the grep, or admin-merge past every gate in the job — both worse than
     the finding. A reviewed row in `no_consumer_names_allowlist.txt` (that
     file's own header carries the full 5-field, 7-rot-rule schema) is the
     annotate path: it is scoped to the exact `(identifier, declaring_path)`
     pair a finding names — never the identifier alone, so one row can never
     amnesty a whole class — and it is re-verified for rot on every run, fail-
     closed. A waived finding still PRINTS, as an informational line naming the
     row and its ruling, so it never goes silent. The allowlist file is in
     `swarm.yml`'s human-amend-only glob: adding a row costs an admin-merge once
     per identifier, same discipline as every other reviewed waiver in this
     repo, but the ruling then lands in the tree instead of in a merge button.

Scope is engine RUNTIME paths only (`crates/**` code / config / fixtures, plus the
workspace `Cargo.toml` / `.cargo`) — never `docs/**` or `.claude/**` prose, which
legitimately *names* these anti-patterns to forbid them.

Fail-closed: any un-waived finding, any malformed allowlist row, or any rotted
allowlist row is a non-zero exit. Every un-waived finding is labelled ADVISORY —
a generic tripwire has false positives, so the exit code says "a human must look",
not "this is definitely a violation" — but a rotted or malformed allowlist row is
a HARD fail, never advisory (a waiver that no longer says what it claims to say
is worse than no waiver at all). Expected result on a clean engine tree with a
clean allowlist: pass.

Run: `python3 ci/scripts/check_no_consumer_names.py`
Hermetic: reads the working tree + `git diff` + `git merge-base`; no network, no build.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
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

# (4) The waiver allowlist — see that file's own header for the full schema.
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "no_consumer_names_allowlist.txt"

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


def added_crate_lines_with_paths(base: str) -> list[tuple[str, str]]:
    """`[(file_path, added_line_text), ...]` for every added (`+`) line under
    `crates/` in `git diff <base>...HEAD` — file-path-attributed (round for
    #508: the allowlist matches `(identifier, declaring_path)` PAIRS, so a
    finding needs to know which file it came from, not just its raw text).
    Tracked via the diff's own `+++ b/<path>` hunk headers; a deleted file's
    `+++ /dev/null` clears the current file (no added lines can attribute to
    it).
    """
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "crates/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    pairs: list[tuple[str, str]] = []
    current_file: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            raw_path = line[len("+++ ") :]
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("+") and current_file is not None:
            pairs.append((current_file, line[1:]))
    return pairs


# --------------------------------------------------------------------------- #
# (4) waiver allowlist — parsing + the 7 mechanically-checked rot rules.
# See `no_consumer_names_allowlist.txt`'s own header for the field schema.
# --------------------------------------------------------------------------- #
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ISSUE_OR_PR_REF_RE = re.compile(r"^#?[1-9][0-9]*$")
MIN_REASON_LEN = 20
BARE_LINE_CITATION_RE = re.compile(r"\bline\s*#?\d+\b|\bL\d+\b|:\d+\s*$", re.IGNORECASE)
ALLOWLIST_FIELD_COUNT = 5


@dataclass(frozen=True)
class AllowlistRow:
    identifier: str
    declaring_path: str
    ruling_sha: str
    ruling_ref: str
    reason: str
    line_no: int


def load_allowlist(path: Path = ALLOWLIST_PATH) -> tuple[list[AllowlistRow], list[str]]:
    """Returns `(rows, parse_failures)`. A missing file is not itself a
    failure (the allowlist starts, and may stay, empty) — `#`-prefixed and
    blank lines are ignored. A malformed row (wrong field count) and an
    exact duplicate `(identifier, declaring_path)` PAIR (rule 7 — a row
    waives that pair, never the identifier alone) are both hard parse
    failures naming the line; a duplicate's SECOND occurrence is dropped
    from `rows` (the first stands), never silently merged.
    """
    if not path.is_file():
        return [], []
    rows: list[AllowlistRow] = []
    failures: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != ALLOWLIST_FIELD_COUNT:
            failures.append(
                f"{path.name}:{line_no}: malformed row (expected {ALLOWLIST_FIELD_COUNT} "
                f"TAB-delimited fields <identifier><TAB><declaring_path><TAB><ruling_sha><TAB>"
                f"<ruling_ref><TAB><reason>, got {len(fields)}): {raw!r}"
            )
            continue
        identifier, declaring_path, ruling_sha, ruling_ref, reason = (f.strip() for f in fields)
        key = (identifier, declaring_path)
        if key in seen:
            failures.append(
                f"{path.name}:{line_no}: duplicate row for (identifier, declaring_path) = "
                f"{key!r} (rule 7 — already present at line {seen[key]}; a row waives the pair, "
                "never the identifier alone, so a second site needs its OWN row, not a repeat)"
            )
            continue
        seen[key] = line_no
        rows.append(AllowlistRow(identifier, declaring_path, ruling_sha, ruling_ref, reason, line_no))
    return rows, failures


def _public_idents_in_file(path: Path) -> set[str]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return set()
    return set(PUB_DECL_RE.findall(text))


def _identifier_appears_anywhere(identifier: str) -> bool:
    """Rule 5: `identifier` appears (whole-word) SOMEWHERE in the scanned
    engine-runtime tree — never merely at `declaring_path` (a rename may
    have moved it, which rule 1 catches separately; this rule only asks
    "does this identifier still exist at all").
    """
    pattern = re.compile(rf"\b{re.escape(identifier)}\b")
    for path in iter_scan_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            return True
    return False


def _is_ancestor(sha: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _resolve_ruling_ref(ref: str) -> str | None:
    """Returns a failure message if `ref` does not resolve, else None.

    Either a bare issue/PR number (`#123`/`123`) — accepted on FORMAT alone;
    this gate is hermetic (no network), so it cannot dial GitHub to confirm
    the issue carries a verdict, only that the citation is well-formed — or
    a `<doc-path>#<heading>` pair, resolved the same substring way
    `check_constitution_anchors.py`'s `doc_heading` anchor kind resolves
    (the heading text must literally appear in the doc).
    """
    if ISSUE_OR_PR_REF_RE.match(ref):
        return None
    if "#" not in ref:
        return (
            f"`{ref}` does not resolve — expected an issue/PR number (`#123`) or a "
            "`<doc-path>#<heading>` citation"
        )
    rel_path, heading = ref.split("#", 1)
    doc_path = REPO_ROOT / rel_path
    if not doc_path.is_file():
        return f"doc citation file not found: `{rel_path}`"
    try:
        text = doc_path.read_text(errors="ignore")
    except OSError:
        return f"doc citation file unreadable: `{rel_path}`"
    if heading not in text:
        return f"heading `{heading}` not found in `{rel_path}`"
    return None


def _reason_leads_with_governance_verb(reason: str) -> str | None:
    """The reason's OWN first word, lowercased, starts with a governance
    verb stem (a cheap, generic conjugation match — `registers`/`registered`
    both start with `register` — never an exhaustive grammar). This refuses
    a reason that merely ECHOES the tripwire's own stem back
    (`"Registers a UDF"` restates what the identifier's name already says)
    rather than explaining, in DIFFERENT mechanism vocabulary, what state
    the identifier actually changes.
    """
    first = re.match(r"[A-Za-z]+", reason.strip())
    if not first:
        return None
    word = first.group(0).lower()
    for verb in GOVERNANCE_VERBS:
        if word.startswith(verb):
            return verb
    return None


def _reason_rot_message(reason: str) -> str | None:
    stripped = reason.strip()
    if len(stripped) < MIN_REASON_LEN:
        return (
            f"reason is trivially short ({len(stripped)} chars < {MIN_REASON_LEN}) — must name, "
            "in mechanism vocabulary, what state the identifier changes"
        )
    verb = _reason_leads_with_governance_verb(stripped)
    if verb is not None:
        return (
            f"reason itself leads with governance-verb stem `{verb}` as its main verb — a "
            "reason that merely restates the tripwire's own stem is not a mechanism explanation"
        )
    if BARE_LINE_CITATION_RE.search(stripped):
        return "reason carries a bare line-number citation instead of a mechanism sentence"
    return None


def check_allowlist_rot(rows: list[AllowlistRow]) -> list[str]:
    """Rules 1-6, re-verified on EVERY run, every one a hard FAIL — a row
    that no longer says what it claims to say is worse than no waiver at
    all (rule 7 is enforced at load time — see `load_allowlist`).
    """
    findings: list[str] = []
    for row in rows:
        loc = f"{ALLOWLIST_PATH.name}:{row.line_no}"
        decl_path = REPO_ROOT / row.declaring_path
        if not decl_path.is_file():
            findings.append(
                f"{loc}: declaring_path `{row.declaring_path}` does not exist (rule 2) — "
                "delete or fix the row"
            )
        elif row.identifier not in _public_idents_in_file(decl_path):
            findings.append(
                f"{loc}: identifier `{row.identifier}` no longer matches a public declaration "
                f"in `{row.declaring_path}` (rule 1) — a rename must re-earn its ruling"
            )
        if not GIT_SHA_RE.match(row.ruling_sha):
            findings.append(
                f"{loc}: ruling_sha `{row.ruling_sha}` is not a well-formed 40-hex sha (rule 3)"
            )
        elif not _is_ancestor(row.ruling_sha):
            findings.append(
                f"{loc}: ruling_sha `{row.ruling_sha}` is not an ancestor of HEAD (rule 3)"
            )
        ref_failure = _resolve_ruling_ref(row.ruling_ref)
        if ref_failure is not None:
            findings.append(f"{loc}: ruling_ref {ref_failure} (rule 4)")
        if not _identifier_appears_anywhere(row.identifier):
            findings.append(
                f"{loc}: identifier `{row.identifier}` appears nowhere in the crates tree "
                "(rule 5) — dead waiver, delete the row"
            )
        reason_failure = _reason_rot_message(row.reason)
        if reason_failure is not None:
            findings.append(f"{loc}: {reason_failure} (rule 6)")
    return findings


def _waiver_line(kind: str, identifier: str, path: str, row: AllowlistRow) -> str:
    return (
        f"WAIVED ({kind}): `{identifier}` in `{path}` — {ALLOWLIST_PATH.name}:{row.line_no}, "
        f"ruling {row.ruling_ref} ({row.ruling_sha[:12]}): {row.reason}"
    )


def check_governance_tripwire(rows: list[AllowlistRow]) -> tuple[list[str], list[str]]:
    """(1) NEW public governance-verb identifiers in the diff (advisory,
    unless waived). Returns `(findings, waived_lines)`.
    """
    base = resolve_diff_base()
    if base is None:
        print(
            "no-consumer-names: no diff base available "
            "(SWARM_DIFF_BASE / origin/<base> / origin/main / main) — "
            "skipping the diff-scoped governance-verb tripwire.",
            file=sys.stderr,
        )
        return [], []

    allowlist_by_key = {(r.identifier, r.declaring_path): r for r in rows}
    findings: list[str] = []
    waived: list[str] = []
    for file_path, line in added_crate_lines_with_paths(base):
        for ident in PUB_DECL_RE.findall(line):
            verb = governance_stem(ident)
            if verb is None:
                continue
            row = allowlist_by_key.get((ident, file_path))
            if row is not None:
                waived.append(_waiver_line("governance-verb", ident, file_path, row))
                continue
            findings.append(
                f"ADVISORY: new public identifier `{ident}` in `{file_path}` has governance-verb "
                f"stem `{verb}` — confirm it is open-core MECHANISM "
                "(list/describe/delete/federation), not platform GOVERNANCE "
                "(governance is platform-owned; LESSONS L24). If a human has already ruled this "
                f"mechanism, add a reviewed row to {ALLOWLIST_PATH.name} citing the ruling — "
                "never rename a correct identifier to dodge this tripwire."
            )
    return findings, waived


def check_leak_smells(rows: list[AllowlistRow]) -> tuple[list[str], list[str]]:
    """(2) philosophy leak-smell tokens + (3) optional out-of-band denylist.
    Returns `(findings, waived_lines)` — the allowlist scopes ONLY the
    leak-smell leg (2); the out-of-band denylist (3) matches literal
    consumer NAMES supplied out-of-band per box and is never waivable in a
    committed file (a committed consumer name is the bug this gate exists
    to prevent).
    """
    denylist: list[str] = []
    if LOCAL_DENYLIST.is_file():
        denylist = [
            ln.strip()
            for ln in LOCAL_DENYLIST.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    allowlist_by_key = {(r.identifier, r.declaring_path): r for r in rows}
    findings: list[str] = []
    waived: list[str] = []
    for path in iter_scan_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for token in LEAK_SMELL_TOKENS:
            if token in text:
                row = allowlist_by_key.get((token, rel))
                if row is not None:
                    waived.append(_waiver_line("leak-smell", token, rel, row))
                    continue
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
    return findings, waived


def main() -> int:
    allowlist_rows, allowlist_parse_failures = load_allowlist()
    rot_findings = check_allowlist_rot(allowlist_rows) if not allowlist_parse_failures else []
    gov_findings, gov_waived = check_governance_tripwire(allowlist_rows)
    leak_findings, leak_waived = check_leak_smells(allowlist_rows)

    # A waived finding still PRINTS — visible in every CI log, never silent.
    for line in gov_waived + leak_waived:
        print(line)

    findings = allowlist_parse_failures + rot_findings + gov_findings + leak_findings

    if findings:
        print("no-consumer-names: FINDINGS", file=sys.stderr)
        for f in findings:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\nno-consumer-names: {len(findings)} finding(s). A governance-verb/leak-smell "
            "finding is a generic tripwire, not a proven violation — confirm mechanism-not-"
            f"governance and no consumer name, then either resolve it, or add a reviewed row to "
            f"{ALLOWLIST_PATH.name} (see that file's own header for the schema) citing the "
            "ruling. A malformed or rotted allowlist row is a hard fail on its own, independent "
            "of any live finding.",
            file=sys.stderr,
        )
        return 1

    print(
        "no-consumer-names: OK — no governance-verb leak in the diff, no philosophy "
        "leak-smell token in the engine tree, allowlist clean."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
