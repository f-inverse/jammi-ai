#!/usr/bin/env python3
"""The mechanical half of the discipline test: flag platform pull leaking into the engine.

Jammi is an engine of generic primitives that **names no consumer** and owns
*mechanism*, not *governance* (the model-catalog boundary: the
registry mechanism — list/describe/delete — is open-core; lifecycle governance —
promote/retire/approve — is platform-owned, lives in the consumer's repo). This
gate is the cheap, generic backstop behind human review; it asserts a class,
never a bug-signature, and it hardcodes **no consumer name** —
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
     it is mechanism, not governance. Advisory. Resolved against the REAL
     `symbol-index` `syn` parse (`ci/tools/symbol-index`), cross-referenced
     against the diff's own added-line-number set — never a regex re-scan of
     each added line's bare text, which cannot see a `pub fn` whose signature
     wraps across lines and can be fooled by matching text inside a string or
     comment a real parser is not.

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

  4. **The waiver allowlist (`no_consumer_names_allowlist.txt`).** A finding a
     maintainer has ruled MECHANISM (not governance; no consumer name) needs an
     exit better than renaming a correct identifier to dodge the grep. A
     reviewed row in `no_consumer_names_allowlist.txt` (that file's own header
     carries the full 5-field, 7-rot-rule schema) is the annotate path: it is
     scoped to the exact `(identifier, declaring_path)`
     pair a finding names — never the identifier alone, so one row can never
     amnesty a whole class — and it is re-verified for rot on every run, fail-
     closed. A waived finding still PRINTS, as an informational line naming the
     row and its ruling, so it never goes silent, and the ruling lands in the
     tree where a reviewer reads it.

Scope is engine RUNTIME paths only (`crates/**` code / config / fixtures, plus the
workspace `Cargo.toml` / `.cargo`) — never `docs/**` prose, which legitimately
*names* these anti-patterns to forbid them.

Fail-closed: any un-waived finding, any malformed allowlist row, or any rotted
allowlist row is a non-zero exit. Every un-waived finding is labelled ADVISORY —
a generic tripwire has false positives, so the exit code says "a human must look",
not "this is definitely a violation" — but a rotted or malformed allowlist row is
a HARD fail, never advisory (a waiver that no longer says what it claims to say
is worse than no waiver at all). Expected result on a clean engine tree with a
clean allowlist: pass.

Run: `python3 ci/scripts/check_no_consumer_names.py`
Reads the working tree + `git diff` + `git merge-base`; no network. NOT
build-free in the strict sense: the governance-verb tripwire (1) and the
allowlist's own rule-1 rot check both resolve against the REAL
`symbol-index` tool (`cargo run --release -p symbol-index`, a `syn` AST
parse) rather than a regex reader — a cold build pays once, cached the same
way every other cargo-invoking gate/agent in this repo caches (sccache /
CI cache).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SYMBOL_INDEX_CRATE = "symbol-index"


def build_symbol_index(roots: list[str], cwd: Path = REPO_ROOT) -> dict:
    """Runs the REAL `symbol-index` tool (`ci/tools/symbol-index`, a `syn`
    AST parse) over `roots` and returns the parsed JSON index. The SAME
    tool `check_plan_citations.py` resolves construct citations against —
    one indexer, not two independently-maintained regex readers (the
    retired `PUB_DECL_RE` this migration removes was the second). See
    that tool's own module doc for the full item shape and scope.
    `cargo run --release` so a cold build pays once; this function never
    overrides `CARGO_TARGET_DIR`/`RUSTC_WRAPPER` — the caller's own
    environment (if any) is inherited unchanged.
    """
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", SYMBOL_INDEX_CRATE, "--", *roots],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} failed (rc={proc.returncode}) over {roots}:\n"
            f"{proc.stderr.strip()[-4000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{SYMBOL_INDEX_CRATE} did not emit valid JSON on stdout ({exc}); "
            f"stderr tail:\n{proc.stderr.strip()[-2000:]}"
        ) from exc

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

# The CamelCase NOUN form of a governance verb is ALSO
# governance-shaped (`Retirement`, `RetirementPolicy`, `retirement_policy`)
# even though a plain participle/agentive/plural form of the SAME verb
# (`Retired`, `Retiring`, `Retires`, `Registered`) is legitimate engine
# vocabulary — measured directly against the real tree: a
# naive participle rule would newly flag Registered(54) Gated(29)
# Registering(3) Registers(4) Gates(2) Promoting/Promotes(1), all real,
# non-governance engine identifiers. A closed, per-verb table — never a
# generic "-tion/-ment" suffix rule, which would re-introduce exactly that
# false-positive class. `gate`→"Gating" is deliberately NOT a row here
# (ambiguous with the engine's own `Gated`/`Gates` vocabulary — `gate`
# keeps only its verb-head rule below); `stage`→"Staging" is likewise
# excluded (same reason); `transition` needs no row — it is already
# noun-shaped, so the plain verb-head rule below already matches the bare
# identifier `Transition`/`transition`.
GOVERNANCE_NOUNS = {
    "promote": "Promotion",
    "retire": "Retirement",
    "register": "Registration",
    "approve": "Approval",
}

# (2) Philosophy leak-smell tokens — specific boundary-break identifiers.
LEAK_SMELL_TOKENS = ("get_vector", "LifecycleTable", "record_actor", "sign_with")

# (3) Optional out-of-band denylist (gitignored, never committed).
LOCAL_DENYLIST = REPO_ROOT / "ci" / "scripts" / ".consumer_names.local"

# (4) The waiver allowlist — see that file's own header for the full schema.
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "no_consumer_names_allowlist.txt"

# A `pub` item declaration and its identifier are found through the REAL
# `symbol-index` `syn` parse (`build_symbol_index`, `_public_idents_in_file`),
# never a regex over raw text: a regex cannot see a declaration wrapped across
# lines and matches text inside string literals and comments.


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

    A verb WITH a row in `GOVERNANCE_NOUNS` also matches its own
    CamelCase NOUN form, and that noun's PLURAL, as a whole head token —
    the exact identifier, a snake_case head (`retirement_…`), or a
    CamelCase head (`Retirement` + an uppercase hump). The noun's
    SPELLING (not merely its prefix) is what excludes a participle/
    agentive/plural of the bare VERB: `Retired`/`Retiring`/`Retires`/
    `Registered` all diverge from `Retirement`/`Registration` within the
    shared verb root, so they never match this rule OR the plain
    verb-head rule above (which requires an uppercase/digit boundary
    immediately after the verb itself, absent here). Measured directly
    against the real tree (symbol-index parse of `crates`): the noun and
    plural widening together introduce 0 new findings over the bare
    verb-head rule, so neither reproduces the participle/agentive false-
    positive class the noun table itself was built to avoid.

    `transition` carries no `GOVERNANCE_NOUNS` row (it is already noun-
    shaped, so the verb-head rule above already matches the bare
    identifier `Transition`/`transition`) but its PLURAL, `Transitions`,
    still needs the same whole-head-token check the other verbs' nouns
    get — also measured at 0 new findings.
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
        noun = GOVERNANCE_NOUNS.get(verb)
        forms: tuple[str, ...] = ()
        if noun is not None:
            forms = (noun, noun + "s")
        elif verb == "transition":
            forms = (cap + "s",)  # "Transitions" -- the singular is already covered above
        for form in forms:
            form_low = form.lower()
            if low == form_low or low.startswith(form_low + "_"):
                return verb
            if ident.startswith(form) and len(ident) > len(form) and ident[len(form)].isupper():
                return verb
    return None


def _census_governance_findings(index: dict, nouns: dict[str, str] | None) -> set[tuple[str, str]]:
    """Every (identifier, path) `pub` item, TREE-WIDE (never diff-scoped —
    `check_governance_tripwire` stays diff-scoped; this is
    the self-test's own tree-wide oracle only), whose stem matches `governance_stem` under the
    given noun table: `nouns={}` reproduces the verb-only rule
    (temporarily empties `GOVERNANCE_NOUNS`, restored in `finally`, so this
    calls the REAL `governance_stem` rather than duplicating its logic —
    the two paths cannot drift apart by construction); `nouns=None` uses
    the real, committed table."""
    global GOVERNANCE_NOUNS
    real_nouns = GOVERNANCE_NOUNS
    GOVERNANCE_NOUNS = real_nouns if nouns is None else nouns
    try:
        found: set[tuple[str, str]] = set()
        for item in index["items"]:
            if not item["vis"].startswith("pub"):
                continue
            if governance_stem(item["name"]) is not None:
                found.add((item["name"], item["path"]))
        return found
    finally:
        GOVERNANCE_NOUNS = real_nouns


def resolve_diff_base() -> str | None:
    """Resolve a git ref to diff against, or None if none is available."""
    candidates = [
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


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_crate_lines_with_paths(base: str) -> list[tuple[str, int, str]]:
    """`[(file_path, new_file_line_no, added_line_text), ...]` for every
    added (`+`) line under `crates/` in `git diff <base>...HEAD` —
    file-path-attributed (the allowlist matches
    `(identifier, declaring_path)` PAIRS, so a finding needs to know which
    file it came from, not just its raw text) and line-number-attributed
    (so `check_governance_tripwire` can cross-reference a REAL parsed
    item's own declared line against this diff's added-line set, rather
    than re-parsing the line's own bare text). Tracked via the diff's own
    `+++ b/<path>` hunk headers and each hunk's `@@ -a,b +c,d @@` new-file
    starting line; a deleted file's `+++ /dev/null` clears the current
    file (no added lines can attribute to it). `--unified=0` means every
    non-`+++`/`@@` line in a hunk is an add, so the running new-line
    counter only needs to seed from `c` and increment on `+`.
    """
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD", "--", "crates/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    triples: list[tuple[str, int, str]] = []
    current_file: str | None = None
    new_line_no = 0
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            raw_path = line[len("+++ ") :]
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("@@ "):
            m = _HUNK_HEADER_RE.match(line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if line.startswith("+") and current_file is not None:
            triples.append((current_file, new_line_no, line[1:]))
            new_line_no += 1
    return triples


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


def _public_idents_in_file(rel_path: str, index: dict) -> set[str]:
    """Every `pub`(-any-form) item's bare NAME declared in `rel_path`
    (repo-relative), per the REAL `symbol-index` `syn` parse — never a
    regex reader over the file's own text (see the module docstring's own
    "PUB_DECL_RE retired" note). An impl method's own bare method name is
    included (an `impl` block's `pub fn` is a public declaration the same
    way a free `pub fn` is), matching what the retired regex also matched
    textually, now derived from a real AST instead.
    """
    return {
        it["name"]
        for it in index["items"]
        if it["path"] == rel_path and it["vis"].startswith("pub")
    }


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
    a `<doc-path>#<heading>` pair (the heading text must literally appear in
    the doc).
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

    The symbol index is built ONCE, here, and threaded through every row's
    own rule-1 check — never rebuilt per row (a real `cargo run` per row
    would be a needless multiplication of an already-fast, single-build
    cost).
    """
    findings: list[str] = []
    index = build_symbol_index(["crates"]) if rows else None
    for row in rows:
        loc = f"{ALLOWLIST_PATH.name}:{row.line_no}"
        decl_path = REPO_ROOT / row.declaring_path
        if not decl_path.is_file():
            findings.append(
                f"{loc}: declaring_path `{row.declaring_path}` does not exist (rule 2) — "
                "delete or fix the row"
            )
        elif row.identifier not in _public_idents_in_file(row.declaring_path, index):
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

    Cross-references the REAL symbol index (every `pub` item's own
    declared line, from a real `syn` parse) against the diff's own
    added-line-number set — never a regex re-scan of each added line's
    bare text (the retired `PUB_DECL_RE` approach), which could not see a
    `pub fn` whose signature wraps across lines, and could be fooled by a
    string/comment containing the same text a real parser is not.

    An un-waived finding here is what makes `main()` return 1 (fail the
    step) — `governance_stem`'s own oracle for whether a finding is
    CORRECT (a real governance-shaped identifier, never a false positive)
    is the self-test's tree-wide census (`_census_governance_findings`),
    never this diff-scoped function, which only ever sees whatever the
    CURRENT diff happens to add.
    """
    base = resolve_diff_base()
    if base is None:
        print(
            "no-consumer-names: no diff base available "
            "(origin/<base> / origin/main / main) — "
            "skipping the diff-scoped governance-verb tripwire.",
            file=sys.stderr,
        )
        return [], []

    added = added_crate_lines_with_paths(base)
    added_lines_by_file: dict[str, set[int]] = {}
    for file_path, line_no, _text in added:
        added_lines_by_file.setdefault(file_path, set()).add(line_no)
    if not added_lines_by_file:
        return [], []

    index = build_symbol_index(["crates"])
    allowlist_by_key = {(r.identifier, r.declaring_path): r for r in rows}
    findings: list[str] = []
    waived: list[str] = []
    for item in index["items"]:
        if not item["vis"].startswith("pub"):
            continue
        file_path = item["path"]
        added_set = added_lines_by_file.get(file_path)
        if not added_set or item["line"] not in added_set:
            continue
        ident = item["name"]
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
            "(governance is platform-owned). If a human has already ruled this "
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


# --------------------------------------------------------------------------- #
# self-test (one fixture per rot rule 1-7, plus the
# duplicate-pair vs. same-identifier-different-path distinction)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: object = "") -> None:
        if not cond:
            failures.append(f"{label}: {detail}")

    # The REAL, committed row this repo already ships -- reused as the
    # baseline for every mutation below, rather than a synthetic tempdir
    # tree: `register_content_hash_udf` genuinely exists at this exact
    # path, its ruling_sha is a genuine ancestor of HEAD, and the
    # identifier genuinely appears elsewhere in the crates tree (e.g.
    # `pinned_source_gate.rs`) -- so a mutation on ONE field
    # exercises exactly the rule that field governs, nothing else.
    good_row = AllowlistRow(
        identifier="register_content_hash_udf",
        declaring_path="crates/jammi-ai/src/query/content_hash_udf.rs",
        ruling_sha="c0e1faec00a3de0d59499801755e705579dbe202",
        ruling_ref="#508",
        reason=(
            "Installs a Datafusion scalar UDF into the query session's function catalog so SQL "
            "statements can invoke it by name."
        ),
        line_no=1,
    )

    # Positive control: the real, committed row must produce ZERO findings
    # -- proves this checker can find a genuinely correct row, not merely
    # reject everything handed to it.
    got = check_allowlist_rot([good_row])
    check("positive control (the real committed row)", got == [], got)

    # Rule 1: identifier no longer matches a public declaration at
    # declaring_path (a rename must re-earn its ruling).
    renamed = replace(good_row, identifier="register_content_hash_udf_renamed_xyz")
    got = check_allowlist_rot([renamed])
    check("rule 1 (renamed identifier)", any("(rule 1)" in g for g in got), got)

    # Rule 2: declaring_path does not exist.
    missing_path = replace(good_row, declaring_path="crates/jammi-ai/src/does_not_exist_xyz.rs")
    got = check_allowlist_rot([missing_path])
    check("rule 2 (missing declaring_path)", any("(rule 2)" in g for g in got), got)

    # Rule 3a: ruling_sha is not a well-formed 40-hex sha at all.
    bad_shape = replace(good_row, ruling_sha="not-a-real-sha")
    got = check_allowlist_rot([bad_shape])
    check("rule 3 (malformed ruling_sha)", any("(rule 3)" in g and "well-formed" in g for g in got), got)

    # Rule 3b: well-formed 40-hex sha that is NOT an ancestor of HEAD.
    non_ancestor = replace(good_row, ruling_sha="f" * 40)
    got = check_allowlist_rot([non_ancestor])
    check(
        "rule 3 (well-formed but non-ancestor ruling_sha)",
        any("(rule 3)" in g and "not an ancestor" in g for g in got),
        got,
    )

    # Rule 4: ruling_ref does not resolve (neither an issue/PR number nor a
    # `<doc-path>#<heading>` citation).
    bad_ref = replace(good_row, ruling_ref="not-a-real-citation")
    got = check_allowlist_rot([bad_ref])
    check("rule 4 (unresolvable ruling_ref)", any("(rule 4)" in g for g in got), got)

    # Rule 5: identifier appears NOWHERE in the scanned engine-runtime tree
    # -- a dead waiver.
    dead = replace(good_row, identifier="definitely_absent_identifier_98765_xyz")
    got = check_allowlist_rot([dead])
    check("rule 5 (dead identifier, appears nowhere)", any("(rule 5)" in g for g in got), got)

    # Rule 6a: reason is trivially short.
    short_reason = replace(good_row, reason="ok")
    got = check_allowlist_rot([short_reason])
    check(
        "rule 6 (trivially short reason)",
        any("(rule 6)" in g and "trivially short" in g for g in got),
        got,
    )

    # Rule 6b: reason itself leads with a governance-verb stem as its main
    # verb -- "confirmed, it's fine" phrased as a restatement, refused.
    verb_reason = replace(
        good_row,
        reason="Registers a new content-hash row so downstream code can look it up by name later.",
    )
    got = check_allowlist_rot([verb_reason])
    check(
        "rule 6 (reason leads with a governance verb)",
        any("(rule 6)" in g and "leads with governance-verb" in g for g in got),
        got,
    )

    # Rule 6c: reason carries a bare line-number citation instead of a
    # mechanism sentence.
    line_reason = replace(good_row, reason="Confirmed clean per the ruling discussion at content_hash_udf.rs:42")
    got = check_allowlist_rot([line_reason])
    check(
        "rule 6 (bare line-number citation)",
        any("(rule 6)" in g and "bare line-number citation" in g for g in got),
        got,
    )

    # Rule 7: an EXACT duplicate (identifier, declaring_path) PAIR is a
    # hard parse failure at load time (never seen by check_allowlist_rot
    # at all), and only the FIRST occurrence survives into `rows`.
    with tempfile.TemporaryDirectory() as td:
        allow_path = Path(td) / "allow.txt"
        allow_path.write_text(
            f"{good_row.identifier}\t{good_row.declaring_path}\t{good_row.ruling_sha}\t"
            f"{good_row.ruling_ref}\t{good_row.reason}\n"
            f"{good_row.identifier}\t{good_row.declaring_path}\t{good_row.ruling_sha}\t"
            "#999\tA second, differently-worded ruling for the identical pair.\n",
            encoding="utf-8",
        )
        rows, parse_failures = load_allowlist(allow_path)
        check(
            "rule 7 (duplicate pair is a hard parse failure)",
            any("rule 7" in f for f in parse_failures),
            parse_failures,
        )
        check("rule 7 (only the first occurrence survives into rows)", len(rows) == 1, rows)

    # Negative control: the SAME identifier at a DIFFERENT declaring_path
    # is NOT rule 7 -- the real committed allowlist carries exactly this
    # shape (register_content_hash_udf at two distinct paths) and must
    # load with zero parse failures.
    with tempfile.TemporaryDirectory() as td:
        allow_path = Path(td) / "allow.txt"
        allow_path.write_text(
            f"{good_row.identifier}\t{good_row.declaring_path}\t{good_row.ruling_sha}\t"
            f"{good_row.ruling_ref}\t{good_row.reason}\n"
            f"{good_row.identifier}\tcrates/jammi-ai/tests/it/pinned_source_gate.rs\t"
            f"{good_row.ruling_sha}\t#554\tA different site, its own row, not a duplicate pair.\n",
            encoding="utf-8",
        )
        rows, parse_failures = load_allowlist(allow_path)
        check(
            "same identifier, different declaring_path is NOT rule 7",
            parse_failures == [] and len(rows) == 2,
            (parse_failures, rows),
        )

    # governance_stem's per-verb noun morphology. A CamelCase
    # NOUN form of a governance verb is a match; a participle/agentive/
    # plural form of the SAME verb is not.
    check("noun pair: Retirement -> retire", governance_stem("Retirement") == "retire",
          governance_stem("Retirement"))
    check("noun pair: Registered -> None", governance_stem("Registered") is None,
          governance_stem("Registered"))
    check("noun pair: RegistrationTable -> register", governance_stem("RegistrationTable") == "register",
          governance_stem("RegistrationTable"))
    check("noun pair: Registers -> None", governance_stem("Registers") is None,
          governance_stem("Registers"))
    # The PLURAL noun forms are governance-shaped too.
    check("plural noun pair: Registrations -> register", governance_stem("Registrations") == "register",
          governance_stem("Registrations"))
    check("plural noun pair: Approvals -> approve", governance_stem("Approvals") == "approve",
          governance_stem("Approvals"))
    check("plural noun pair: Promotions -> promote", governance_stem("Promotions") == "promote",
          governance_stem("Promotions"))
    check("plural noun pair: RetirementsTable -> retire", governance_stem("RetirementsTable") == "retire",
          governance_stem("RetirementsTable"))
    check("plural noun pair: Transitions -> transition", governance_stem("Transitions") == "transition",
          governance_stem("Transitions"))

    # Tree-wide census — the oracle is the REAL symbol-index
    # parse of `crates`, never the diff-scoped tripwire (`check_
    # governance_tripwire` examines only pub items on the diff's OWN added
    # lines, so it cannot see whether the noun table would newly flag
    # something ALREADY in the tree). This self-test runs ONLY in ci.yml's
    # container-backed `symbol-index-gates` job (a toolchain is always
    # present there), so an index
    # build failure here is a genuine environment problem, not a
    # graceful-skip arm like this file's own missing-cargo advisory arm
    # elsewhere.
    census_index = build_symbol_index(["crates"])
    pub_items = [it for it in census_index["items"] if it["vis"].startswith("pub")]
    check(
        "census: pub item floor (committed literal)",
        len(pub_items) >= 6838,
        f"only {len(pub_items)} pub item(s) indexed tree-wide, expected >= 6838 "
        "(a shrink this large means the index itself is broken, not that the tree got smaller)",
    )
    verb_only = _census_governance_findings(census_index, nouns={})
    with_nouns = _census_governance_findings(census_index, nouns=None)
    new_from_nouns = with_nouns - verb_only
    check(
        "census: the noun table introduces 0 NEW tree-wide findings (committed literal)",
        len(new_from_nouns) == 0,
        f"the noun table newly flags {sorted(new_from_nouns)} tree-wide — either a genuine leak "
        "(fix it, with its own commit) or the noun table needs narrowing, never loosened silently",
    )

    if failures:
        print("no-consumer-names self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "no-consumer-names self-test: OK — every allowlist rot rule (1-7) bites: a renamed "
        "identifier, a missing declaring_path, a malformed AND a well-formed-but-non-ancestor "
        "ruling_sha, an unresolvable ruling_ref, a dead identifier, a trivially-short / "
        "governance-verb-leading / bare-line-cited reason, and a duplicate (identifier, "
        "declaring_path) pair (with the SAME identifier at a DIFFERENT path staying clean) -- "
        "plus a positive control on the real, committed register_content_hash_udf row."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

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
