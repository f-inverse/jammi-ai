#!/usr/bin/env python3
"""Diff-scoped mechanical gate: no journey marker lands on a published surface.

Docs reflect current state, never the journey (CLAUDE.md): no private-program
bookkeeping — a unit-id token (a capital letter followed by digits, an
optional trailing lowercase letter, and an optional hyphenated suffix), a
round-number phrase, a ruling-id citation of the shape this program's own
`.claude/agents/lead.md` bookkeeping uses throughout (a single uppercase
letter immediately followed by one to three digits and an optional trailing
prime mark — its concrete instances are deliberately not reproduced in THIS
module's own docstring, a published surface by this gate's own rule; see
`VOCABULARY` below for the exact patterns), a bracketed round-tracking
marker naming its own number, or an "as of this exact commit" hedge — on a
surface a reader outside the program consumes. Issue #555: the
discipline-test-auditor (an LLM) caught this class three rounds running on
three different units and, each time, cleared only the sites it happened to
enumerate — prose discipline held for the round it was checked, never past
it. This gate makes the class MECHANICAL: a fixed vocabulary, scanned only
over what a diff actually ADDS (so the tree's pre-existing hits under
``crates/`` — measured at roughly forty when this gate was written — never
block an unrelated PR), with published surfaces (rustdoc on a ``pub`` item,
a ``.proto`` comment, a Python docstring) BLOCKING and everything else
(a private ``//`` comment, a test body) only ADVISORY.

**Why the vocabulary is broader than the issue's literal `VOCABULARY`
list.** The issue's own "Why" names "audit-finding ids" as a whole class,
not only the one narrow phrase (the word "finding" immediately followed by
a capital F and a digit) its Rebuild section spells out as an example —
this program's actual bookkeeping vocabulary (visible throughout
`.claude/agents/lead.md`) uses that same single-uppercase-letter-plus-digits
shape under several different leading letters as a citable ruling/round id.
A pattern narrow enough to match only the literal "finding F"-prefixed form
would never catch the others — so this gate's `ruling_id` pattern in
`VOCABULARY` matches that whole shape generically, across every uppercase
leading letter, which is also why it needs an explicit allowlist for the
ONE sanctioned family sharing the identical shape: the constitution's own
seven invariant ids (`docs/swarm/CONSTITUTION.md:59-65`) — see
`ALLOWLISTED_RULING_IDS`. Known, stated limit: this broad net can
false-positive on an unrelated domain acronym that happens to share the
same letter-plus-digits shape (a classification-metric name, say) — that is
exactly why this is a human-amend-only gate whose PR is admin-merged after a
human reviews the vocabulary (see the wiring note below), not a claim that
every flagged token is a real violation.

**Allowlist (sanctioned vocabulary, never a finding).** (1) The
constitution's seven invariant ids (`docs/swarm/CONSTITUTION.md:59-65`,
listed verbatim in `ALLOWLISTED_RULING_IDS`) — the ONLY letter-plus-digits
family this gate ever excludes by literal token, checked at match time.
(2) A public issue/PR link (a bare `#`-number, `issue #`-number,
`PR #`-number, or a `github.com/.../issues/<n>` or `.../pull/<n>` URL) —
sanctioned by construction, never by a runtime filter: no vocabulary
pattern here ever matches a bare `#`-number on its own (the closest
pattern, `block_hash`, requires the literal word "BLOCK" immediately before
the `#`, so an issue reference on its own never collides with it in the
first place). (3) `docs/plans/**` and `docs/rigor/**` — the program's OWN
record; skipped by path, entirely, before any pattern runs.

**Definition exemption (a `ruling_id`- or `paren_ruling`-shaped token the
SAME FILE itself DEFINES, never a citation of the swarm's own ledger).**
The `ruling_id` pattern's own broad net (module docstring, above) catches
not only a reference to THIS program's bookkeeping but a `check_*.py`
gate's OWN rule- or test-case-id vocabulary about the tool it checks (a
single letter followed by digits, a `T`-letter id with a parenthesized
sub-index, or a `#`-prefixed `F`-letter id — deliberately not reproduced
as a literal instance here, same discipline as the module docstring's own
ruling-id family, above). A stable identifier a FILE DEFINES is current-
state vocabulary, exactly like K1–K7; a journey marker is an id the text
only REFERENCES from the swarm's own ledger, never one the file under
scan declares for itself. `_defined_ids_in_file` computes, per file, the
set of bare ids exempted via either of two independent mechanisms:

  1. The file's OWN module docstring (`.py` only) names it in the shape a
     tool declares its OWN rule/test vocabulary — captured once per file,
     REGARDLESS of where else in the file the id is later referenced.
  2. ANYWHERE in the file's own comments/docstrings, a DEFINITION-SHAPED
     line — a markdown table row, a markdown heading, or a line whose own
     text starts with the id immediately followed by `--`/`—`/`:` — names
     it; every `ruling_id`-shaped token that SAME line carries is exempted
     (a table row or heading names its subject anywhere on the line, not
     only at column 0).

Applies ONLY to the `ruling_id`/`paren_ruling` categories — `round_n`,
`unit_id`, `at_this_commit`, `fix_round`, `campaign_audit`,
`readme_ruling`, `block_hash`, and `finding_f` are NEVER exempted by this
mechanism and stay BLOCK unconditionally regardless of what a file
defines: a wave/group/unit id, a round number, and an "as of this commit"
hedge are journey shape by construction, never a tool's own current-state
rule name.

Run: `python3 ci/scripts/check_journey_markers.py`
Self-test: `python3 ci/scripts/check_journey_markers.py --self-test`
Hermetic: reads `git diff <base>...HEAD` + the tree at HEAD; no network, no build.
Wired in `swarm.yml` beside `check_no_consumer_names.py`.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths where the program's own bookkeeping legitimately lives — exempt
# entirely, before any pattern runs.
EXEMPT_PATH_PREFIXES = ("docs/plans/", "docs/rigor/")

# The constitution's own invariant ids — the one sanctioned family sharing
# RULING_ID's shape. See module docstring.
ALLOWLISTED_RULING_IDS = frozenset({"K1", "K2", "K3", "K4", "K5", "K6", "K7"})

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

VOCABULARY: list[tuple[str, re.Pattern[str]]] = [
    ("unit_id", re.compile(r"\bU\d+[a-z]?(?:-[0-9a-z]+)?\b")),
    ("fix_round", re.compile(r"\bfix round\b", re.IGNORECASE)),
    ("round_n", re.compile(r"\bround\s+\d+\b", re.IGNORECASE)),
    ("readme_ruling", re.compile(r"\bREADME r\d+\b")),
    ("paren_ruling", re.compile(r"\(P\d+'?[,)]")),
    ("finding_f", re.compile(r"\bfinding F\d+\b", re.IGNORECASE)),
    ("block_hash", re.compile(r"\bBLOCK #\d+\b")),
    ("at_this_commit", re.compile(r"\bat this commit\b", re.IGNORECASE)),
    ("campaign_audit", re.compile(r"\bcampaign audit\b", re.IGNORECASE)),
    # The general ruling/round-id shape — see module docstring for why this
    # is broader than the issue's literal "finding F5" example.
    ("ruling_id", re.compile(r"\b[A-Z]\d{1,3}'?\b")),
]


@dataclass(frozen=True)
class VocabMatch:
    pattern_name: str
    text: str


def find_vocabulary_matches(text: str) -> list[VocabMatch]:
    """Every vocabulary hit in `text`, K1-K7 already filtered out (the ONE
    allowlisted family sharing `ruling_id`'s shape — see module docstring).
    """
    matches: list[VocabMatch] = []
    for name, pattern in VOCABULARY:
        for m in pattern.finditer(text):
            matched = m.group(0)
            if name == "ruling_id" and matched in ALLOWLISTED_RULING_IDS:
                continue
            matches.append(VocabMatch(name, matched))
    return matches


# --------------------------------------------------------------------------- #
# surface classification
# --------------------------------------------------------------------------- #

# Rust: a `///`/`//!` doc-comment line is "published" only when it is
# actually attached to a fully-`pub` item (not `pub(crate)`/private) — the
# ONE-line attribute skip below is a stated limit: a multi-line attribute
# (`#[cfg(\n  feature = "x"\n)]`) defeats the forward scan.
_RUST_DOC_LINE_RE = re.compile(r"^\s*(///|//!)")
_RUST_ATTR_LINE_RE = re.compile(r"^\s*#\[.*\]\s*$")
_RUST_PUB_ITEM_RE = re.compile(r"^\s*pub(?!\()\b")

_PROTO_COMMENT_RE = re.compile(r"^\s*(//|/\*|\*)")

_TEST_PATH_RE = re.compile(
    r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|_tests?\.rs$"
)


def is_exempt_path(rel_path: str) -> bool:
    return any(rel_path.startswith(p) for p in EXEMPT_PATH_PREFIXES)


def is_test_path(rel_path: str) -> bool:
    """A stated, path-based heuristic — a `#[cfg(test)]` module inline
    inside an otherwise-non-test-named file is NOT caught by this (known
    limit, same shape as the rust doc-attachment scan's own)."""
    return bool(_TEST_PATH_RE.search(rel_path))


def _python_docstring_ranges(source: str) -> list[tuple[int, int]]:
    """`[(start_line, end_line), ...]`, 1-indexed inclusive, for every real
    docstring (module/class/function's own first `Expr` statement whose
    value is a string constant) in `source`. Returns `[]` on a syntax error
    (a file mid-edit, or one this gate cannot parse) rather than raising —
    the classifier then falls back to PRIVATE for every line in it, the
    conservative (non-blocking) direction.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ranges: list[tuple[int, int]] = []
    candidates: list[ast.AST] = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            candidates.append(node)
    for node in candidates:
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = getattr(first, "end_lineno", first.lineno)
            ranges.append((first.lineno, end))
    return ranges


def classify_surface(rel_path: str, file_lines: list[str], line_no: int) -> str:
    """`"rust_pub_doc" | "proto_comment" | "python_docstring" | "private"`
    for line `line_no` (1-indexed) of a file whose FULL content (at the
    revision the finding was read from) is `file_lines`. A test path is
    always `"private"`, regardless of comment shape.
    """
    if is_test_path(rel_path):
        return "private"
    if 1 <= line_no <= len(file_lines):
        line = file_lines[line_no - 1]
    else:
        line = ""

    if rel_path.endswith(".proto"):
        if _PROTO_COMMENT_RE.match(line):
            return "proto_comment"
        return "private"

    if rel_path.endswith(".rs"):
        if not _RUST_DOC_LINE_RE.match(line):
            return "private"
        # Forward scan for the item this doc comment attaches to.
        idx = line_no  # 0-indexed "next line" == line_no (1-indexed current)
        while idx < len(file_lines):
            nxt = file_lines[idx]
            stripped = nxt.strip()
            if not stripped:
                idx += 1
                continue
            if _RUST_DOC_LINE_RE.match(nxt) or _RUST_ATTR_LINE_RE.match(nxt):
                idx += 1
                continue
            break
        if idx < len(file_lines) and _RUST_PUB_ITEM_RE.match(file_lines[idx]):
            return "rust_pub_doc"
        return "private"

    if rel_path.endswith(".py"):
        ranges = _python_docstring_ranges("\n".join(file_lines))
        for start, end in ranges:
            if start <= line_no <= end:
                return "python_docstring"
        return "private"

    return "private"


SURFACE_SEVERITY = {
    "rust_pub_doc": "block",
    "proto_comment": "block",
    "python_docstring": "block",
    "private": "advisory",
}


# --------------------------------------------------------------------------- #
# definition exemption — see module docstring's own "Definition exemption"
# section for the full statement; this is the mechanical half.
# --------------------------------------------------------------------------- #

_RULING_ID_CORE_RE = re.compile(r"[A-Z]\d{1,3}'?")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
_DEFINITION_LEAD_RE = re.compile(r"^([A-Z]\d{1,3}'?)\s*(--|—|:)")
# A `check_*.py` gate's own rule/test-case-id vocabulary shapes, as such a
# module doc actually declares them (see e.g. `check_gpu_prove_once.py`'s
# `P1`-`P6` and its own "F4 audit fix" cross-references, later CITED with a
# leading `#` at its own call sites; `check_bundle_fixture.py`'s own
# citation of `test_bundle_cuda_libs.sh`'s `T4(3)`): `T<n>(<m>)` (a test
# case), `P<n>`/`R<n>` (a rule id), an `F<n>` audit-fix id (its own module
# doc never prefixes the DEFINING mention with `#`, only later citations
# do — the `#` is therefore OPTIONAL here, never required) — never
# reproduced as a literal instance in THIS file's own docstring (see
# module doc).
_TOOL_VOCAB_RE = re.compile(r"\bT\d+\(\d+\)|\b[PR]\d{1,3}\b|#?\bF\d{1,3}\b")
_WHOLE_PROSE_EXTS = {".md", ".mdx", ".rst", ".txt"}
_LINE_COMMENT_MARKERS = ("///", "//!", "//", "/*", "*", "#")


def _comment_or_docstring_line_numbers(rel_path: str, file_lines: list[str]) -> set[int]:
    """1-indexed line numbers, over the WHOLE file, a definition can live
    in — a real Python docstring range (`_python_docstring_ranges`, never a
    second docstring detector) plus every `#`-comment line for `.py`; every
    line, for a whole-prose extension; a `//`/`///`/`//!`/`/*`/`*`/`#`
    -prefixed line otherwise (this repo's own established comment-marker
    convention, see `lead-gate-lib.py`'s `_LINE_COMMENT_MARKERS`)."""
    ext = Path(rel_path).suffix.lower()
    if ext == ".py":
        out: set[int] = set()
        for start, end in _python_docstring_ranges("\n".join(file_lines)):
            out.update(range(start, end + 1))
        for i, line in enumerate(file_lines, start=1):
            if line.strip().startswith("#"):
                out.add(i)
        return out
    if ext in _WHOLE_PROSE_EXTS:
        return set(range(1, len(file_lines) + 1))
    return {
        i for i, line in enumerate(file_lines, start=1)
        if line.strip().startswith(_LINE_COMMENT_MARKERS)
    }


def _module_docstring_text(rel_path: str, file_lines: list[str]) -> str:
    """The file's OWN module docstring text (`.py` only — the one place a
    `check_*.py` gate already states its rules, by this repo's own
    convention). `""` for a non-`.py` file or a file with no module
    docstring."""
    if Path(rel_path).suffix.lower() != ".py":
        return ""
    ranges = _python_docstring_ranges("\n".join(file_lines))
    if not ranges:
        return ""
    start, end = ranges[0]
    return "\n".join(file_lines[start - 1:end])


def _defined_ids_in_file(rel_path: str, file_lines: list[str]) -> set[str]:
    """The set of bare `[A-Z]\\d{1,3}` ids (trailing apostrophe ALWAYS
    stripped — see `_match_core_id`'s own docstring for why a comparison
    against this set must never depend on whether a matched CITATION
    happened to be possessive) this FILE ITSELF defines — see module
    docstring's "Definition exemption" section for the full statement of
    the two mechanisms this computes."""
    defined: set[str] = set()

    module_doc = _module_docstring_text(rel_path, file_lines)
    for m in _TOOL_VOCAB_RE.finditer(module_doc):
        core = _RULING_ID_CORE_RE.search(m.group(0))
        if core:
            defined.add(core.group(0).rstrip("'"))

    ext = Path(rel_path).suffix.lower()
    for i in _comment_or_docstring_line_numbers(rel_path, file_lines):
        raw = file_lines[i - 1]
        # Strip ONE leading comment marker (if any) BEFORE testing any of
        # the three definition shapes below — a table row / heading inside
        # a `#`/`//`-marked comment line is preceded by the marker, not at
        # the raw line's own column 0 (only true for a whole-prose file,
        # where `content` below is already unchanged from `stripped_raw`).
        content = raw.strip()
        for marker in _LINE_COMMENT_MARKERS:
            if content.startswith(marker):
                content = content[len(marker):].strip()
                break
        if content.startswith("|"):
            defined.update(m.group(0).rstrip("'") for m in _RULING_ID_CORE_RE.finditer(content))
            continue
        if ext in _WHOLE_PROSE_EXTS and _HEADING_LINE_RE.match(content):
            defined.update(m.group(0).rstrip("'") for m in _RULING_ID_CORE_RE.finditer(content))
            continue
        m2 = _DEFINITION_LEAD_RE.match(content)
        if m2:
            defined.add(m2.group(1).rstrip("'"))
    return defined


def _match_core_id(vm: VocabMatch) -> str | None:
    """The bare `[A-Z]\\d{1,3}` core (the trailing `'?` NEVER kept — a
    POSSESSIVE reference captures `ruling_id`'s own OPTIONAL apostrophe as
    part of the match, which would otherwise silently mismatch
    `_defined_ids_in_file`'s own apostrophe-free definitions and defeat the
    exemption on exactly the possessive-citation shape this program's own
    prose most commonly uses) a `ruling_id` or `paren_ruling`
    match carries — `ruling_id`'s own matched text already IS that core
    (modulo the trailing apostrophe); `paren_ruling`'s matched text wraps
    it in the `(`/`[,)]` punctuation its own pattern requires. `None` for
    every OTHER vocabulary category — `round_n`/`unit_id`/
    `at_this_commit`/`fix_round`/`campaign_audit`/`readme_ruling`/
    `block_hash`/`finding_f` are never eligible for the definition
    exemption (see module docstring), and this function is the ONE gate
    that enforces that scope."""
    if vm.pattern_name == "ruling_id":
        return vm.text.rstrip("'")
    if vm.pattern_name == "paren_ruling":
        m = _RULING_ID_CORE_RE.search(vm.text)
        return m.group(0).rstrip("'") if m else None
    return None


# --------------------------------------------------------------------------- #
# diff scoping
# --------------------------------------------------------------------------- #


def resolve_diff_base() -> str | None:
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


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines_with_line_numbers(base: str) -> list[tuple[str, int, str]]:
    """`[(rel_path, new_file_line_no, added_text), ...]` for every line
    ADDED (`+`, never `+++`) in `git diff --unified=0 <base>...HEAD`, over
    the WHOLE tree (every surface kind this gate cares about lives outside
    `crates/`). Line numbers are derived from each hunk's own `@@ -a,b +c,d
    @@` header — `--unified=0` means every non-`+++`/`@@` line in a hunk is
    either an add or a context line at zero width, so the running new-file
    counter only needs to seed from `c` and increment on `+`.
    """
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    out: list[tuple[str, int, str]] = []
    current_file: str | None = None
    new_line_no = 0
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            raw_path = line[len("+++ ") :]
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("@@ "):
            m = _HUNK_HEADER_RE.match(line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if current_file is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append((current_file, new_line_no, line[1:]))
            new_line_no += 1
    return out


def read_file_lines(rel_path: str) -> list[str]:
    try:
        return (REPO_ROOT / rel_path).read_text(errors="ignore").splitlines()
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finding:
    rel_path: str
    line_no: int
    severity: str
    surface: str
    matches: tuple[str, ...]
    text: str

    def render(self) -> str:
        vocab = ", ".join(self.matches)
        return (
            f"{self.severity.upper():>8}  {self.rel_path}:{self.line_no}  "
            f"[{self.surface}] {vocab}  -- {self.text.strip()!r}"
        )


def scan_added_lines(added: list[tuple[str, int, str]]) -> list[Finding]:
    findings: list[Finding] = []
    file_cache: dict[str, list[str]] = {}
    defined_ids_cache: dict[str, set[str]] = {}
    for rel_path, line_no, text in added:
        if is_exempt_path(rel_path):
            continue
        vocab_matches = find_vocabulary_matches(text)
        if not vocab_matches:
            continue
        if rel_path not in file_cache:
            file_cache[rel_path] = read_file_lines(rel_path)
        if rel_path not in defined_ids_cache:
            defined_ids_cache[rel_path] = _defined_ids_in_file(rel_path, file_cache[rel_path])
        defined_ids = defined_ids_cache[rel_path]
        # Definition exemption (module docstring's own section) — scoped to
        # ruling_id/paren_ruling ONLY (`_match_core_id` returns `None` for
        # every other category, which then never matches `in defined_ids`
        # regardless of that set's own contents).
        vocab_matches = [
            vm for vm in vocab_matches
            if not (_match_core_id(vm) is not None and _match_core_id(vm) in defined_ids)
        ]
        if not vocab_matches:
            continue
        surface = classify_surface(rel_path, file_cache[rel_path], line_no)
        severity = SURFACE_SEVERITY[surface]
        findings.append(
            Finding(
                rel_path=rel_path,
                line_no=line_no,
                severity=severity,
                surface=surface,
                matches=tuple(sorted({m.pattern_name for m in vocab_matches})),
                text=text,
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # --- vocabulary forms, each on a published rust surface (BLOCK) ---
        rust_src = (
            "/// fix round 3 closed this.\n"
            "pub fn a() {}\n"
            "\n"
            "/// See U4b for the unit.\n"
            "pub fn b() {}\n"
            "\n"
            "/// README r29 explains this.\n"
            "pub fn c() {}\n"
            "\n"
            "/// (P3', see above)\n"
            "pub fn d() {}\n"
            "\n"
            "/// finding F5 was the cause.\n"
            "pub fn e() {}\n"
            "\n"
            "/// BLOCK #12 covered this.\n"
            "pub fn f() {}\n"
            "\n"
            "/// True at this commit.\n"
            "pub fn g() {}\n"
            "\n"
            "/// A campaign audit found this.\n"
            "pub fn h() {}\n"
            "\n"
            "/// Round 7 fixed this.\n"
            "pub fn i() {}\n"
            "\n"
            "/// See Z17 for detail.\n"
            "pub fn j() {}\n"
        )
        _write(root, "src/lib.rs", rust_src)
        rust_lines = rust_src.splitlines()
        published_lines = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
        vocab_names = [
            "fix_round", "unit_id", "readme_ruling", "paren_ruling", "finding_f",
            "block_hash", "at_this_commit", "campaign_audit", "round_n", "ruling_id",
        ]
        for line_no, expected_vocab in zip(published_lines, vocab_names):
            matches = find_vocabulary_matches(rust_lines[line_no - 1])
            check(
                f"vocab:{expected_vocab} matches",
                any(m.pattern_name == expected_vocab for m in matches),
            )
            surface = classify_surface("src/lib.rs", rust_lines, line_no)
            check(f"vocab:{expected_vocab} surface=rust_pub_doc", surface == "rust_pub_doc")
            check(
                f"vocab:{expected_vocab} severity=block",
                SURFACE_SEVERITY[surface] == "block",
            )

        # --- the SAME rust doc comment, but on a PRIVATE (non-pub) item ---
        private_rust = "/// fix round 3, private item.\nfn a() {}\n"
        _write(root, "src/priv.rs", private_rust)
        priv_lines = private_rust.splitlines()
        surf = classify_surface("src/priv.rs", priv_lines, 1)
        check("private rust item -> private surface", surf == "private")
        check("private rust item -> advisory", SURFACE_SEVERITY[surf] == "advisory")

        # --- pub(crate) is NOT published ---
        crate_rust = "/// fix round 3, crate-only.\npub(crate) fn a() {}\n"
        _write(root, "src/crate_only.rs", crate_rust)
        crate_lines = crate_rust.splitlines()
        surf = classify_surface("src/crate_only.rs", crate_lines, 1)
        check("pub(crate) item -> private surface", surf == "private")

        # --- .proto comment (BLOCK) ---
        proto_src = "// fix round 3 changed this field.\nmessage M { int32 x = 1; }\n"
        _write(root, "a.proto", proto_src)
        proto_lines = proto_src.splitlines()
        surf = classify_surface("a.proto", proto_lines, 1)
        check(".proto comment -> proto_comment surface", surf == "proto_comment")
        check(".proto comment -> block", SURFACE_SEVERITY[surf] == "block")

        # --- python docstring (BLOCK) vs a plain python comment (advisory) ---
        py_src = (
            '"""fix round 3 module docstring."""\n'
            "\n"
            "# fix round 3 plain comment, not a docstring.\n"
            "def f():\n"
            '    """fix round 3 function docstring."""\n'
            "    return 1\n"
        )
        _write(root, "a.py", py_src)
        py_lines = py_src.splitlines()
        surf_mod = classify_surface("a.py", py_lines, 1)
        check("python module docstring -> python_docstring", surf_mod == "python_docstring")
        surf_comment = classify_surface("a.py", py_lines, 3)
        check("python plain comment -> private", surf_comment == "private")
        surf_fn = classify_surface("a.py", py_lines, 5)
        check("python function docstring -> python_docstring", surf_fn == "python_docstring")

        # --- test path forces private regardless of surface ---
        test_rust = "/// fix round 3, in a test file.\npub fn t() {}\n"
        _write(root, "tests/it/journey.rs", test_rust)
        test_lines = test_rust.splitlines()
        surf = classify_surface("tests/it/journey.rs", test_lines, 1)
        check("rust doc comment in tests/ path -> private", surf == "private")

        py_test = '"""fix round 3 docstring in a test module."""\n'
        _write(root, "cookbook/book/tests/test_x.py", py_test)
        surf = classify_surface(
            "cookbook/book/tests/test_x.py", py_test.splitlines(), 1
        )
        check("python docstring in test_*.py -> private", surf == "private")

        # --- K1-K7 allowlist: the SAME shape as ruling_id, never a finding ---
        k_line = "/// See K5 for the invariant."
        matches = find_vocabulary_matches(k_line)
        check("K5 is allowlisted (no ruling_id match)", not matches)
        # RED against a mutation that removes the allowlist filter: with the
        # filter disabled, K5 DOES match ruling_id's raw pattern -- proving
        # the allowlist is load-bearing, not a vacuous no-op.
        raw_ruling_matches = [
            m.group(0)
            for name, pat in VOCABULARY
            if name == "ruling_id"
            for m in pat.finditer(k_line)
        ]
        check("K5 WOULD match ruling_id's raw shape (oracle can go red)", raw_ruling_matches == ["K5"])

        # A non-allowlisted single-letter id of the identical shape DOES fire.
        z_line = "/// See Z17 for the round."
        z_matches = find_vocabulary_matches(z_line)
        check("Z17 (non-allowlisted) matches ruling_id", any(m.text == "Z17" for m in z_matches))

        # --- definition exemption: a `check_*.py` gate's OWN rule/test-id
        # vocabulary, declared in its module doc, is current-state text, not
        # a journey reference (coordinator's own binding correction) -------
        gate_src = (
            '"""Some gate. Six rules:\n\n'
            "  P3 (PROMOTION_TABLE, every row reconciled): every reviewed row.\n"
            '"""\n'
            "\n"
            "def check_p3():\n"
            '    """the same ambiguity discipline P3\'s other rules already hold to,\n'
            "    and this step-gated row (P3) only ever pins the NAMED step's own\n"
            '    if:, never an UNDEFINED Z17-style id or a round 3 hedge."""\n'
            "    return True\n"
        )
        gate_lines = gate_src.splitlines()
        defined = _defined_ids_in_file("ci/scripts/fake_gate.py", gate_lines)
        check("module-doc-declared P3 is captured as defined", "P3" in defined)

        def _survives(vocab_matches):
            return [
                vm for vm in vocab_matches
                if not (_match_core_id(vm) is not None and _match_core_id(vm) in defined)
            ]

        possessive_line = gate_lines[6]  # "the same ambiguity discipline P3's other rules..."
        check(
            "a possessive citation of a module-doc-declared id (P3's) produces NO finding",
            not _survives(find_vocabulary_matches(possessive_line)),
        )
        paren_line = gate_lines[7]  # "and this step-gated row (P3) only ever pins..."
        check(
            "a parenthetical citation of the SAME declared id ((P3), BOTH paren_ruling "
            "and ruling_id) produces NO finding",
            not _survives(find_vocabulary_matches(paren_line)),
        )
        undefined_line = "See Z17 for the invariant, never declared by this file."
        check(
            "an UNDEFINED id of the identical shape (Z17) still produces a finding",
            bool(_survives(find_vocabulary_matches(undefined_line))),
        )
        round_line = gate_lines[8]  # "if:, never an UNDEFINED Z17-style id or a round 3 hedge."
        check(
            "round_n stays BLOCK unconditionally, even inside a file with declared vocabulary",
            any(vm.pattern_name == "round_n" for vm in _survives(find_vocabulary_matches(round_line))),
        )

        # --- definition exemption, mechanism 2: a table-row/heading/`<ID>
        # --`-shaped line ANYWHERE in the file's own prose, never restricted
        # to the module docstring ------------------------------------------
        table_src = (
            '"""No module-doc vocabulary declared here."""\n'
            "\n"
            "def f():\n"
            "    # | rule | outcome |\n"
            "    # | R2 | denies |\n"
            '    """R2 governs this branch."""\n'
            "    return True\n"
        )
        table_lines = table_src.splitlines()
        table_defined = _defined_ids_in_file("ci/scripts/fake_table.py", table_lines)
        check("a table-row-cited id (R2) is captured as defined", "R2" in table_defined)
        check(
            "an in-prose reference to the table-defined id produces NO finding",
            not [
                vm for vm in find_vocabulary_matches("R2 governs this branch.")
                if not (_match_core_id(vm) is not None and _match_core_id(vm) in table_defined)
            ],
        )

        lead_src = (
            '"""No module-doc vocabulary declared here either."""\n'
            "\n"
            "def f():\n"
            "    # R9 -- this branch's own denial text.\n"
            '    """cross-referenced again: R9 denies for the same reason."""\n'
            "    return True\n"
        )
        lead_lines = lead_src.splitlines()
        lead_defined = _defined_ids_in_file("ci/scripts/fake_lead.py", lead_lines)
        check("an `<ID> --`-at-line-start definition (R9) is captured as defined", "R9" in lead_defined)

        # --- public issue links never collide (sanctioned by construction) ---
        issue_line = "Fixes a bug (see #557); closes issue #558; ref PR #559."
        check("bare issue/PR links produce no finding", not find_vocabulary_matches(issue_line))

        # --- docs/plans/** and docs/rigor/** are exempt by path ---
        check("docs/plans/** is exempt", is_exempt_path("docs/plans/67-x/DESIGN.md"))
        check("docs/rigor/** is exempt", is_exempt_path("docs/rigor/x.jsonl"))
        check("crates/** is NOT exempt", not is_exempt_path("crates/jammi-db/src/lib.rs"))

        # --- diff-scoping: added-lines extraction from a real hunk header ---
        diff_text = (
            "diff --git a/src/x.rs b/src/x.rs\n"
            "index 1111111..2222222 100644\n"
            "--- a/src/x.rs\n"
            "+++ b/src/x.rs\n"
            "@@ -10,0 +11,2 @@ fn existing() {\n"
            "+/// fix round 3\n"
            "+pub fn added() {}\n"
        )
        parsed = _parse_unified_diff_for_test(diff_text)
        check(
            "hunk header line-number seeding",
            parsed == [("src/x.rs", 11, "/// fix round 3"), ("src/x.rs", 12, "pub fn added() {}")],
        )

    if failures:
        print("check_journey_markers self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "check_journey_markers self-test: OK -- every vocabulary form fires on every "
        "surface kind with the right severity, K1-K7 is allowlisted (and shown to be "
        "load-bearing against the raw pattern), public issue links never collide, "
        "docs/plans and docs/rigor are exempt by path, hunk-header line-number "
        "seeding is correct, and the definition exemption (a module-doc-declared or "
        "definition-line-declared id) suppresses ONLY the matching citation while an "
        "undefined id of the identical shape, and round_n regardless of any declared "
        "vocabulary, both still fire."
    )
    return 0


def _parse_unified_diff_for_test(diff_text: str) -> list[tuple[str, int, str]]:
    """The same hunk-walking logic as `added_lines_with_line_numbers`, over
    a literal diff string instead of a real `git diff` invocation -- lets
    the self-test exercise the line-number arithmetic without a git repo.
    """
    out: list[tuple[str, int, str]] = []
    current_file: str | None = None
    new_line_no = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw_path = line[len("+++ ") :]
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_file = None if raw_path == "/dev/null" else raw_path
            continue
        if line.startswith("@@ "):
            m = _HUNK_HEADER_RE.match(line)
            if m:
                new_line_no = int(m.group(1))
            continue
        if current_file is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append((current_file, new_line_no, line[1:]))
            new_line_no += 1
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    base = resolve_diff_base()
    if base is None:
        print(
            "check_journey_markers: no diff base resolvable (no origin/main, no main); "
            "skipping (nothing to diff-scope against)."
        )
        return 0

    added = added_lines_with_line_numbers(base)
    findings = scan_added_lines(added)

    blocking = [f for f in findings if f.severity == "block"]
    advisory = [f for f in findings if f.severity == "advisory"]

    for f in sorted(advisory, key=lambda f: (f.rel_path, f.line_no)):
        print(f.render())
    for f in sorted(blocking, key=lambda f: (f.rel_path, f.line_no)):
        print(f.render(), file=sys.stderr)

    if blocking:
        print(
            f"\ncheck_journey_markers: {len(blocking)} BLOCKING finding(s) on a published "
            "surface (rustdoc on a pub item, a .proto comment, a python docstring). Docs "
            "reflect current state, never the journey (CLAUDE.md) -- remove the marker or "
            "state the invariant without it.",
            file=sys.stderr,
        )
        return 1

    if advisory:
        print(
            f"\ncheck_journey_markers: {len(advisory)} advisory finding(s) on a private "
            "surface (a comment or a test) -- printed, not blocking."
        )
    else:
        print("check_journey_markers: OK -- no journey marker in the diff's added lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
