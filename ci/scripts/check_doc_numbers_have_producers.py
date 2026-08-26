#!/usr/bin/env python3
"""Every measurement-shaped number in a kernel/test doc comment or `assert!`
message must cite what produced it.

The class this closes: a doc comment or an `assert!`/`panic!` message states
a precise-looking measurement — "26% of elements", "1/6144", "100%
(6144/6144)", a bare cosine like "0.796" — that nothing in the tree actually
produces. The number reads as evidence (a real run, a real oracle) but is
unverifiable: no test, no artifact, no script anywhere computes it, so a
reader (or an implementer citing it as ground truth) cannot re-derive or even
locate where it came from. `docs/maintainer/cuda-kernel-guide.md` §3.9 states
this rule in one paragraph, alongside the adjacent oracle disciplines §3.7
(write comparisons affirmatively) and §3.8 (no absolute ULP floor) this is a
third leg of.

BINDING UNIT: the contiguous doc-comment BLOCK — a maximal run of `///`/
`//!`/`//` lines — containing the number (round-3 audit fix: a 12-line
window both missed a same-block citation written one sentence AWAY from the
number it covers, and could reach a citation several lines off that has
nothing to do with the number it happened to sit near). A citation ANYWHERE
in the block covers every number in that block. `no-producer: <reason>` is
the one exception — it stays LINE-scoped: it exempts only the number(s) on
its own line, never the rest of the block, so a stray opt-out cannot
silently launder an unrelated number several lines away. An
`assert!`/`panic!`-family message is always its own singleton block (it is
code, not a comment run).

WHAT COUNTS AS "measurement-shaped": three number shapes, each REQUIRED to
have one of `ADJACENT_WORD_ALTS` appear (case-insensitively) ANYWHERE in the
number's own BLOCK — not a plain N/M, X%, or 0.XXX floating around in math
notation, an enumeration, a fixed statistical constant, or a defined
threshold, all of which carry no claim of an observed run:

  - `N/M` element/mismatch counts, REQUIRED to have `M >= 64` (a smaller
    denominator is almost always an index/enumeration/small ratio in this
    codebase, never a real mismatch-element count, which starts in the
    hundreds) and NOT: immediately preceded by `.` (a decimal-embedded
    fragment, e.g. the `3/16` inside `1.3/16`), `/` or immediately followed
    by `/` (a fragment of a longer `a/b/c` chain — excluding both
    directions kills every non-first segment of a chain in one rule), `^`
    (an exponent, `x^2/2`), or an `n=`-prefixed shape enumeration
    (`n=32/900`).
  - percentages (`X%`, `X.Y%`).
  - bare cosines (`0.dd` through `0.dddddd` — 2 to 6 decimal digits).

  A version string (`0.11.0`, or a known crate name immediately before a
  bare `X.Y` — `candle 0.11`, `candle-core 0.11's`) is excluded from all
  three shapes: it is never a measurement.

  Round-3 audit fix: the round-2 "definitional operator" exclusion
  (anything right of `=`/`>=`/etc. was dropped) and the "same line as a
  `[`CONST`]` doc-link" exclusion (a whole line was blanket-suppressed by
  the mere presence of an unrelated link on it) are BOTH REMOVED — a
  measured value restated as an equality (`measured on the A100: 3.1e-3 =
  0.39`) still counts.

WHERE: doc comments (`///`, `//!`) and plain `//` comment blocks, plus the
message string of an `assert!`/`assert_eq!`/`assert_ne!`/`debug_assert!`/
`panic!` call (single-line message only — a multi-line `format!` message
inside the macro call is out of scope for this pass, the same "cheap,
visible, first-pass tripwire" cost `check_sqlite_isms.py`'s own doc accepts),
under `crates/jammi-kernels/{src,tests}`, `crates/jammi-encoders/src`,
`crates/jammi-lora/src`, `crates/jammi-bench/src` — `.rs` and CUDA source
(`.cu`/`.cuh`) files.

PRODUCER CITATION GRAMMAR (round-3 audit fix: the round-2 grammar only
recognised a BARE identifier after `see`/`printed by`, rejecting this
repo's own dominant idioms — `` see [`fn`] `` alone outnumbers the bare
form several times over — while ALSO accepting `see below`/`see table
3`/`see step 2`, none of which name a real producer). A citation is one of:

  - `` see [`<path>`] ``, `` see `<path>` ``, or `see <path>` — `<path>` is
    a `::`-separated identifier chain (`ops::softmax::MAX_LAST_DIM`,
    `Self::with_scale`, a bare `check_bit_identity_fixture`); resolved by
    its LAST segment.
  - `printed by` in the same three forms, resolved the same way.
  - `measured by <path>` — `<path>` is a filesystem path; resolved by
    `git ls-files` membership (a committed artifact), never a bare
    `Path.exists()` (which would accept an untracked, gitignored file).
  - `` [producer: <path or fn form>] `` — the same resolution as `see`,
    OR (if the content isn't a `::`-path/identifier shape) a tracked path.

A resolved fn name is REQUIRED to be a `#[test]`-attributed function
(including `#[tokio::test]`/any attribute whose text contains "test"), or a
`pub fn` declared in a file under a `tests/` directory — never "any fn with
that name exists somewhere", which is what let `see below` resolve against
a real, but wholly unrelated, non-test `fn below` a prior round of this
gate shipped with. This is the one piece of real discipline in the
grammar: it is what makes `see table 3` and `see step 2` correctly stay
unresolved (neither names a real identifier at all) and what makes `see
below` correctly stay unresolved EVEN THOUGH a real `fn below` exists
elsewhere in the tree (it is not a test).

FAIL-CLOSED, file:line, offending number. A committed allowlist
(`ALLOWLIST_PATH`) carries pre-existing debt, keyed on `file:number:sha1` —
`sha1` of the NORMALIZED text of the number's own line (not the line
number), so a line shift elsewhere in the file does not silently re-trigger
an already-tracked entry, and does not silently un-flag a genuinely edited
line either (the hash changes with the text). `ci/doc_number_allowlist_
classification.md` marks every seeded entry real/noise by hand, with the
noise rate computed directly from that table — not asserted in prose.

The allowlist may only shrink: `--check-allowlist-only-shrinks` fetches
`origin/main` (fails CLOSED — `check=True` — if the fetch itself fails,
never silently proceeding on stale/absent data), requires `origin/main` to
resolve at all (`git rev-parse --verify origin/main`, fails the leg if
not), and only THEN treats "no allowlist file at `origin/main`" as a
legitimate bootstrap (this PR is the one introducing the file) rather than
conflating it with "the ref didn't resolve" — the message states which
case fired.

Run: `python3 ci/scripts/check_doc_numbers_have_producers.py`
Self-test: `python3 ci/scripts/check_doc_numbers_have_producers.py --self-test`
Allowlist-only-shrinks leg (network: fetches `origin/main`):
`python3 ci/scripts/check_doc_numbers_have_producers.py --check-allowlist-only-shrinks`
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = [
    REPO_ROOT / "crates" / "jammi-kernels" / "src",
    REPO_ROOT / "crates" / "jammi-kernels" / "tests",
    REPO_ROOT / "crates" / "jammi-encoders" / "src",
    REPO_ROOT / "crates" / "jammi-lora" / "src",
    REPO_ROOT / "crates" / "jammi-bench" / "src",
]

# Matches the module doc's WHERE section exactly: `.rs` and CUDA source only.
SCAN_EXTENSIONS = {".rs", ".cu", ".cuh"}

ALLOWLIST_PATH = REPO_ROOT / "ci" / "doc_number_allowlist.txt"
CLASSIFICATION_PATH = REPO_ROOT / "ci" / "doc_number_allowlist_classification.md"

# --- trigger words -----------------------------------------------------

ADJACENT_WORD_ALTS = [
    r"cosine",
    r"mismatch(?:es)?",
    r"elements",
    r"ceiling",
    r"bit-match\w*",
    r"diverg\w*",
    r"measured",
    r"observed",
    r"differ(?:s|ed)?",
    r"disagree\w*",
    r"values",
    r"run",
    r"pod",
    r"a100",
    r"seed\w*",
]
_ADJACENT_RE = re.compile(r"\b(?:" + "|".join(ADJACENT_WORD_ALTS) + r")\b", re.IGNORECASE)

# --- number shapes -----------------------------------------------------

NM_RE = re.compile(r"\b\d{1,7}\s*/\s*\d{1,7}\b")
PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,4})?\s*%")
COSINE_RE = re.compile(r"\b0\.\d{2,6}\b")
VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")

# A crate name (this workspace's own deps that show up in prose as "candle
# 0.11", "candle-core 0.11's") immediately before a bare `X.Y` reads as a
# version mention, not a measurement — `VERSION_RE` above only catches
# 3-component versions (`0.11.0`); this catches the 2-component form this
# codebase actually writes crate versions as. Kept per the round-3 audit's
# "keep the version-string exclusion" — this is the same exclusion, just
# covering the 2-component spelling `VERSION_RE` cannot.
_CRATE_NAME_PREFIX_RE = re.compile(
    r"(?i)\b(?:candle(?:-[a-z]+)*|torch|cudarc|half|rayon|tokio|serde(?:-[a-z]+)*|"
    r"anyhow|thiserror|usearch)\s*$"
)

_N_EQUALS_CHAIN_RE = re.compile(r"(?i)\bn\s*=\s*$")

ASSERT_CALL_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|panic)!\s*\("
)
STRING_LITERAL_RE = re.compile(r'"')

NO_PRODUCER_RE = re.compile(r"\bno-producer:")

_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://!|///|//)")
_FN_LINE_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
_ATTR_LINE_RE = re.compile(r"^\s*#!?\[")

# --- citation grammar ----------------------------------------------------

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_PATH = rf"{_IDENT}(?:::{_IDENT})*"

_BRACKET_BACKTICK_TARGET_RE = re.compile(rf"^\[`({_PATH})`\]")
_BACKTICK_TARGET_RE = re.compile(rf"^`({_PATH})`")
_BARE_TARGET_RE = re.compile(rf"^({_PATH})")

_SEE_KW_RE = re.compile(r"\bsee\s+")
_PRINTED_BY_KW_RE = re.compile(r"\bprinted by\s+")
_MEASURED_BY_KW_RE = re.compile(r"\bmeasured by\s+")
_PRODUCER_TAG_RE = re.compile(r"\[producer:\s*([^\]]+)\]")
_ARTIFACT_PATH_RE = re.compile(r"^`?([^\s`,;]+)")


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


def build_test_fn_index() -> set[str]:
    """Every `fn <name>` in a tracked `.rs` file that is either
    `#[test]`-attributed (any attribute whose text contains "test",
    case-insensitively — covers `#[test]`, `#[tokio::test]`,
    `#[test_log::test]`, ...; walks backward through however many stacked
    attribute lines sit directly above the `fn` line) or a `pub fn`
    declared in a file under a `tests/` directory.

    Round-3 audit fix: the round-2 index accepted ANY `fn` with a matching
    name, which is what let `` see [`below`] ``/`` see below `` resolve
    against a real, but wholly unrelated, non-test `fn below` — a citation
    is only real evidence if it points at something that actually
    PRODUCED a result (a test), not merely at some function that happens
    to share a word with ordinary English prose ("below", "table", "step"
    are all plausible fn names somewhere in a repo this size).
    """
    names: set[str] = set()
    for rel in _run(["git", "ls-files", "--", "*.rs"]).splitlines():
        path = REPO_ROOT / rel
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        under_tests_dir = f"/{rel}".find("/tests/") != -1 or rel.startswith("tests/")
        for i, line in enumerate(lines):
            m = _FN_LINE_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            is_pub = bool(re.match(r"^\s*pub\b", line))
            has_test_attr = False
            j = i - 1
            while j >= 0 and _ATTR_LINE_RE.match(lines[j]):
                if "test" in lines[j].lower():
                    has_test_attr = True
                j -= 1
            if has_test_attr or (is_pub and under_tests_dir):
                names.add(name)
    return names


def is_test_fn(name: str, fn_index: set[str]) -> bool:
    return name in fn_index


def strip_comment_prefix(line: str) -> str:
    return _COMMENT_PREFIX_RE.sub("", line, count=1).lstrip()


def _citation_target(text_after_keyword: str) -> str | None:
    """`text_after_keyword` is everything right after `see `/`printed by `
    in a joined, comment-prefix-stripped block string. Returns the full
    matched `::`-path (e.g. `ops::softmax::MAX_LAST_DIM`), trying the
    rustdoc-link form first, then a bare backtick-wrapped form, then a
    fully bare form — or `None` if none match.
    """
    for pattern in (_BRACKET_BACKTICK_TARGET_RE, _BACKTICK_TARGET_RE, _BARE_TARGET_RE):
        m = pattern.match(text_after_keyword)
        if m:
            return m.group(1)
    return None


def _last_segment(path: str) -> str:
    return path.rsplit("::", 1)[-1]


def is_block_bound(block_text: str, fn_index: set[str], tracked: set[str]) -> bool:
    """`block_text` is the WHOLE comment block's text, comment prefixes
    stripped and lines joined by a single space (so a `see`/`printed by`
    keyword at one line's end and its target on the next line resolve
    exactly like an unwrapped one-line citation would). True iff ANY
    citation anywhere in it resolves.
    """
    for kw_re in (_SEE_KW_RE, _PRINTED_BY_KW_RE):
        for m in kw_re.finditer(block_text):
            target = _citation_target(block_text[m.end() :])
            if target and is_test_fn(_last_segment(target), fn_index):
                return True

    for m in _MEASURED_BY_KW_RE.finditer(block_text):
        am = _ARTIFACT_PATH_RE.match(block_text[m.end() :])
        if am:
            candidate = am.group(1).strip("`").rstrip(".,;")
            if candidate in tracked:
                return True

    for m in _PRODUCER_TAG_RE.finditer(block_text):
        raw = m.group(1).strip()
        target = _citation_target(raw)
        if target and is_test_fn(_last_segment(target), fn_index):
            return True
        candidate = raw.strip("`").rstrip(".,;")
        if candidate in tracked:
            return True

    return False


# --- scanning ------------------------------------------------------------


@dataclass(frozen=True)
class NumberHit:
    text: str  # the matched number's literal text
    kind: str  # "N/M" | "%" | "cosine"


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def is_scan_line(line: str) -> bool:
    """A "doc comment / // block" line, OR a line invoking one of the
    assert-family macros with a string-literal message on it (single-line
    only — see module doc's WHERE section for the multi-line limitation).
    """
    stripped = line.lstrip()
    if stripped.startswith("//"):
        return True
    if ASSERT_CALL_RE.search(line) and STRING_LITERAL_RE.search(line):
        return True
    return False


def is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("//")


def compute_blocks(lines: list[str]) -> list[int]:
    """`block_of[i]` is the block id line `i` belongs to. Contiguous
    comment lines (any run of lines starting with `//`, covering `///`,
    `//!`, and plain `//` — including a bare `//`/`///`/`//!` paragraph
    separator, which still starts with `//` and so does NOT break the
    block) share one id; every other line (code, including an
    `assert!`/`panic!` line) is its own singleton block.
    """
    block_of = [0] * len(lines)
    next_id = 0
    current: int | None = None
    for i, line in enumerate(lines):
        if is_comment_line(line):
            if current is None:
                current = next_id
                next_id += 1
            block_of[i] = current
        else:
            block_of[i] = next_id
            next_id += 1
            current = None
    return block_of


def find_number_hits(line: str) -> list[NumberHit]:
    """Measurement-shaped numbers on `line`, passed through the shape
    exclusions the module doc's WHAT COUNTS AS section documents. Trigger-
    word and binding checks are NOT done here — both are BLOCK-scoped (see
    `scan_lines`), not line-scoped.
    """
    version_spans = [m.span() for m in VERSION_RE.finditer(line)]

    def in_version(span: tuple[int, int]) -> bool:
        return any(_overlaps(span, v) for v in version_spans)

    def is_versioned(start: int) -> bool:
        tail = line[:start].rstrip().rstrip("`").rstrip()
        return bool(_CRATE_NAME_PREFIX_RE.search(tail))

    hits: list[NumberHit] = []

    for m in NM_RE.finditer(line):
        span = m.span()
        start, end = span
        if in_version(span) or is_versioned(start):
            continue
        text = m.group(0)
        _, denom_str = re.split(r"\s*/\s*", text)
        try:
            denom = int(denom_str)
        except ValueError:
            denom = None
        if denom is None or denom < 64:
            continue
        prev_char = line[start - 1] if start > 0 else ""
        if prev_char in (".", "/", "^"):
            continue
        if end < len(line) and line[end] == "/":
            continue
        if _N_EQUALS_CHAIN_RE.search(line[:start].rstrip()):
            continue
        hits.append(NumberHit(text, "N/M"))

    pct_spans: list[tuple[int, int]] = []
    for m in PCT_RE.finditer(line):
        span = m.span()
        if in_version(span) or is_versioned(span[0]):
            continue
        pct_spans.append(span)
        hits.append(NumberHit(m.group(0), "%"))

    for m in COSINE_RE.finditer(line):
        span = m.span()
        if in_version(span) or is_versioned(span[0]):
            continue
        # One hit per token: a percentage's own fractional digits (e.g. the
        # `0.52` inside `0.52%`) are already claimed above as the `%` hit —
        # don't also flag the identical span as a separate bare cosine.
        if any(_overlaps(span, p) for p in pct_spans):
            continue
        hits.append(NumberHit(m.group(0), "cosine"))

    return hits


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    number: str
    kind: str
    line_text: str  # the raw text of the number's own line, for allowlist hashing


def scan_lines(
    lines: list[str], file_label: str, fn_index: set[str], tracked: set[str]
) -> list[Finding]:
    """`lines` is 0-indexed; line numbers reported are 1-indexed."""
    n = len(lines)
    scan_mask = [is_scan_line(l) for l in lines]
    block_of = compute_blocks(lines)

    block_line_indices: dict[int, list[int]] = {}
    for i in range(n):
        block_line_indices.setdefault(block_of[i], []).append(i)

    trigger_text_cache: dict[int, str] = {}
    clean_text_cache: dict[int, str] = {}

    def trigger_text(bid: int) -> str:
        if bid not in trigger_text_cache:
            trigger_text_cache[bid] = "\n".join(lines[j] for j in block_line_indices[bid])
        return trigger_text_cache[bid]

    def clean_text(bid: int) -> str:
        if bid not in clean_text_cache:
            clean_text_cache[bid] = " ".join(
                strip_comment_prefix(lines[j]) for j in block_line_indices[bid]
            )
        return clean_text_cache[bid]

    findings: list[Finding] = []
    for i, line in enumerate(lines):
        if not scan_mask[i]:
            continue
        hits = find_number_hits(line)
        if not hits:
            continue
        bid = block_of[i]
        if not _ADJACENT_RE.search(trigger_text(bid)):
            continue
        if NO_PRODUCER_RE.search(line):
            continue
        if is_block_bound(clean_text(bid), fn_index, tracked):
            continue
        for hit in hits:
            findings.append(Finding(file_label, i + 1, hit.text, hit.kind, line))
    return findings


def scan_tree() -> list[Finding]:
    fn_index = build_test_fn_index()
    tracked = tracked_files()
    findings: list[Finding] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            raise SystemExit(f"check_doc_numbers_have_producers: scan root not found: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_EXTENSIONS:
                continue
            text = path.read_text(errors="ignore")
            rel = str(path.relative_to(REPO_ROOT))
            findings.extend(scan_lines(text.splitlines(), rel, fn_index, tracked))
    return findings


# --- allowlist -------------------------------------------------------------


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def line_hash(text: str) -> str:
    return hashlib.sha1(normalize_line(text).encode("utf-8")).hexdigest()


def parse_allowlist_text(text: str) -> set[tuple[str, str, str]]:
    """`file:number:sha1` triples — `#`-comments and blank lines ignored.
    `number` may itself contain no `:` (the shapes this gate matches never
    do), so a 3-way `split(":", 2)` is exact.
    """
    entries: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(":", 2)
        if len(parts) != 3:
            continue
        file_part, number_part, sha_part = parts
        entries.add((file_part, number_part, sha_part))
    return entries


def load_allowlist() -> set[tuple[str, str, str]]:
    if not ALLOWLIST_PATH.exists():
        return set()
    return parse_allowlist_text(ALLOWLIST_PATH.read_text())


def report(findings: list[Finding]) -> int:
    allowlist = load_allowlist()
    unallowlisted = [
        f for f in findings if (f.file, f.number, line_hash(f.line_text)) not in allowlist
    ]

    if not unallowlisted:
        print(
            "doc-numbers-have-producers: OK — "
            f"{len(findings)} measurement-shaped number(s) found, "
            f"all cited or allowlisted."
        )
        return 0

    print("doc-numbers-have-producers: FAIL", file=sys.stderr)
    for f in unallowlisted:
        print(
            f"  {f.file}:{f.line_no}: {f.kind} `{f.number}` has no producer citation "
            "resolvable anywhere in its comment block (see [`fn`] / see mod::fn / "
            "printed by <same> / measured by <path> / [producer: <same>]), and no "
            "`no-producer:` opt-out on its own line.",
            file=sys.stderr,
        )
    print(
        "\ndoc-numbers-have-producers: a measurement-shaped number in a kernel/test doc "
        "comment or assert! message claims an observed run with nothing in the tree that "
        "produces it. Cite the real producer anywhere in the same comment block (`see "
        "[`fn`]`, `see mod::fn`, `printed by <same>`, `measured by <path>`, or `[producer: "
        "<same>]`, where the resolved fn must be a #[test] function or a pub fn under "
        "tests/), or tag the number `no-producer: <reason>` on its own line if it is "
        f"genuinely derived — never add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)}, "
        "which only ever shrinks.",
        file=sys.stderr,
    )
    return 1


def check_allowlist_only_shrinks() -> int:
    """The allowlist tracks PRE-EXISTING debt only and may only shrink.

    Round-3 audit fix: the round-2 version used `check=False` on `git
    fetch` and a single `returncode == 0` test on `git show`, so a FAILED
    fetch (network flake, `origin` misconfigured) and a genuinely ABSENT
    file on `origin/main` were indistinguishable — both silently took the
    "nothing to compare against, pass" branch, meaning a fetch failure
    made this leg fail OPEN instead of closed. Now: `git fetch` raises via
    `check=True`; `git rev-parse --verify origin/main` must succeed (a
    real ref) or the leg FAILS outright with that fact stated; only once
    the ref is confirmed to resolve is "the allowlist file itself doesn't
    exist there" checked via `git cat-file -e`, and ONLY THAT case is
    treated as the legitimate bootstrap (this PR is the one introducing
    the file) — the message states which case fired.
    """
    try:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(
            f"  git fetch origin main failed (exit {exc.returncode}): "
            f"{exc.stderr.strip() if exc.stderr else '<no stderr>'}",
            file=sys.stderr,
        )
        print(
            "\ndoc-number-allowlist-only-shrinks: cannot compare against origin/main "
            "because the fetch itself failed — this fails CLOSED (a network/config "
            "problem is not license to skip the only-shrinks check), not open.",
            file=sys.stderr,
        )
        return 1

    rev = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if rev.returncode != 0:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        print(
            "  origin/main does not resolve after a successful fetch — "
            f"`git rev-parse --verify origin/main` exit {rev.returncode}: "
            f"{rev.stderr.strip()}",
            file=sys.stderr,
        )
        print(
            "\ndoc-number-allowlist-only-shrinks: this is NOT the bootstrap case (that "
            "requires origin/main to resolve, just without the allowlist file) — it is an "
            "unresolvable ref, which fails CLOSED.",
            file=sys.stderr,
        )
        return 1

    file_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", "origin/main:ci/doc_number_allowlist.txt"],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    current = load_allowlist()

    if not file_exists:
        print(
            "doc-number-allowlist-only-shrinks: OK (bootstrap) — origin/main resolves "
            f"({rev.stdout.strip()}) but has no {ALLOWLIST_PATH.relative_to(REPO_ROOT)} "
            f"yet: this branch's {len(current)} entries establish the baseline."
        )
        return 0

    base_text = _run(["git", "show", "origin/main:ci/doc_number_allowlist.txt"])
    base = parse_allowlist_text(base_text)
    added = current - base

    if added:
        print("doc-number-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for entry in sorted(added):
            print(f"  + {':'.join(entry)}", file=sys.stderr)
        print(
            "\ndoc-number-allowlist-only-shrinks: this branch adds a NEW entry to "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}. The allowlist tracks pre-existing "
            "debt from before this gate landed and may never grow. Do not add an entry "
            "here — cite the real producer instead anywhere in the number's comment block "
            "(`see [`fn`]`, `see mod::fn`, `printed by <same>`, `measured by <path>`, or "
            "`[producer: <same>]`), or tag the number `no-producer: <reason>` on its own "
            "line if it is genuinely derived, not measured.",
            file=sys.stderr,
        )
        return 1

    print(
        f"doc-number-allowlist-only-shrinks: OK — {len(current)} entries "
        f"({len(base) - len(current)} shrunk vs origin/main)."
    )
    return 0


# --- self-test ---------------------------------------------------------


def _pick_real_test_fn(fn_index: set[str]) -> str:
    """A REAL `#[test]`/pub-fn-under-tests name, picked from the live
    tree at self-test time — never hardcoded, so this fixture never rots
    as the repo's own test names change. Deterministic (sorted) so the
    self-test's own output is stable across runs.
    """
    if not fn_index:
        raise SystemExit(
            "check_doc_numbers_have_producers self-test: build_test_fn_index() found "
            "ZERO test functions in the tracked tree — cannot build the positive-"
            "resolution fixture. This means either the tree has no #[test] fns at all "
            "(implausible) or build_test_fn_index() itself is broken."
        )
    return sorted(fn_index)[0]


def _scan_fragment(text: str, label: str, fn_index: set[str], tracked: set[str]) -> list[Finding]:
    return scan_lines(text.strip("\n").splitlines(), label, fn_index, tracked)


def self_test() -> int:
    failures: list[str] = []
    fn_index = build_test_fn_index()
    tracked = tracked_files()
    real_fn = _pick_real_test_fn(fn_index)

    def flagged(name: str, text: str) -> None:
        hits = _scan_fragment(text, f"<self-test: {name}>", fn_index, tracked)
        if not hits:
            failures.append(f"self-test FAILED ({name}): expected a finding, got none")

    def clean(name: str, text: str) -> None:
        hits = _scan_fragment(text, f"<self-test: {name}>", fn_index, tracked)
        if hits:
            failures.append(f"self-test FAILED ({name}): expected no finding, got {hits}")

    # --- (A) citation grammar: positive forms, resolved via a REAL test fn ---
    clean(
        "see [`fn`] rustdoc-link form",
        f"""
// measured 5145/16384 elements differed from the eager chain — see [`{real_fn}`]
// for the run that produced this count.
""",
    )
    clean(
        "see mod::path::fn form, resolved by last segment",
        f"""
// measured 5145/16384 elements differed from the eager chain — see
// some::nested::module::{real_fn} for the run that produced this count.
""",
    )
    clean(
        "printed by `fn` form",
        f"""
// 100% (6144/6144) elements bit-match — printed by `{real_fn}` for the run
// that produced this count.
""",
    )
    clean(
        "[producer: mod::fn] tag form",
        f"""
// worst observed 0.796123 cosine mismatch [producer: some::mod::{real_fn}] on
// this fixture.
""",
    )
    clean(
        "measured by <tracked path>",
        """
// 100% (6144/6144) elements bit-match — measured by Cargo.toml, a real
// tracked path (used here only as a stand-in artifact).
""",
    )

    # --- (A) citation grammar: must NOT resolve, even if a real fn exists ---
    flagged(
        "see below must not resolve (fn below is not a #[test]/pub-under-tests fn)",
        """
// measured 87/128 elements differed from the eager chain — see below for
// the run that produced this count.
fn below() {}
""",
    )
    flagged(
        "see table 3 must not resolve (names no real identifier)",
        """
// measured 87/128 elements differed from the eager chain — see table 3
// for the breakdown.
""",
    )
    flagged(
        "see step 2 must not resolve (names no real identifier)",
        """
// measured 87/128 elements differed from the eager chain — see step 2 for
// the setup.
""",
    )

    # --- (B) binding unit is the whole block ---
    clean(
        "citation one sentence earlier in the same block covers a later number",
        f"""
// The bound is verified directly against a real run — see [`{real_fn}`] for
// the harness. Over that run the worst divergence measured is 0.512345 at
// the fixture's peak magnitude, comfortably inside tolerance.
""",
    )
    flagged(
        "a blank-line-separated (different) block is NOT covered by an earlier block's citation",
        f"""
// see [`{real_fn}`] for the harness this whole module relies on.

fn unrelated_marker() {{}}

// measured 87/128 elements differed from the eager chain in a totally
// separate block with no citation of its own.
""",
    )
    clean(
        "no-producer: on the number's own line",
        """
// bf16 mantissa ceiling is 0.780000 observed — no-producer: derived from
// the IEEE 754 bf16 format, not measured.
""",
    )
    flagged(
        "no-producer: on a DIFFERENT line in the SAME block must still flag (line-scoped, not block-scoped)",
        """
// bf16 mantissa ceiling is 0.780000 observed, on this fixture.
// no-producer: derived from the IEEE 754 bf16 format, not measured.
""",
    )

    # --- (C) number shapes ---
    flagged(
        "definitional '=' no longer suppresses a measured value",
        """
// measured on the A100: 3.1e-3 = 0.390000 of the sup; a looser bound let
// a mutation pass.
""",
    )
    flagged(
        "a [`CONST`] doc-link on the line no longer blanket-suppresses it",
        """
// [`SOME_CONST`] aside, the worst observed divergence measured on an
// otherwise-idle box was 0.273456, run three of three.
""",
    )
    flagged(
        "5-6 decimal cosines are now in COSINE_RE's range",
        """
// worst observed cosine mismatch measured on this run: 0.123456.
""",
    )
    clean(
        "N/M with M < 64 stays excluded (index/enumeration, not a mismatch count)",
        """
// measured 29/32 elements differed on this fixture, a small run.
""",
    )
    clean(
        "N/M immediately preceded by '.' stays excluded (decimal-embedded fragment)",
        """
// measured on the A100: swept scaling 1.3/1600 against the fixture and
// observed no divergence worth flagging in this run.
""",
    )
    clean(
        "N/M chain (a/b/c) is fully excluded in both directions",
        """
// measured on a run: n=32/900/1024/6144 elements differed across the
// sweep, a shape enumeration not a single mismatch count.
""",
    )
    clean(
        "N/M immediately preceded by '^' stays excluded (exponent)",
        """
// measured phi(x) = exp(-x^2/6144) on this run, observed to match the
// standard normal density.
""",
    )
    clean(
        "version strings (3- and 2-component) stay excluded",
        """
// candle-kernels-0.11.0 measured this; candle-core 0.11's own run
// observed no divergence worth flagging at all in this fixture.
""",
    )
    flagged(
        "a genuine N/M (M >= 64, no chain/decimal/exponent context) is still caught",
        """
// candle-core 0.11's own run observed no divergence worth flagging, but
// separately measured 128/6144 elements as a genuine finding.
""",
    )

    # --- (F) fn_index has no extra_fns escape hatch in the production path ---
    if is_test_fn("definitely_not_a_real_fn_anywhere", fn_index):
        failures.append(
            "self-test FAILED: is_test_fn resolved a name that isn't in the real fn_index "
            "— there is no extra_fns parameter to explain this; the resolver itself is "
            "broken."
        )

    # --- the eight audit-reworded probes (all MUST be caught) ----------------
    probes = {
        "probe 1: measured X = Y (definitional-= no longer kills it)": """
// The RED drill's fused arm measured on the A100: 67546/262144 elements
// differed = 0.257600 of the total, no producer cited.
""",
        "probe 2: [`CONST`] link no longer blanket-suppresses the line": """
// [`REL_TOL`] context aside, bf16 scored 0.610000 against the f32 truth
// run, versus 0.810000 for the f32-only baseline — neither is cited.
""",
        "probe 3: differ on N/M values, M >= 64": """
// the two arms differ on 176/4096 values in this sweep, uncited.
""",
        "probe 4: percentage measured on a pod": """
// the fused kernel gives a 39% memory reduction measured on that pod,
// with nothing here saying how.
""",
        "probe 5: 5-6 decimal cosine": """
// worst observed cosine mismatch measured on this run: 0.512345, uncited.
""",
        "probe 6: see below (real fn, wrong kind)": """
// measured 512/4096 elements differed on this run — see below for the
// harness.
fn below() {}
""",
        "probe 7: see table 3 (no real identifier)": """
// measured 512/4096 elements differed on this run — see table 3 for the
// per-shape breakdown.
""",
        "probe 8: no-producer one line off in the same block": """
// measured 512/4096 elements differed on this run, uncited on this exact
// line.
// no-producer: this reasoning applies to a DIFFERENT number, not this one.
""",
    }
    for name, text in probes.items():
        flagged(name, text)

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("doc-numbers-have-producers self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "doc-numbers-have-producers self-test: OK — citation grammar (rustdoc-link/path/"
        "bare forms) resolves via a REAL #[test] fn and rejects see-below/table/step; "
        "block-level binding covers a same-block citation and refuses a different block's; "
        "no-producer: stays line-scoped; every shape exclusion (M>=64, decimal/chain/"
        "exponent/n=/version) holds without over-firing; the definitional-= and "
        "[`CONST`]-link exclusions are gone; all eight reworded probes caught."
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()

    findings = scan_tree()
    if findings and "--list-by-file" in argv:
        by_file: dict[str, int] = {}
        for f in findings:
            by_file[f.file] = by_file.get(f.file, 0) + 1
        for file, count in sorted(by_file.items()):
            print(f"{file}: {count}")
    return report(findings)


if __name__ == "__main__":
    sys.exit(main())
