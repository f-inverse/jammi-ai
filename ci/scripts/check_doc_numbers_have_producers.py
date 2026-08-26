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

WHAT COUNTS AS "measurement-shaped": three number shapes, each REQUIRED to
appear within `ADJACENT_WORDS` (case-insensitive) on the SAME scanned line —
the trigger-word requirement is not narrowed to bare cosines only. A plain
`N/M`, `X%`, or `0.XXX` floating around in math notation, an enumeration
("cell 3/7"), a domain bound ("rank 1/2/3"), a fixed statistical constant
("a 95% CI", "a 90% miscoverage target"), or a defined threshold
("DEFAULT_REGRESSION_THRESHOLD, 30%, see the module docs") is NOT
measurement-shaped by this gate — it carries no claim of an observed run, so
it needs no producer. Only a number sitting next to a word that signals "this
was seen to happen" — {cosine, mismatch, elements, ceiling, bit-match,
diverg} — is treated as a claimed measurement:

  - `N/M` element/mismatch counts:      `\\d+\\s*/\\s*\\d+`
  - percentages:                        `\\d+(\\.\\d+)?\\s*%`
  - bare cosines (e.g. `0.610`,
    `0.796`):                           `0\\.\\d{2,4}`

WHERE: doc comments (`///`, `//!`) and plain `//` comment blocks, plus the
message string of an `assert!`/`assert_eq!`/`assert_ne!`/`debug_assert!`/
`panic!` call (single-line message only — a multi-line `format!` message
inside the macro call is out of scope for this pass, the same "cheap,
visible, first-pass tripwire" cost `check_sqlite_isms.py`'s own doc accepts),
under `crates/jammi-kernels/{src,tests}`, `crates/jammi-encoders/src`,
`crates/jammi-lora/src`, `crates/jammi-bench/src` — `.rs` and CUDA source
(`.cu`/`.cuh`) files, since the class this closes was found in a `.cu` file
(`adamw_step.cu`) as often as a `.rs` one.

PRODUCER CITATION: a measurement-shaped number is REQUIRED to have, within
`WINDOW` lines above or below it in the SAME file, one of:

  - `see <fn>`       — `<fn>` must exist as `fn <fn>` somewhere in the repo
                        (a real test/helper function, found by grep).
  - `printed by <fn>` — same existence check.
  - `measured by <path>` — `<path>` must be a file `git ls-files` tracks
                        (a committed artifact, e.g. a CUDA-run JSON under
                        `crates/jammi-kernels/artifacts/cuda-runs/`).
  - `no-producer: <reason>` — an explicit, reviewable opt-out for a number
    that is genuinely DERIVED, not measured (e.g. `2^-7` bf16 ULP, a
    hand-computed bound) — never a way to silence a real unproduced claim.

Everything else in the citation space (a bare "see the module docs", "see
that commit's message") does NOT satisfy this gate: it names no
grep-verifiable function and no tracked artifact, so a reader still cannot
resolve the claim. That is intentional, not a bug — those are exactly the
un-resolvable citations this gate exists to catch.

FAIL-CLOSED, file:line, offending number. A committed, append-only-shrinking
allowlist (`ALLOWLIST_PATH`) carries lines already on `main` at the time this
gate was authored, one `path:line:number` per line — the allowlist may only
shrink (a future PR either fixes the line or the reviewer is knowingly
carrying a stale entry forward; it is never grown to silence a NEW instance
of this class).

Run: `python3 ci/scripts/check_doc_numbers_have_producers.py`
Self-test (positive: an unproduced measurement is flagged; negative: a cited
one and a no-producer-tagged derived constant are clean):
`python3 ci/scripts/check_doc_numbers_have_producers.py --self-test`
Hermetic: reads only files in the working tree (or an in-memory fragment
under `--self-test`) plus `git ls-files`/`git grep`-equivalent local greps;
no network, no build.
"""

from __future__ import annotations

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

SCAN_EXTENSIONS = {".rs", ".cu", ".cuh", ".h", ".cpp"}

ALLOWLIST_PATH = REPO_ROOT / "ci" / "doc_number_allowlist.txt"

# How many lines above/below a flagged number a producer citation (or a
# no-producer opt-out) may sit and still count as "adjacent" to it.
WINDOW = 12

# A measurement-shaped number is only in scope when one of these words
# appears (case-insensitive) within `TRIGGER_WINDOW` lines of the number —
# this is what turns a plain N/M or X% (math notation, an enumeration, a
# fixed threshold) into a CLAIMED measurement. `diverg` is a stem (matches
# diverge/diverges/divergence/diverging); `bit-match` matches
# bit-match/bit-matched/BIT-match. `TRIGGER_WINDOW` (not the full `WINDOW`
# used for the producer-citation search) stays small and deliberately
# proximate — prose commonly wraps a phrase like "5145/16384 `m` elements
# differed" across a line boundary in a doc comment, so same-line-only
# adjacency would miss it, but going as wide as `WINDOW` would rope in
# unrelated trigger words from the surrounding paragraph and over-flag plain
# math notation sitting nearby in the same comment block.
TRIGGER_WINDOW = 3
ADJACENT_WORDS = ["cosine", "mismatch", "elements", "ceiling", "bit-match", "diverg"]
_ADJACENT_RE = re.compile("|".join(re.escape(w) for w in ADJACENT_WORDS), re.IGNORECASE)

NM_RE = re.compile(r"\b\d{1,7}\s*/\s*\d{1,7}\b")
PCT_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,4})?\s*%")
COSINE_RE = re.compile(r"(?<!\d)0\.\d{2,4}\b")

SEE_RE = re.compile(r"\bsee\s+`?([A-Za-z_][A-Za-z0-9_]*)\b")
PRINTED_BY_RE = re.compile(r"\bprinted by\s+`?([A-Za-z_][A-Za-z0-9_]*)\b")
MEASURED_BY_RE = re.compile(r"\bmeasured by\s+`?([^\s`,;]+)")
NO_PRODUCER_RE = re.compile(r"\bno-producer:")

ASSERT_CALL_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|panic)!\s*\("
)
STRING_LITERAL_RE = re.compile(r'"')


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def tracked_files() -> set[str]:
    return set(_run(["git", "ls-files"]).splitlines())


def fn_exists(name: str, cache: dict[str, bool], extra: set[str] | None = None) -> bool:
    """True if `fn <name>` (or `fn <name><...>` for a generic fn) is grep-
    findable anywhere in the repo's tracked `.rs` files, OR `name` is in
    `extra` — the self-test's injected set of fixture-local fn names, so the
    self-test's positive/negative fragments never depend on a REAL function
    name in the live tree (which would rot as the repo evolves out from
    under this gate's own fixtures).
    """
    if extra and name in extra:
        return True
    if name in cache:
        return cache[name]
    pattern = rf"\bfn\s+{re.escape(name)}\s*[(<]"
    try:
        out = subprocess.run(
            ["git", "grep", "-InE", pattern, "--", "*.rs"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        found = out.returncode == 0 and bool(out.stdout.strip())
    except FileNotFoundError:
        found = False
    cache[name] = found
    return found


@dataclass(frozen=True)
class NumberHit:
    line_no: int
    text: str  # the matched number's literal text
    kind: str  # "N/M" | "%" | "cosine"


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


def find_number_hits(line: str, trigger_context: str) -> list[NumberHit]:
    """Measurement-shaped numbers on `line`, each REQUIRED to sit within
    `TRIGGER_WINDOW` lines (i.e. somewhere in `trigger_context`) of one of
    `ADJACENT_WORDS` (see module doc's WHAT COUNTS AS section) — a plain N/M
    or X% with no nearby trigger word is not measurement-shaped, it is math
    notation / an enumeration / a fixed constant.
    """
    if not _ADJACENT_RE.search(trigger_context):
        return []
    hits: list[NumberHit] = []
    for m in NM_RE.finditer(line):
        hits.append(NumberHit(0, m.group(0), "N/M"))
    for m in PCT_RE.finditer(line):
        hits.append(NumberHit(0, m.group(0), "%"))
    for m in COSINE_RE.finditer(line):
        # A cosine-shaped float that is really the fractional part of an N/M
        # or % match already reported above is not a separate finding —
        # dedupe by containment isn't needed here because COSINE_RE requires
        # a literal "0." prefix that NM_RE/PCT_RE do not themselves consume
        # as a standalone token; leave as-is (a conservative over-report, if
        # any, is a visible false positive a human reads at file:line, never
        # a silent miss).
        hits.append(NumberHit(0, m.group(0), "cosine"))
    return hits


@dataclass(frozen=True)
class Finding:
    file: str
    line_no: int
    number: str
    kind: str


def scan_lines(
    lines: list[str],
    file_label: str,
    cache: dict[str, bool],
    extra_fns: set[str] | None = None,
) -> list[Finding]:
    """`lines` is 0-indexed; line numbers reported are 1-indexed."""
    n = len(lines)
    scan_mask = [is_scan_line(l) for l in lines]

    # Pre-index every citation/opt-out line so the WINDOW search is O(1) per
    # flagged number instead of re-scanning the window's text each time.
    citation_lines: dict[int, bool] = {}
    for i, line in enumerate(lines):
        ok = False
        m = SEE_RE.search(line)
        if m and fn_exists(m.group(1), cache, extra_fns):
            ok = True
        m = PRINTED_BY_RE.search(line)
        if m and fn_exists(m.group(1), cache, extra_fns):
            ok = True
        m = MEASURED_BY_RE.search(line)
        if m:
            candidate = m.group(1).strip("`").rstrip(".,;")
            tracked = cache.setdefault("__tracked__", None)
            if tracked is None:
                tracked = tracked_files()
                cache["__tracked__"] = tracked
            if candidate in tracked or (REPO_ROOT / candidate).exists():
                ok = True
        if NO_PRODUCER_RE.search(line):
            ok = True
        citation_lines[i] = ok

    findings: list[Finding] = []
    for i, line in enumerate(lines):
        if not scan_mask[i]:
            continue
        trig_lo = max(0, i - TRIGGER_WINDOW)
        trig_hi = min(n - 1, i + TRIGGER_WINDOW)
        trigger_context = "\n".join(lines[trig_lo : trig_hi + 1])
        for hit in find_number_hits(line, trigger_context):
            lo = max(0, i - WINDOW)
            hi = min(n - 1, i + WINDOW)
            covered = any(citation_lines.get(j, False) for j in range(lo, hi + 1))
            if not covered:
                findings.append(Finding(file_label, i + 1, hit.text, hit.kind))
    return findings


def scan_tree() -> list[Finding]:
    cache: dict = {}
    findings: list[Finding] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            raise SystemExit(f"check_doc_numbers_have_producers: scan root not found: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_EXTENSIONS:
                continue
            text = path.read_text(errors="ignore")
            rel = str(path.relative_to(REPO_ROOT))
            findings.extend(scan_lines(text.splitlines(), rel, cache))
    return findings


def load_allowlist() -> set[tuple[str, int, str]]:
    if not ALLOWLIST_PATH.exists():
        return set()
    entries: set[tuple[str, int, str]] = set()
    for line in ALLOWLIST_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(":", 2)
        if len(parts) != 3:
            continue
        file_part, line_part, number_part = parts
        try:
            entries.add((file_part, int(line_part), number_part))
        except ValueError:
            continue
    return entries


def report(findings: list[Finding]) -> int:
    allowlist = load_allowlist()
    unallowlisted = [f for f in findings if (f.file, f.line_no, f.number) not in allowlist]

    by_file: dict[str, int] = {}
    for f in findings:
        by_file[f.file] = by_file.get(f.file, 0) + 1

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
            f"(see <fn> / printed by <fn> / measured by <path>) within {WINDOW} lines, "
            "and no `no-producer:` opt-out.",
            file=sys.stderr,
        )
    print(
        "\ndoc-numbers-have-producers: a measurement-shaped number in a kernel/test doc "
        "comment or assert! message claims an observed run with nothing in the tree that "
        "produces it. Cite the real producer (`see <fn>`, `printed by <fn>`, `measured by "
        "<path>`), tag it `// no-producer: <reason>` if it is genuinely derived (not "
        f"measured), or add it to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} if it is a "
        "pre-existing entry being tracked down separately (the allowlist only shrinks).",
        file=sys.stderr,
    )
    return 1


# --- self-test -------------------------------------------------------------

POSITIVE_FRAGMENT = """
// measured on jammi-a100: 5145/16384 `m` elements differed from the eager
// CUDA chain at t=3 with nonzero prior moments — no producer cited anywhere
// near this claim.
"""

NEGATIVE_CITED_FRAGMENT = """
// 100% (6144/6144) elements bit-match — see check_bit_identity_fixture for
// the run that produced this count.
"""
CITED_FIXTURE_FNS = {"check_bit_identity_fixture"}

NEGATIVE_NO_PRODUCER_FRAGMENT = """
// bf16 mantissa ceiling is 2^-7 = 0.0078125, i.e. 0.78% — no-producer:
// derived from the IEEE 754 bf16 format, not measured.
"""

NEGATIVE_NO_TRIGGER_WORD_FRAGMENT = """
// the significance level is a 95% CI (DEFAULT_REGRESSION_THRESHOLD, 30%,
// see the module docs) and the RNG split is 1/8, both fixed constants with
// no claim of an observed run.
"""

NEGATIVE_UNRESOLVABLE_CITATION_FRAGMENT = """
// 26% of elements diverge — see the module docs for context.
"""


def self_test() -> int:
    failures: list[str] = []
    cache: dict = {}

    hits = scan_lines(POSITIVE_FRAGMENT.strip("\n").splitlines(), "<self-test: positive>", cache)
    if not hits:
        failures.append(
            "self-test FAILED: an unproduced measurement (5145/16384, adjacent to "
            "'elements') was not flagged"
        )

    hits = scan_lines(
        NEGATIVE_CITED_FRAGMENT.strip("\n").splitlines(),
        "<self-test: cited>",
        cache,
        CITED_FIXTURE_FNS,
    )
    if hits:
        failures.append(
            "self-test FAILED: a number with a resolvable `see <fn>` citation to a real "
            f"fn was flagged: {hits}"
        )

    hits = scan_lines(
        NEGATIVE_NO_PRODUCER_FRAGMENT.strip("\n").splitlines(),
        "<self-test: no-producer opt-out>",
        cache,
    )
    if hits:
        failures.append(
            f"self-test FAILED: a `no-producer:`-tagged derived constant was flagged: {hits}"
        )

    hits = scan_lines(
        NEGATIVE_NO_TRIGGER_WORD_FRAGMENT.strip("\n").splitlines(),
        "<self-test: no trigger word>",
        cache,
    )
    if hits:
        failures.append(
            "self-test FAILED: fixed constants/notation with no adjacent trigger word were "
            f"flagged: {hits}"
        )

    hits = scan_lines(
        NEGATIVE_UNRESOLVABLE_CITATION_FRAGMENT.strip("\n").splitlines(),
        "<self-test: unresolvable citation>",
        cache,
    )
    if not hits:
        failures.append(
            "self-test FAILED: 'see the module docs' (names no grep-verifiable fn, no "
            "tracked artifact) incorrectly satisfied the producer requirement"
        )

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("doc-numbers-have-producers self-test: FAIL", file=sys.stderr)
        return 1

    print(
        "doc-numbers-have-producers self-test: OK — unproduced measurement flagged, "
        "resolvable `see`/`no-producer:` fragments clean, unresolvable citation still flagged."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    findings = scan_tree()
    by_file: dict[str, list[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)
    if by_file and "--list-by-file" in sys.argv[1:]:
        for file, fs in sorted(by_file.items()):
            print(f"{file}: {len(fs)}")
    return report(findings)


if __name__ == "__main__":
    sys.exit(main())
