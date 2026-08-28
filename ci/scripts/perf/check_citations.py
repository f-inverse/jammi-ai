#!/usr/bin/env python3
"""Re-resolve every `finetune_step.rs:<n>` / `grad_oracle.rs:<n>` PATH:LINE
citation under `crates/jammi-bench/**`, `ci/scripts/perf/**`, and
`crates/jammi-kernels/artifacts/cuda-runs/**` — against the actual file
content AT HEAD for ordinary living files, or against the CITING artifact's
own recorded `git_sha` for committed evidence (see this doc's own
"Committed artifacts are append-only evidence" section below) — advisory
(i), round-2 audit fix on PR #372; the artifact sha-relative resolution is
the M1b audit round's own fix.

WHY THIS EXISTS: round 1 of this fix round "corrected" a stale citation in
`finetune_step.rs` (originally naming Rust source lines 253 through 264,
"fixed" to name line 290) by eyeballing the diff — but the code had ALREADY
moved again by the time that commit landed, so the "fixed" citation (line
290) was ALSO stale (the real line was 299) the moment it was committed.
`p1_softmax_scale_fold_ab.json` carried this SAME stale line-290 citation in
TWO fields, and `torch_finetune_step.py`/`README.md` independently carried a
DIFFERENT stale citation naming Rust source lines 112 and 233 together
(the second of those two line numbers pointed at unrelated LoRA-builder
code, not the VRAM baseline capture it was meant to cite) that neither round
noticed by eye either. A
citation that is only ever checked "by eye" at commit time is exactly the
kind of claim this repo's own `implementer-acceptance-clause` ("resolvable
citations") exists to stop being trusted on prose alone — this script is the
mechanical re-check, run every CI, not a one-time manual pass.

CONVENTION THIS SCRIPT ENFORCES: every citation in scope must be immediately
preceded by a backtick-quoted CODE IDENTIFIER (allowing only
whitespace/commas/parens/an apostrophe-s/the literal phrase "at the time of
writing" between the identifier's closing backtick and the citation itself —
see `_find_adjacent_identifier`). For example, this crate's own
`finetune_step.rs` module already carries a self-citation of exactly this
resolvable shape, right next to its own `peak.saturating_sub` expression —
see that function's test-suite doc for the literal text this script accepts
as a positive control. A BARE citation with no adjacent identifier (naming
two Rust source lines together with no code quoted next to either one — the
shape this fix replaces) is a HARD FAIL, not a lint warning — it is exactly
the shape that let a citation go stale
unnoticed, since there is nothing in the text itself a script (or a human)
can mechanically check it against. For each citation found, this script:

    1. Resolves `<file>.rs` to its one known location under
       `crates/jammi-bench/src/` (this script only knows about
       `finetune_step.rs`/`grad_oracle.rs` — the two files this crate's own
       fix-round dispatches have named; extending `_KNOWN_FILES` is a one-line
       change if a THIRD `.rs` file starts being cited this way).
    2. Checks the cited LINE NUMBER is in-bounds for that file's CURRENT
       (HEAD) line count.
    3. Checks the adjacent backtick-quoted identifier's text (whitespace-
       normalized) is a SUBSTRING of that exact line's CURRENT content
       (also whitespace-normalized) — this is what actually catches
       "the line number is in-bounds but points at different code now",
       which an in-bounds-only check (or eyeballing the diff) would miss.

## Committed artifacts are append-only evidence, not living prose (M1b audit)

A citation living inside a file under an `artifacts/` directory (any path
segment named exactly `artifacts`, not just this repo's one instance today)
that itself carries a top-level, well-formed `git_sha` is evidence ABOUT
THAT SHA'S TREE — the run the artifact's `_comment`/`provenance` fields
narrate was measured against the code as it stood at `git_sha`, and the
artifact is never edited again to track later refactors (the class this
repo's own `check_cuda_run_artifacts.py` schema already enforces: `git_sha`
is "the measured tip, kept verbatim"). Re-resolving such a citation against
HEAD is a category error, not a stricter check — it demands an edit to
historical evidence every time ANY later commit moves the cited line,
whether or not the artifact's claim about ITS OWN tree was ever wrong. Every
`file.rs:<n>` citation found inside such a file is instead resolved via `git
show <git_sha>:<path>` (this same repo's sha-relative provenance discipline,
`check_cuda_run_artifacts.py`'s ancestry model, reused rather than
reinvented) — a citation that was true at recording time PASSES regardless
of how far the working tree has since drifted; a citation that was NEVER
true even at its own declared sha still FAILS, now correctly attributed to
the artifact's own authoring mistake rather than misreported as "the code
moved". Every OTHER citing file (not under `artifacts/`, or an `artifacts/`
file with no resolvable `git_sha`) keeps the HEAD-resolution behaviour above
unchanged — living docs/scripts describe the code as it IS today, and their
citations should track that.

This needs REAL commit history to mean anything, the identical shallow-
checkout hazard `check_cuda_run_artifacts.py`'s own rule (d) already
documents: `git show <sha>:<path>` on a shallow clone (`actions/checkout`'s
default `fetch-depth: 1`) fails to find a sha outside the single fetched
commit, indistinguishable from a genuinely bad citation without checking
first. `_require_history` checks `git rev-parse --is-shallow-repository`
before the FIRST such lookup and raises one explicit, named `CitationError`
("shallow checkout") instead of N misleading per-citation findings that
would look like real drift — `.github/workflows/ci.yml`'s `citation
resolver` leg is given `fetch_depth: "0"` for exactly this reason (only that
one leg pays the deeper-clone cost; every other leg in that matrix stays at
the normal shallow default).

Run: `python3 ci/scripts/perf/check_citations.py`
Hermetic for every non-artifact citation (reads only files in the working
tree; no network, no build). An artifact-scoped citation additionally shells
out to `git show`/`git rev-parse --is-shallow-repository` against the local
checkout's own object database — still no network, no build, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The repo root every git subprocess call below runs against — a module
# variable (not a hardcoded `REPO_ROOT` reference inside each helper) so
# `test_check_citations.py` can monkeypatch it onto a throwaway `git
# init`'d fixture repo, the SAME pattern this file's own `_KNOWN_FILES`/
# `_SEARCH_ROOTS` already use for isolating a test from this repo's real
# citation inventory.
_GIT_REPO_ROOT = REPO_ROOT

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

SHALLOW_CHECKOUT_MESSAGE = (
    "shallow checkout — sha-relative artifact-citation resolution needs real "
    "commit history; use fetch-depth: 0"
)


class CitationError(Exception):
    """Uncomputable input (a shallow checkout, when sha-relative resolution
    is needed) — fails closed with ONE explicit message, never N misleading
    per-citation findings that would look like real drift."""

# The only `.rs` files this script currently knows how to resolve citations
# against — both live under `crates/jammi-bench/src/`. Add a new mapping
# here (not a heuristic search) the day a THIRD file starts being cited this
# way, so a typo'd filename fails loudly (KeyError-shaped) rather than
# silently resolving to the wrong file.
_KNOWN_FILES = {
    "finetune_step.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "finetune_step.rs",
    "grad_oracle.rs": REPO_ROOT / "crates" / "jammi-bench" / "src" / "grad_oracle.rs",
    # round-4 audit fold-in on PR #372: the determinant tables in
    # `grad_oracle.rs`/`ab_merge.py` cite dozens of `.py:<n>` lines in the
    # torch reference scripts — those citations were NEVER mechanically
    # re-checked (this script only knew about the two `.rs` files above),
    # which is exactly how the `.py` line-drift this round's own audit
    # caught went unnoticed.
    "torch_grad_oracle.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_grad_oracle.py",
    "torch_finetune_step.py": REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_finetune_step.py",
}

# The roots this advisory names, all searched recursively for every file
# (not just `.py`/`.json` -- a `.md` doc citation is just as resolvable
# and just as capable of going stale, see README.md's own citation this
# round fixed).
#
# Unification contract C8.4/NF15: `crates/jammi-kernels/artifacts/cuda-runs`
# joined this tuple in phase 2, the same PR that `git mv`s the two
# `finetune_step_reference.json`/`p1_softmax_scale_fold_ab.json` baselines
# OUT of `crates/jammi-bench/baselines/` and into this directory (contract
# C8) -- without this addition, the moved p1 record's own two citations of
# finetune_step.rs's batched-forward concatenation call site (see that
# record's own `_comment`) would silently drop OUT of this script's coverage
# the moment the move landed (a citation this script used to check would
# simply never be visited again, not a citation that fails loudly), which is
# precisely the "coverage regression" pressure-v2 pin H / NF15 named. (This
# comment deliberately avoids spelling out a bare `file.rs:N` citation of its
# own -- this script scans `ci/scripts/perf/**`, itself included.)
_SEARCH_ROOTS = (
    REPO_ROOT / "crates" / "jammi-bench",
    REPO_ROOT / "ci" / "scripts" / "perf",
    REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs",
)


# --------------------------------------------------------------------------- #
# Artifact sha-relative resolution -- see the module doc's "Committed
# artifacts are append-only evidence" section above.
# --------------------------------------------------------------------------- #
def _run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=_GIT_REPO_ROOT, capture_output=True, text=True)


def _is_shallow_repository() -> bool:
    proc = _run_git(["rev-parse", "--is-shallow-repository"])
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _require_history() -> None:
    """Checked before the FIRST `git show` an artifact-scoped citation
    needs -- one explicit `CitationError` naming the shallow checkout,
    never N misleading per-citation "does not resolve" findings that would
    look like real drift (the same discipline `check_cuda_run_
    artifacts.py`'s `run_gate` already applies to its own ancestry rule).
    """
    if _is_shallow_repository():
        raise CitationError(SHALLOW_CHECKOUT_MESSAGE)


def _artifact_git_sha(path: Path) -> str | None:
    """The `git_sha` this citing file declares as ITS OWN evidence tree, or
    `None` if `path` does not qualify for sha-relative resolution at all
    (not under an `artifacts/` path segment, not JSON, unparsable, not an
    object, or no well-formed top-level `git_sha`). A file that does not
    qualify keeps the ordinary HEAD-resolution behaviour -- this is an
    opt-IN narrowing (only files that both live under `artifacts/` AND
    self-declare a resolved `git_sha` are evidence in the append-only
    sense this function exists to detect), never a heuristic guess.
    """
    if "artifacts" not in path.parts or path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    sha = data.get("git_sha")
    if isinstance(sha, str) and GIT_SHA_RE.match(sha):
        return sha
    return None


def _git_relpath(target_path: Path) -> str | None:
    """`target_path` (a `_KNOWN_FILES` entry, always absolute) expressed
    relative to `_GIT_REPO_ROOT` in POSIX form, the shape `git show
    <sha>:<relpath>` needs -- `None` if `target_path` does not sit under
    `_GIT_REPO_ROOT` at all (never raises; the caller turns this into an
    ordinary `Violation` instead of an uncomputable-input `CitationError`,
    since an unresolvable relpath is a real finding about THIS citation,
    not a global "history is missing" condition).
    """
    try:
        return target_path.resolve().relative_to(_GIT_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _lines_at_sha(sha: str, relpath: str) -> list[str] | None:
    """`relpath`'s content at `sha`, split into lines -- `None` if `git
    show` cannot read it there (the sha or the path does not resolve in
    this checkout's object database at all; a distinct condition from "the
    citation is stale", which needs a successful read to even evaluate).
    """
    _require_history()
    proc = _run_git(["show", f"{sha}:{relpath}"])
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


# A known filename, a colon, and a line number -- optionally wrapped in a
# matched pair of backticks around the whole citation (both forms are used
# across this repo's own docs; see this script's own module doc for an
# example of each). Deliberately does NOT match the legacy
# comma-continuation shorthand this round's own fix retired (naming a
# second Rust source line by writing only a bare `,:<n>` after the first) --
# that shape now simply fails to match at all past the first number, which
# is intentional: an isolated `,:<n>` fragment, unattached to a
# `<filename>:` prefix, is not a citation this script recognizes at all, so
# a leftover comma-shorthand citation degrades to "only the first number is
# checked" rather than being silently mis-parsed as one unit.
def _citation_re() -> re.Pattern:
    """Built fresh from the CURRENT `_KNOWN_FILES` on every call (never
    compiled once at import time against whatever `_KNOWN_FILES` happened to
    be at that moment) -- `test_check_citations.py` monkeypatches
    `_KNOWN_FILES` per test to isolate one fixture at a time, and a
    module-load-time-frozen pattern would silently keep matching only the
    ORIGINAL file list regardless of that patch.
    """
    return re.compile(
        r"`?(?P<file>" + "|".join(re.escape(f) for f in _KNOWN_FILES) + r"):(?P<line>\d+)`?"
    )

# A backtick-quoted span -- candidate identifiers.
_IDENT_RE = re.compile(r"`([^`\n]{1,200})`")

# What is allowed to sit BETWEEN an identifier's closing backtick and the
# citation that names it: whitespace, commas, parens, an apostrophe-`s`, and
# the literal phrase "at the time of writing" (the one hedge phrase this
# repo's own self-citation convention uses, see `finetune_step.rs`'s own
# `vram_sampler_finish_reports_true_delta_not_floored_by_a_baseline_at_the_peak`
# test docstring). Anything else in the gap means the identifier just
# preceding is NOT actually attached to this citation.
_CONNECTOR_RE = re.compile(
    r"^(?:'s)?[\s,\(]*(?:at the time of writing[\s,\(]*)?$", re.IGNORECASE
)

_SEARCH_WINDOW = 300


# A wrapped Rust doc-comment (`/// ...`) or line comment (`// `/`# `)
# continuation onto the next source line -- collapsed to a single space
# before the connector check below, so a citation whose adjacent identifier
# sits one wrapped comment-line above it (e.g. `finetune_step.rs`'s own
# self-citation at the time of writing this script, split across two `///`
# lines) is not penalized for the comment SYNTAX carrying it; only the
# actual prose content of the gap is checked.
_COMMENT_CONTINUATION_RE = re.compile(r"\n[ \t]*(?:///|//|#)?[ \t]*")


def _find_adjacent_identifier(ident_spans: list[re.Match], citation_start: int) -> str | None:
    """`ident_spans` is EVERY backtick-quoted span in the FULL source text
    (`_IDENT_RE.finditer(text)`, computed once per file by the caller) --
    never re-paired from a text[start:end] slice. A slice cut at an
    arbitrary character offset can land INSIDE a real backtick pair (its
    opening backtick outside the slice, only its closing backtick inside),
    which silently shifts which backticks pair with which for every
    subsequent match in that slice -- a dense table row with several
    citations close together can trip this (round-4 audit fold-in on PR
    #372's own row_lengths addition hit it: a `` `...` `` pair straddling
    the 300-char boundary made the NEXT citation's adjacent-identifier
    lookup misfire on a truncated fragment like `') | '`). Operating on
    globally-paired spans and only then filtering to the lookback window
    makes that class of misparse structurally unreachable.
    """
    window_start = max(0, citation_start - _SEARCH_WINDOW)
    candidates = [m for m in ident_spans if m.end() <= citation_start and m.start() >= window_start]
    if not candidates:
        return None
    last = candidates[-1]
    gap = _COMMENT_CONTINUATION_RE.sub(" ", last.string[last.end() : citation_start])
    if _CONNECTOR_RE.match(gap):
        return last.group(1)
    return None


def _normalize(s: str) -> str:
    return " ".join(s.split())


class Violation:
    def __init__(self, source_path: Path, line_no: int, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        try:
            rel = self.source_path.relative_to(REPO_ROOT)
        except ValueError:
            # A path outside REPO_ROOT (e.g. a test fixture in a throwaway
            # tempdir) -- print it absolute rather than raising, so a test
            # driving `main()`/`check_file()` against a fixture never trips
            # on formatting a violation it deliberately provoked.
            rel = self.source_path
        return f"{rel}:{self.line_no}: {self.message}"


def _iter_source_files():
    seen = set()
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".json", ".md", ".rs", ".sh"):
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def check_file(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations

    # This citing file's own evidence sha, if it is a qualifying artifact --
    # computed ONCE per file (not per citation): every citation inside the
    # SAME artifact resolves against the SAME tree.
    artifact_sha = _artifact_git_sha(path)
    ident_spans = list(_IDENT_RE.finditer(text))

    for m in _citation_re().finditer(text):
        cited_file = m.group("file")
        cited_line = int(m.group("line"))
        source_line_no = text.count("\n", 0, m.start()) + 1

        target_path = _KNOWN_FILES[cited_file]

        if artifact_sha is not None:
            relpath = _git_relpath(target_path)
            if relpath is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file} sha-relative to this artifact's own git_sha "
                        f"{artifact_sha}, but {target_path} does not resolve under "
                        f"{_GIT_REPO_ROOT} for `git show`",
                    )
                )
                continue
            target_lines = _lines_at_sha(artifact_sha, relpath)
            if target_lines is None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} sha-relative to this artifact's own "
                        f"recorded git_sha {artifact_sha} (committed evidence is append-only -- "
                        f"never re-resolved against HEAD), but `git show {artifact_sha}:{relpath}` "
                        "could not read that file at that sha",
                    )
                )
                continue
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has "
                        f"{len(target_lines)} lines at this artifact's own git_sha {artifact_sha}",
                    )
                )
                continue
        else:
            if not target_path.exists():
                violations.append(
                    Violation(path, source_line_no, f"cites {cited_file} but {target_path} does not exist")
                )
                continue
            target_lines = target_path.read_text().splitlines()
            if cited_line < 1 or cited_line > len(target_lines):
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} but that file only has {len(target_lines)} lines",
                    )
                )
                continue

        ident = _find_adjacent_identifier(ident_spans, m.start())
        if ident is None:
            violations.append(
                Violation(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} with no resolvable adjacent backtick-quoted "
                    "identifier -- a bare PATH:LINE citation cannot be mechanically re-checked "
                    "(this is exactly the shape that went stale unnoticed on PR #372); rewrite as "
                    f"`` `some_identifier`, {cited_file}:{cited_line} `` naming what is actually at "
                    "that line",
                )
            )
            continue

        cited_line_text = target_lines[cited_line - 1]
        if _normalize(ident) not in _normalize(cited_line_text):
            if artifact_sha is not None:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line "
                        f"reads {cited_line_text.strip()!r} at this artifact's own recorded git_sha "
                        f"{artifact_sha} -- the citation was never true at the tree this evidence "
                        "describes (committed artifacts are append-only, sha-relative evidence -- "
                        "never re-resolve this class of citation against HEAD; the citation's line "
                        "number, or the sha it should have cited, is wrong and needs a hand fix)",
                    )
                )
            else:
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} for identifier {ident!r}, but that line is "
                        f"currently {cited_line_text.strip()!r} -- the citation is STALE (the code moved "
                        "since this was written); re-resolve it against the file at HEAD",
                    )
                )
    return violations


def main() -> int:
    all_violations: list[Violation] = []
    checked = 0
    try:
        for path in _iter_source_files():
            checked += 1
            all_violations.extend(check_file(path))
    except CitationError as exc:
        print(f"check-citations: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if all_violations:
        print("check-citations: FAIL", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(f"check-citations: {checked} file(s) scanned, all PATH:LINE citations resolve "
          "(HEAD for living files, each artifact's own recorded git_sha for committed evidence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
