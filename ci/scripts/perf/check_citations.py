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

## The discriminator is ANCESTRY, never local object presence (post-#411 CI fix)

The FIRST version of this section resolved every artifact citation via `git
show <git_sha>:<path>` unconditionally, on the theory that a well-typed
`git_sha` is enough. It is not: `check_cuda_run_artifacts.py`'s own schema
(this same repo's ancestry model) already distinguishes a `git_sha` that IS
an ancestor of `HEAD` from one that legitimately never can be again — a
tip that was squash-merged (that gate's `merged_as`/`merged_via_pr` pair),
or, for a handful of pre-schema artifacts, a `git_sha` grandfathered into
`LEGACY_NONE_ALLOWLIST` with no ancestry claim made about it at all (see
that file's own module doc). `bf8e807` (the P1 softmax-fold artifact this
round's own audit traced) is exactly the second shape: it predates this
repo's merge-commit discipline, was never an ancestor of `main` even the
day it was recorded, and `check_cuda_run_artifacts.py` accepts it ONLY
through that reviewed-legacy arm — never by claiming its tree is reachable.
A fresh, fully-fetched clone (`fetch-depth: 0`, real history, zero shallow
ambiguity) can still legitimately never contain `bf8e807`'s objects at all,
because nothing on `main`'s own history line ever points at it. `git show`
unconditionally attempting to read it anyway does not fail closed on that —
it fails OPEN, exactly backwards: on a developer's long-lived local
checkout, an old feature branch (or its reflog) can leave `bf8e807`'s
objects sitting in the local object store even though `main` never
reaches it, so the SAME citation reads GREEN on that machine and RED on a
CI runner's fresh clone. Object presence is an accident of which branches
happened to touch a given checkout, never a property of the citation's own
correctness — the exact fail-open, environment-dependent green this gate
family exists to kill (`check_cuda_run_artifacts.py`'s own rule (d) already
refuses to let ancestry-adjacent questions turn on anything but `git
merge-base`).

Every artifact-scoped citation is instead resolved by a DETERMINISTIC
three-way split, decided by ancestry alone:

  1. **`git_sha` IS an ancestor of `HEAD`** (`git merge-base --is-ancestor
     <sha> HEAD`) and the file reads at that sha — sha-relative resolve, the
     behaviour described above, unchanged.
  2. **Ancestor, but `git show` still fails** (the sha/path genuinely does
     not resolve even though `merge-base` calls it reachable) — an ordinary
     `Violation`, same shape as before; this is a real finding about the
     citation, not an environment question.
  3. **`git_sha` is NOT an ancestor of `HEAD`** — the artifact is historical
     evidence whose tree is not on this branch's history line AT ALL, by
     construction, on every checkout with equal fetch depth. Every citation
     inside it is reported as a NAMED, non-failing **EXEMPT** line (never
     silent, never a `Violation`, never gates CI) stating plainly that it is
     historical prose this script cannot mechanically verify from this
     repository line, and that the artifact's own acceptance rests on
     `check_cuda_run_artifacts.py`'s reviewed-legacy arm
     (`LEGACY_NONE_ALLOWLIST`) instead — never on this citation resolving.
     This branch NEVER falls back to HEAD resolution and NEVER attempts
     `git show` at all: doing either would silently reintroduce the exact
     object-presence dependency this fix closes (a `git show` that happens
     to succeed on one machine's stale object store and fail on another's
     clean clone is not more informative than skipping it honestly).

Ancestry itself needs the SAME shallow guard case 1 does — `git merge-base
--is-ancestor` on a shallow clone reads back every sha as a false
non-ancestor, indistinguishable from a genuine case 3 without checking
first (`check_cuda_run_artifacts.py`'s own rule (d) documents this exact
trap for its ancestry check). `_require_history` therefore runs BEFORE any
ancestry conclusion is drawn, not just before the case-1 `git show` —
computed once per artifact file (the same sha decides every citation inside
it), never per citation.

## A maintainer guide's citations are resolved by FULL PATH, not by basename

`_KNOWN_FILES` is a BASENAME map, and that shape is load-bearing for the
roots above: a handful of files are cited by bare filename dozens of times,
so a hand-registered mapping is what makes a typo'd filename fail loudly
(`KeyError`-shaped) instead of silently resolving to the wrong file. It does
not scale past a handful. `docs/maintainer/pod-build-guide.md` alone cites
NINE distinct scripts and `docs/maintainer/cuda-kernel-guide.md` another
seven source files; registering every one of them by basename would also
make the map ambiguous the first time two directories hold a same-named file
(`layer_norm.rs` exists under both `jammi-kernels/src/ops/` and
`jammi-encoders/src/`, today).

Citing files under `_DOC_SEARCH_ROOTS` (the maintainer guides) therefore get
a SECOND, additional citation form: a repo-root-relative FULL PATH, e.g.
`` `rp_tree_dir` (`ci/scripts/runpod_lib.sh:644`) ``. The full-path form is
resolved directly against the working tree, and the loud-failure property
the basename map provides is preserved by a DIFFERENT mechanism rather than
dropped: a full path that does not exist under `REPO_ROOT` is a
`Violation` ("cites <path> but that file does not exist"), never a silently
skipped citation. Both forms are subject to the IDENTICAL adjacent-
identifier rule and in-bounds check — the full-path form buys a citation no
registration step, never a weaker check. A full path is constrained to the
prefixes this repo's own sources live under (`_FULL_PATH_ROOT_PREFIXES`) so
a citation of a VENDORED third-party file (`candle-core-0.11.0/src/op.rs:…`,
which `cuda-kernel-guide.md` legitimately cites and which is not in this
tree at all) is not matched and then reported as a missing path.

The basename form still applies everywhere it did before, including inside
the doc roots. Where both forms match the same text — a full path whose last
component happens to be a registered basename — the FULL-PATH match wins and
the basename match nested inside it is dropped, so one citation is never
reported twice.

DOCUMENTED RESIDUAL, stated rather than implied: the full-path form is
enabled for `_DOC_SEARCH_ROOTS` citing files ONLY, not for the
`_SEARCH_ROOTS` files above. Those roots do carry full-path citations of
their own (in `ci/scripts/perf/finetune_ab.sh`, `gpu_inference_ab.sh`,
`test_finetune_ab_disable_op_keys.py`, and three `cuda-runs` artifacts),
and a survey run at the time this form was added found roughly twenty of
them bare or stale. Turning the form on for those roots is a real coverage
extension and a real fix-up unit — it is NOT done here silently as a side
effect of a docs change, and this paragraph exists so the gap is a named
scope boundary rather than an accidental blind spot a later reader mistakes
for coverage.

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

# The citing roots whose files ALSO get the full-path citation form (see the
# module doc's "A maintainer guide's citations are resolved by FULL PATH"
# section). A separate tuple from `_SEARCH_ROOTS`, not an entry appended to
# it, precisely because the two are not interchangeable: everything here is
# additionally scanned for full-path citations, and every existing test that
# monkeypatches `_SEARCH_ROOTS` onto a throwaway fixture keeps meaning
# exactly what it meant before.
_DOC_SEARCH_ROOTS = (
    REPO_ROOT / "docs" / "maintainer",
)

# A full-path citation must start with one of these — the roots this repo's
# OWN sources live under. Without this constraint a maintainer guide's
# legitimate citation of a VENDORED third-party file (cuda-kernel-guide.md
# cites `candle-core-0.11.0/src/op.rs` and `.../cpu_backend/mod.rs`, neither
# of which is in this tree) would match and then be reported as a missing
# path — a false finding about a citation that is correct as written.
_FULL_PATH_ROOT_PREFIXES = ("ci/scripts/", "crates/", "docs/", ".github/")

# The suffixes a full-path citation may name. Deliberately explicit (not
# "any extension"): the trailing `:<digits>` is the only structural signal
# that a path-shaped token is a citation at all, and a permissive suffix
# set turns ordinary prose mentioning a path plus a number into one.
_FULL_PATH_SUFFIXES = ("sh", "py", "rs", "cu", "toml", "md", "yml", "yaml", "json")


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
    """Checked before the FIRST ancestry conclusion OR `git show` an
    artifact-scoped citation needs -- one explicit `CitationError` naming
    the shallow checkout, never N misleading per-citation "does not
    resolve" findings that would look like real drift (the same discipline
    `check_cuda_run_artifacts.py`'s `run_gate` already applies to its own
    ancestry rule). Ancestry needs this EXACT same guard case 1 does: `git
    merge-base --is-ancestor` on a shallow clone reads back every sha as a
    false non-ancestor, indistinguishable from a genuine case-3 (non-
    ancestor, EXEMPT) citation without checking first.
    """
    if _is_shallow_repository():
        raise CitationError(SHALLOW_CHECKOUT_MESSAGE)


def _is_ancestor(sha: str, target: str = "HEAD") -> bool:
    """Whether `sha` is an ancestor of `target` -- the ONE deterministic
    discriminator for artifact-citation resolution (never local object
    presence, which differs by checkout history and is exactly the
    environment-dependent green this function replaces). A non-zero exit
    covers both "genuinely not an ancestor" and "not a resolvable object at
    all in this checkout" -- both correctly fall through to case 3 (EXEMPT)
    rather than attempting a `git show` whose success would depend on
    which stale branches this particular checkout happens to still hold.
    """
    proc = _run_git(["merge-base", "--is-ancestor", sha, target])
    return proc.returncode == 0


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


def _full_path_citation_re() -> re.Pattern:
    """The FULL-PATH citation form, built fresh from the CURRENT
    `_FULL_PATH_ROOT_PREFIXES`/`_FULL_PATH_SUFFIXES` on every call — same
    reason `_citation_re` is: `test_check_citations.py` monkeypatches those
    module variables per test, and a module-load-time-frozen pattern would
    silently keep matching only whatever they held at import time.

    A leading `(?<![A-Za-z0-9_./-])` guard keeps a prefix from matching in
    the MIDDLE of a longer path (`vendor/crates/foo.rs:1` must not be read
    as a `crates/`-rooted citation of a file this repo does not have).
    """
    prefixes = "|".join(re.escape(p) for p in _FULL_PATH_ROOT_PREFIXES)
    suffixes = "|".join(re.escape(s) for s in _FULL_PATH_SUFFIXES)
    return re.compile(
        r"(?<![A-Za-z0-9_./-])`?(?P<path>(?:" + prefixes + r")[A-Za-z0-9_./-]+\.(?:"
        + suffixes + r")):(?P<line>\d+)`?"
    )


def _cited_targets(path: Path, text: str) -> list[tuple[int, str, Path, int]]:
    """Every citation in `text`, as `(match_start, label, target_path,
    cited_line)`, ordered by position.

    The basename form always applies; the full-path form applies only when
    the CITING file lives under `_DOC_SEARCH_ROOTS` (module doc's
    "resolved by FULL PATH" section, including its documented residual).
    Where a full-path match SPANS a basename match — a path whose last
    component is a registered `_KNOWN_FILES` name — the nested basename
    match is dropped, so one citation is reported once, by its more
    specific form.
    """
    spans: list[tuple[int, int, str, Path, int]] = []
    if _is_under(path, _DOC_SEARCH_ROOTS):
        for m in _full_path_citation_re().finditer(text):
            rel = m.group("path")
            spans.append(
                (m.start(), m.end(), f"{rel}:{m.group('line')}", REPO_ROOT / rel, int(m.group("line")))
            )
    full_spans = [(s, e) for (s, e, _l, _t, _n) in spans]
    for m in _citation_re().finditer(text):
        if any(s <= m.start() and m.end() <= e for (s, e) in full_spans):
            continue
        spans.append(
            (
                m.start(), m.end(),
                f"{m.group('file')}:{m.group('line')}",
                _KNOWN_FILES[m.group("file")],
                int(m.group("line")),
            )
        )
    spans.sort(key=lambda t: t[0])
    return [(s, label, target, n) for (s, _e, label, target, n) in spans]


def _is_under(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False

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


class Exemption:
    """Case 3 of the artifact-citation three-way split (see module doc):
    the citing artifact's own `git_sha` is NOT an ancestor of `HEAD` --
    historical evidence off this branch's history line, reported NAMED and
    printed unconditionally (never silent), but NEVER a `Violation` -- it
    does not gate CI, and no attempt is made to resolve it (against HEAD OR
    via `git show`, which would reintroduce the exact object-presence
    dependency this class exists to avoid). Deliberately a SEPARATE class
    from `Violation` (not a subclass) -- `main()` keeps the two in
    independent lists throughout, so an `Exemption` can never be
    accidentally counted toward a FAIL by a future refactor that forgets
    to filter by type.
    """

    def __init__(self, source_path: Path, line_no: int, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        try:
            rel = self.source_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.source_path
        return f"{rel}:{self.line_no}: {self.message}"


def _iter_source_files():
    seen = set()
    for root in (*_SEARCH_ROOTS, *_DOC_SEARCH_ROOTS):
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


def _check_file_impl(path: Path) -> tuple[list[Violation], list[Exemption]]:
    """The real per-file scan: `check_file` (kept, unchanged signature, for
    every existing caller/test that only wants the failing findings) is a
    thin wrapper discarding the second element. `main()` calls this
    directly so it can also print the non-failing EXEMPT lines the
    three-way split's case 3 produces (see module doc).
    """
    violations: list[Violation] = []
    exemptions: list[Exemption] = []
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations, exemptions

    # This citing file's own evidence sha, if it is a qualifying artifact --
    # computed ONCE per file (not per citation): every citation inside the
    # SAME artifact resolves against the SAME tree, by the SAME ancestry
    # verdict.
    artifact_sha = _artifact_git_sha(path)
    ident_spans = list(_IDENT_RE.finditer(text))

    # Shallow-first ordering (module doc's "The discriminator is ANCESTRY"
    # section): the ancestry test itself needs the SAME shallow guard case
    # 1's `git show` does, checked ONCE per file, before any ancestry
    # conclusion is drawn -- never per citation, never deferred until a
    # case-1 `git show` is about to run.
    sha_is_ancestor: bool | None = None
    if artifact_sha is not None:
        _require_history()
        sha_is_ancestor = _is_ancestor(artifact_sha)

    for citation_start, cited_label, target_path, cited_line in _cited_targets(path, text):
        # `cited_file` keeps naming the citation exactly as the doc wrote it
        # (a bare basename, or a full path) so every message below quotes
        # back the text a reader has to go find and fix.
        cited_file = cited_label.rsplit(":", 1)[0]
        source_line_no = text.count("\n", 0, citation_start) + 1

        if artifact_sha is not None and not sha_is_ancestor:
            # Case 3: the artifact's own git_sha is NOT an ancestor of
            # HEAD -- historical evidence off this branch's history line,
            # by construction, on every checkout with equal fetch depth.
            # NEVER resolved (not against HEAD, not via `git show`, which
            # would silently reintroduce the exact object-presence
            # dependency this split exists to close) -- named, printed,
            # non-failing.
            exemptions.append(
                Exemption(
                    path, source_line_no,
                    f"cites {cited_file}:{cited_line} sha-relative to this artifact's own git_sha "
                    f"{artifact_sha}, which is NOT an ancestor of HEAD -- historical evidence "
                    "predating this repo's merge-commit discipline (typically squash-merged away), "
                    "so its tree is not reachable on this branch's history line at all, on any "
                    "checkout with equal fetch depth. EXEMPT: this citation is historical prose "
                    "that cannot be mechanically verified from this repository line; the artifact's "
                    "own acceptance rests on check_cuda_run_artifacts.py's reviewed-legacy arm "
                    "(LEGACY_NONE_ALLOWLIST), never on this citation resolving.",
                )
            )
            continue

        if artifact_sha is not None:
            # Case 1 (sha_is_ancestor is True here): sha-relative resolve.
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
                # Case 2: an ancestor sha whose `git show` still fails --
                # a real finding about the citation, not an environment
                # question (ancestry already ruled that out).
                violations.append(
                    Violation(
                        path, source_line_no,
                        f"cites {cited_file}:{cited_line} sha-relative to this artifact's own "
                        f"recorded git_sha {artifact_sha} (an ancestor of HEAD; committed evidence "
                        "is append-only -- never re-resolved against HEAD), but "
                        f"`git show {artifact_sha}:{relpath}` could not read that file at that sha",
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

        ident = _find_adjacent_identifier(ident_spans, citation_start)
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
    return violations, exemptions


def check_file(path: Path) -> list[Violation]:
    violations, _exemptions = _check_file_impl(path)
    return violations


def main() -> int:
    all_violations: list[Violation] = []
    all_exemptions: list[Exemption] = []
    checked = 0
    try:
        for path in _iter_source_files():
            checked += 1
            v, e = _check_file_impl(path)
            all_violations.extend(v)
            all_exemptions.extend(e)
    except CitationError as exc:
        print(f"check-citations: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if all_exemptions:
        print(
            f"check-citations: {len(all_exemptions)} EXEMPT citation(s) -- historical evidence "
            "whose git_sha is not an ancestor of HEAD, never mechanically verified from this "
            "repository line (see check_cuda_run_artifacts.py's reviewed-legacy arm instead):"
        )
        for e in all_exemptions:
            print(f"  - EXEMPT: {e}")

    if all_violations:
        print("check-citations: FAIL", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(f"check-citations: {checked} file(s) scanned, all PATH:LINE citations resolve "
          "(HEAD for living files, each artifact's own recorded git_sha for committed evidence "
          f"reachable from HEAD; {len(all_exemptions)} exempt as non-ancestor legacy evidence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
