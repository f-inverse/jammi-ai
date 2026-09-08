#!/usr/bin/env python3
"""Two mechanical, grep-shaped static assertions over every tracked producer
script under `ci/scripts/perf/` — the class two round-N audit findings on
`perf/unification-p2` named: a dry-run-only test knob left live in a REAL
run is a bypass, not a fixture, and a cross-check added to two of four
producers that share the exact same hole is a partial fix, not a closed
class.

## (A) FAKE-knob inertness

Any tracked `.sh` under `ci/scripts/` that references an environment
variable whose name contains `FAKE` (the shape `stacked_sweep.sh`'s
`SWEEP_FAKE_BIN_SHA` set, contract C5.2 — a test-only injection knob for
exercising the provenance-mismatch refusal path without a GPU or a real
binary) must ALSO contain an explicit REFUSAL guard: a line that tests the
knob is set (`-n "${<VAR>...}"`), tests some `*DRY_RUN*` variable `!= "1"`,
and `exit`s — appearing BEFORE every other textual use of that variable in
the file. A knob referenced with no such guard, or referenced (even in a
comment) before its own guard, cannot be trusted to be inert in a real run.

## (B) Producer parity — every jammi-bench-invoking producer carries the
`$BIN provenance` cross-check

Unification contract C5.1: "Every shell/Python producer cross-checks `$BIN
provenance`'s build_sha against the sha it is about to stamp before writing
a GREEN leg." Mechanically: every tracked `.sh` under `ci/scripts/perf/`
whose text names a `jammi-bench` BINARY PATH (a variable assignment ending
`/jammi-bench` — the shape every producer's own `$BIN`/`$B`/`$JAMMI_BIN`
takes; never a source-tree reference like `crates/jammi-bench/...`) must
also contain BOTH `provenance` and `build_sha` somewhere in its text — the
two tokens the cross-check itself is built from
(`"$BIN" provenance` / `["build_sha"]`). This is the exact `grep -l
jammi-bench` / `grep -l provenance` methodology the audit that found the
gap used — reproduced here as a standing gate, not a one-time hand check,
so a FIFTH producer landing tomorrow cannot silently reopen the class.

Both are deliberately mechanical (name/pattern presence), not a semantic
understanding of the guard's control flow — the same "grep for the shape,
not the meaning" stance `check_ci_guard_wiring.py`'s own module doc states.

## Comment handling — a bash-aware lexer, applied uniformly across (A), (B), (C)

Every token/regex test in all three checks below runs against a LEXED
reading of each line, never the raw line: bash treats an unquoted,
unescaped, word-initial `#` as the start of a comment running to end of
line, so a trailing `# ...` on an otherwise-real line of code must not be
able to satisfy (B)'s `provenance`/`build_sha` cross-check, nor forge a
guard shape (A)/(C) would otherwise accept. Conversely, a `#` that is not
bash's comment marker at all (`$#`, `${#arr}`, `${var#pattern}`, a
`$((10#$N))` base-literal, or any other `#` that is not the first
character of a shell word) must never be mistaken for one — truncating
real code as though it were a trailing comment.

`_lex_stream` (below) is a single, complete state machine over an entire
file's lines, threading single-/double-quote and `$(`/`$((`/`${` nesting
state ACROSS line boundaries (never reset per line): a physical line that
opens a quote or bracket construct it does not itself close is a genuine,
common bash shape (a `python3 -c '...'` payload piped across several
physical lines, a `"$(...)" ` capturing a multi-line command
substitution), and getting this wrong in either direction is a real,
exploitable gap — a bug this exact file's differential corpus scan (see
`self_test`'s `--self-test` arm) found LIVE on two tracked lines
(`pod_push_stamp.sh`, `test_pod_substrate.sh`: a line that legitimately
CLOSES a quote OPENED several lines earlier was misread as OPENING a new
one, so the line's own real trailing comment was never stripped — a
fail-open that could let arbitrary prose satisfy a guard or a cross-check)
and three more (`runpod_lib.sh` at two lines, `test_pod_substrate.sh` at a
third: a `#` inside `$((10#$VAR))` or `${var#pattern}` was misread as a
comment marker, truncating real code — a false-strip that could hide a
real guard from this scanner entirely) — five pinned lines in total (see
the five named cases below).

Two lines the lexer cannot fully resolve on their own are marked
UNDECIDABLE rather than guessed at: a `#` that is not bash's comment
marker AND not one of the recognized `#`-is-never-a-comment contexts
(inside `$((...))` or `${...}`) — i.e. a `#` that is simply mid-word — and
the terminal case of a file ending while a construct this lexer opened is
still unresolved. Every consumer below reads an UNDECIDABLE line through
ONE of two accessors, chosen by what that specific check is testing:

  - `_trigger_text` (used for "does this line REFERENCE token X" —
    finding a knob's use sites, finding a `jammi-bench` binary-path
    assignment that pulls a script into (B)'s scope): an UNDECIDABLE line
    reads as its RAW, unstripped text — the WIDEST possible scope, so a
    real use this lexer could not fully parse is never silently dropped.
  - `_guard_text` (used for "does this line SATISFY a guard/cross-check
    shape" — (A)/(C)'s `!= "1"` + `exit` guard, (B)'s `provenance`/
    `build_sha` tokens, the `if`/`fi` depth walks that bound a guard's own
    block): an UNDECIDABLE line contributes NO text at all — a shape this
    lexer could not fully resolve must never be able to forge a passing
    guard or cross-check.

A decidable line (the overwhelming majority) reads identically through
either accessor: its comment (if any) is already excised, so there is
nothing left for the trigger/guard distinction to change.

`self_test`'s `--self-test` arm additionally runs a full corpus scan: every
tracked `.sh` line under `ci/scripts/` is lexed by `_lex_stream` AND by a
second, independently-written IMPLEMENTATION of the exact same frame model
(`_independent_lex_stream` — full `$(`/`$((`/`${` frame typing and the
`ARITH`/`PARAM` `#`-exemption included, coded as a token-regex walk rather
than the primary's per-character state machine, and with its OWN,
separately-written heredoc-opener recognition — see its own doc for why a
"simpler, conceptually different" description would be false and why the
two lexers share no heredoc-detection code), and the two are asserted to
agree on every line. The five lines above are pinned as explicit fixture
cases with their expected classification, alongside the general
corpus-wide agreement check, so a future TRANSCRIPTION regression in
either implementation — a typo, a dropped case, an off-by-one — shows up
as a named, attributable failure rather than a silent corpus-wide
disagreement; it is not, and was never meant to be, an oracle for a
different CONCEPT of the rules.

## (C) `*_DRY_RUN_*` knob admissibility — a second knob shape beyond `*FAKE*`

(A) only ever looks at names containing the literal substring `FAKE`. A
SECOND shape of dry-run-only test lever carries no `FAKE` in its name at
all — `profile_421_legs.sh`'s `PROFILE_421_LEGS_DRY_RUN_EXTRA_REQUESTED_KEY`
/ `_TRUNCATE_CORPUS_VAR` and `lora_bias_ab.sh`'s `LORA_BIAS_AB_DRY_RUN_FAIL_OP`
/ `_FAIL_PREDICATE` — every one named `<PREFIX>_DRY_RUN_<SUFFIX>`, i.e. the
producer's OWN dry-run toggle (`<PREFIX>_DRY_RUN`) with a real suffix
appended, structurally invisible to (A)'s `FAKE`-only name filter. (C)
below closes that gap.

A knob in this class is admissible by EITHER of two routes (never both
required):

  1. **Containment.** Every non-comment, non-self-defaulting read site of
     the knob sits textually inside SOME `if [ "$<PREFIX>_DRY_RUN" = "1" ]`
     -guarded region — most commonly a heredoc body written while
     `<PREFIX>_DRY_RUN=1` (`profile_421_legs.sh`'s `fake_bench.sh` stub,
     `lora_bias_ab.sh`'s `fake_bench.sh` stub): the knob is only ever READ
     by code that cannot execute unless the toggle is already on, so no
     separate preflight refusal is possible OR needed — there is no "real
     run" code path that could ever reach it.
  2. **Preflight refusal**, the exact (A) shape generalized off the `FAKE`
     name requirement: a line combining the knob, the governing
     `<PREFIX>_DRY_RUN` toggle, `!=`, and `"1"`, that `exit`s, appearing
     BEFORE every other use — `profile_421_legs.sh`'s own
     `PROFILE_421_LEGS_DRY_RUN_TRUNCATE_CORPUS_VAR` guard (its action
     truncates a corpus file in place, which WOULD corrupt a real leg, so
     it needs the same "refuse before any leg runs" contract (A) already
     enforces for a `FAKE` knob).

A "self-defaulting" read (`VAR="${VAR:-default}"`, the ordinary bash
env-var-with-default idiom every knob in this file uses to declare its
own default up front) is never counted as a read site: it captures the
ambient value (or a fallback) into a same-named local, with no
consequence of its own — the knob's real effect is wherever that value is
later dereferenced for real, which the containment/refusal check inspects
independently. This test is itself computed off `_guard_text` (never
`_trigger_text`): an UNDECIDABLE line must never be granted the
self-default EXEMPTION (which would let this scanner skip checking it
entirely) — it is always treated as a real, uncovered read site instead.

Block extent (both for the (C) containment check above and reused nowhere
else) is computed HEREDOC-AWARE: a heredoc body between a `<<[-]TERM`
opener and its bare-`TERM` terminator line is treated as opaque data, never
inspected for `if`/`fi` tokens of its own — exactly how bash itself treats
it. A naive per-line `if`/`fi` depth counter that did NOT skip heredoc
bodies would run straight past its block's real closing `fi`: both
`profile_421_legs.sh`'s and `lora_bias_ab.sh`'s DRY_RUN stub heredocs embed
a `python3 -c '...'` payload whose OWN `if`/`else` statements never close
with a bash `fi` at all (Python doesn't have one) — those bare `if` tokens
would inflate the depth counter with nothing to bring it back down,
so the walker would search past the real `fi` chasing python conditionals
that can never satisfy it. `_lex_stream` independently treats a heredoc
body as opaque for its OWN comment-boundary purposes too (lexed line by
line with fresh, throwaway quote state): the prose inside a heredoc-embedded
script or comment routinely contains ordinary English contractions
("gpu-dev.sh's own", "doesn't") whose apostrophes must never be read as
REAL, cross-line-persisting bash quote characters that could otherwise
corrupt every line of live code that follows the heredoc's closing
terminator.

Both block-extent walkers ((A)/(C)'s shared `_guard_block_lines` and (C)'s
own `_heredoc_aware_block_extent`, via their shared `_if_taken_branch_
extent`) window on the TAKEN branch only: an `if`/`fi` depth walk that
additionally stops the instant it sees an `else`/`elif` token at the
guard's OWN depth (never a nested `if`'s). A guard's condition selects
exactly one branch to execute; a `!= "1"` refusal's `exit` sitting only in
a dead `else` arm never actually refuses anything, and a `= "1"`
containment guard's read sitting only in its dead `else`/`elif` arm was
never actually contained by it — crediting either would prove a bypass is
inert when it is provably live in a REAL run. A heredoc-body line can never
itself be treated as this guard's own opener either: `_dry_run_true_block_
intervals` and (A)/(C)'s own preflight-refusal detection both skip any
line `_lex_stream` marked `in_heredoc_body` before matching the guard
shape, so a stub script's heredoc payload that happens to contain the
literal text of a real bash guard can never open a phantom "covered"
interval from inside opaque data.

Run: `python3 ci/scripts/perf/check_producer_provenance_gates.py`
Self-test (RED cases for (A), (B) and (C), on throwaway fixture files, plus
a full corpus scan cross-checking the lexer against a second, independent
implementation):
`python3 ci/scripts/perf/check_producer_provenance_gates.py --self-test`
Hermetic: reads tracked files via `git ls-files` only (no network, no
build, no GPU).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[3]
# Fail loud rather than scan-zero-silently: this file lives at
# `ci/scripts/perf/check_producer_provenance_gates.py`, three directories
# below the repo root, and every OTHER sibling script under `ci/scripts/
# perf/` already uses `parents[3]` for exactly this reason (a prior
# `parents[2]` resolved to `<repo>/ci` instead — `git ls-files <prefix>` run
# with THAT as `cwd` looks for `ci/scripts/**` under `<repo>/ci/ci/scripts/
# **`, which never exists, so both `run_gate` and this module's own
# `self_test`'s "the real tree is clean" end-to-end arm passed VACUOUSLY —
# zero files scanned, zero findings, reads as PASS). A silent scan-zero is
# worse than a loud crash: this assertion makes a future re-introduction of
# that mistake (or a refactor that moves this file another directory deep)
# fail on the very first line of `main()`/`self_test()` that touches
# `REPO_ROOT`, not silently downstream as an empty findings list that looks
# identical to "everything is fine".
# An explicit `if`/`raise`, never a bare `assert` -- `assert` is stripped
# entirely under `python -O`, which would silently disable this exact
# anti-vacuity guard (the one thing standing between a wrong `parents[N]`
# and a scan-zero-silently PASS) in exactly the deployment shape that
# removes the safety net without removing the code path it protects.
if not (REPO_ROOT / "Cargo.toml").is_file():
    raise AssertionError(
        f"REPO_ROOT resolved to {REPO_ROOT}, which has no Cargo.toml -- "
        "parents[N] is wrong for this file's depth under the repo root"
    )
PERF_DIR = REPO_ROOT / "ci" / "scripts" / "perf"

FAKE_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]*FAKE[A-Z0-9_]*)\b")
DRY_RUN_VAR_RE = re.compile(r"[A-Z][A-Z0-9_]*DRY_RUN")
# `/jammi-bench` NOT immediately followed by another `/` — the BINARY path
# shape (`.../release/jammi-bench"`, `.../jammi-bench` end-of-token), never
# the CRATE source-tree shape (`crates/jammi-bench/reference/...`, which
# also contains the bare substring `/jammi-bench` but is followed by `/`).
BIN_ASSIGN_RE = re.compile(r"/jammi-bench(?!/)")

# (C) — a `<PREFIX>_DRY_RUN_<SUFFIX>` test knob: the producer's OWN
# `<PREFIX>_DRY_RUN` toggle with a REAL suffix appended (`_EXTRA_REQUESTED_
# KEY`, `_FAIL_OP`, ...). Requires at least one char after the second
# underscore so the BARE toggle itself (`PROFILE_421_LEGS_DRY_RUN`,
# `MANIFEST_DRY_RUN`) never self-matches as its own sub-knob.
DRY_RUN_KNOB_RE = re.compile(r"\b([A-Z][A-Z0-9_]*_DRY_RUN_[A-Z0-9_]+)\b")


def _tracked_sh_under(repo_root: Path, prefix: str) -> list[Path]:
    proc = subprocess.run(["git", "ls-files", prefix], cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`git ls-files {prefix}` failed: {proc.stderr.strip()}")
    return sorted(
        repo_root / rel
        for rel in proc.stdout.splitlines()
        if rel.endswith(".sh")
    )


# ---------------------------------------------------------------------------
# The comment/quote lexer.
# ---------------------------------------------------------------------------

# The heredoc TERMINATOR shape (`[-]TERM`, `[-]'TERM'`, `[-]"TERM"`) that
# follows a `<<` operator ALREADY recognized, at a position the lexer itself
# has already established is a genuine top-level heredoc opener (never a
# raw-line search — see `_lex_one_line`'s own `<<` handling, the ONLY call
# site). Anchored via `Pattern.match(line, pos)` (matches only exactly at
# `pos`, never scans ahead), so this never independently decides WHETHER a
# `<<` is a heredoc opener — only what its terminator name is once the lexer
# has already decided it is one. `_heredoc_aware_block_extent` no longer
# re-detects heredoc openers at all: it consumes the per-line `in_heredoc_
# body` spans `_lex_stream` already computed.
_HEREDOC_OPEN_AT_RE = re.compile(r"(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")

# A `#` is bash's comment marker only when it is the first character of a
# shell word: the start of the line, or immediately preceded by whitespace
# or one of these token delimiters.
_WORD_INITIAL_DELIMS = (" ", "\t", ";", "|", "&", "(")

_IF_FI_TOKEN_RE = re.compile(r"\b(if|fi)\b")
# `else`/`elif` included alongside `if`/`fi`: both block-extent walkers below
# must stop at the guard's OWN `else`/`elif` (see `_if_taken_branch_extent`)
# so an `if [ ... ]; then <taken> else <dead> fi` guard's window covers only
# the branch that actually executes when the guard's condition is true,
# never the sibling branch that never runs.
_IF_FI_ELSE_ELIF_TOKEN_RE = re.compile(r"\b(if|fi|else|elif)\b")
_GUARD_BLOCK_MAX_SCAN = 40
_BLOCK_MAX_SCAN = 4000


class _Frame:
    """One entry of `_lex_one_line`'s nesting stack. `kind` is one of `SQ`
    (single-quoted), `DQ` (double-quoted), `CMD` (`$(...)`), `ARITH`
    (`$((...))`), or `PARAM` (`${...}`). `paren_depth` tracks BARE `(`/`)`
    nesting inside a `CMD`/`ARITH` frame (a subshell `(cmd)` inside a
    command substitution, or ordinary grouping parens inside an arithmetic
    expression) so that an inner paren pair does not prematurely close the
    frame's own `)`/`))`."""

    __slots__ = ("kind", "paren_depth")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.paren_depth = 0


class _LineLex(NamedTuple):
    """One physical line's lex result. `code` is the line's content up to
    (never including) a recognized comment start, or the full line when no
    comment was found. `undecidable` is True when this line's own
    classification could not be fully resolved (see module doc) — every
    consumer reads `code`/`undecidable` through `_trigger_text`/
    `_guard_text`, never directly. `in_heredoc_body` is True for a line
    `_lex_stream` classified as opaque heredoc-body data (between a `<<
    [-]TERM` opener line, exclusive, and its bare-`TERM` terminator line,
    inclusive) — the span `_heredoc_aware_block_extent` consumes directly
    instead of re-detecting heredoc openers itself."""

    code: str
    undecidable: bool
    in_heredoc_body: bool = False


def _lex_one_line(line: str, stack: list[_Frame]) -> tuple[str, bool, str | None]:
    """Advance `stack` (the CROSS-LINE nesting state -- mutated in place,
    shared across every line of one file) through exactly one physical
    line, returning `(code, undecidable, heredoc_open)` for THIS line.
    `heredoc_open` is the terminator name of a `<<[-]TERM` heredoc opener
    recognized DURING this line's walk, or `None` if none was. See module
    doc for the full rule set; in short:

    - Backslash escapes the next character in every state (including
      inside single quotes -- a deliberate simplification of real bash,
      which treats `\\` as fully literal inside `'...'`; the "subset that
      matters" here is never fooled by an escaped quote/hash, in exchange
      for not perfectly modeling that one single-quote corner case).
    - `'`/`"` open/close single-/double-quoted regions; `$(`, `$((`, `${`
      open a nested `CMD`/`ARITH`/`PARAM` frame (from top level OR from
      inside a double-quoted string OR from inside another such frame --
      real, common bash nesting, e.g. a command substitution's OWN
      argument being a double-quoted string). A bare `'`/`$(`/`$((`/`${`
      has NO special meaning while the top of the stack is `SQ` or `DQ`
      other than the specific close/nest cases each state recognizes below
      (single quotes are 100% literal; double quotes recognize only their
      own close and a nested substitution).
    - `#` inside `ARITH` or `PARAM` is NEVER a comment marker (arithmetic's
      `10#$N` base-literal syntax, parameter expansion's `#`/`##` prefix
      operators) -- regardless of what precedes it.
    - `#` anywhere else starts a comment only when word-initial (see
      `_WORD_INITIAL_DELIMS`); a `#` that is mid-word and not otherwise
      exempted returns UNDECIDABLE for the whole line rather than guessing.
    - A line ending with the stack's TOP frame still `SQ` is ALSO
      UNDECIDABLE: a single-quoted region spanning multiple physical
      lines is always genuinely opaque, unparsed data from bash's own
      perspective (an embedded `python3 -c '...'`/`bash -c '...'` script,
      arbitrary prose) -- this scanner must never read that data's own
      `#`/`exit`/`!=`/`"1"` substrings as though they were this file's own
      top-level guard code. A multi-line DOUBLE-quoted or `$(...)`-nested
      span is NOT flagged this way: `"$(...)"` spanning several physical
      lines is this codebase's single most common multi-line shape (a
      captured command's output), and it is ordinary, fully-tracked real
      code, not opaque literal data.
    - `<<`/`<<-` is a heredoc-opener TOKEN this lexer recognizes ONLY when
      encountered at genuine top level (`top == "TOP"` -- stack completely
      empty, never inside a quote or a `CMD`/`ARITH`/`PARAM` frame: a `<<`
      inside `'<<REMOTE'` or `"$(... <<REMOTE ...)"` is data/nested-command
      text, not this line's own heredoc redirection) AND only when it is
      not the leading two characters of a `<<<` here-string operator (a
      bare `<<<` is never treated as a heredoc opener at all -- checked
      BEFORE attempting the terminator parse, so `<<<foo`'s inner `<<` can
      never be mistaken for one either). This is a product of the lexer's
      OWN character position, never a regex re-scan of the finished line --
      the exact class of bug (a raw-line regex matching a `<<` sitting
      inside an already-closed quote, or inside a `<<<` operator) that is
      live on `test_gpu_prove_lane.sh:93`'s
      `grep -n '<<REMOTE' "$PROVE_SH"` (a single-quoted, fully top-level-
      resolved-by-line-end argument, not a real heredoc opener at all)."""
    n = len(line)
    i = 0
    comment_at: int | None = None
    undecidable = False
    heredoc_open: str | None = None
    while i < n:
        c = line[i]
        if c == "\\":
            if i + 1 < n:
                i += 2
                continue
            # A trailing, unescaped backslash at end-of-line is an ordinary
            # bash line-continuation (the newline is spliced away): every
            # real occurrence in this codebase is `... token \` followed by
            # an indented continuation, never a mid-word join, so this is
            # treated as a plain character and the NEXT physical line's own
            # word-initial check decides independently -- never an
            # UNDECIDABLE bailout, which would blank every multi-line
            # `if [ ... ] \` / `&& [ ... ]` guard's own text.
            i += 1
            continue
        top = stack[-1].kind if stack else "TOP"
        if top == "SQ":
            if c == "'":
                stack.pop()
            i += 1
            continue
        if top == "DQ":
            if c == '"':
                stack.pop()
                i += 1
                continue
            if c == "$" and i + 1 < n:
                if line[i + 1 : i + 3] == "((":
                    stack.append(_Frame("ARITH"))
                    i += 3
                    continue
                if line[i + 1] == "(":
                    stack.append(_Frame("CMD"))
                    i += 2
                    continue
                if line[i + 1] == "{":
                    stack.append(_Frame("PARAM"))
                    i += 2
                    continue
            i += 1
            continue
        # TOP / CMD / ARITH / PARAM -- code-like contexts: real shell
        # syntax, never opaque literal data.
        if c == "'":
            stack.append(_Frame("SQ"))
            i += 1
            continue
        if c == '"':
            stack.append(_Frame("DQ"))
            i += 1
            continue
        if c == "$" and i + 1 < n:
            if line[i + 1 : i + 3] == "((":
                stack.append(_Frame("ARITH"))
                i += 3
                continue
            if line[i + 1] == "(":
                stack.append(_Frame("CMD"))
                i += 2
                continue
            if line[i + 1] == "{":
                stack.append(_Frame("PARAM"))
                i += 2
                continue
        if top == "TOP" and c == "<" and i + 1 < n and line[i + 1] == "<":
            if i + 2 < n and line[i + 2] == "<":
                # `<<<` here-string operator -- never a heredoc opener,
                # and its inner `<<` (positions i+1, i+2) must never be
                # re-examined as one either.
                i += 3
                continue
            m = _HEREDOC_OPEN_AT_RE.match(line, i + 2)
            if m:
                if heredoc_open is None:
                    heredoc_open = m.group(3)
                i = m.end()
                continue
            # `<<` with no parseable terminator following (malformed, or a
            # shape outside this lexer's scope) -- ordinary characters,
            # never a heredoc opener, never an error.
            i += 2
            continue
        if top in ("CMD", "ARITH") and c == "(":
            stack[-1].paren_depth += 1
            i += 1
            continue
        if top in ("CMD", "ARITH") and c == ")":
            frame = stack[-1]
            if frame.paren_depth > 0:
                frame.paren_depth -= 1
                i += 1
                continue
            if top == "ARITH" and i + 1 < n and line[i + 1] == ")":
                stack.pop()
                i += 2
                continue
            stack.pop()
            i += 1
            continue
        if top == "PARAM" and c == "}":
            stack.pop()
            i += 1
            continue
        if c == "#":
            if top in ("ARITH", "PARAM"):
                i += 1
                continue
            prev = line[i - 1] if i > 0 else None
            if i == 0 or prev in _WORD_INITIAL_DELIMS:
                comment_at = i
                break
            undecidable = True
            break
        i += 1
    code = line[:comment_at] if comment_at is not None else line
    if stack and stack[-1].kind == "SQ":
        undecidable = True
    return code, undecidable, heredoc_open


def _lex_stream(lines: list[str]) -> list[_LineLex]:
    """`_lex_one_line`, threaded across every line of a file: the nesting
    stack persists from one line to the next (never reset), so a construct
    opened on one physical line and closed on a later one is tracked
    correctly across the boundary. A heredoc body (`<<[-]TERM` opener up to
    its bare-`TERM` terminator line) is the one exception: it is lexed with
    a FRESH, throwaway stack per line (matching how `_heredoc_aware_block_
    extent` now reads it back out via `in_heredoc_body`) so that prose
    inside an embedded script or comment can never leak cross-line quote
    state into the real code that follows the heredoc's close. The heredoc
    opener itself is a TOKEN `_lex_one_line` recognizes off its own
    character position (top level, never inside a quote/frame, never the
    `<<<` here-string operator -- see its own doc), never a regex re-scan
    of the finished line; a candidate is only honored once this line's OWN
    nesting has ALSO fully settled back to top level by line's end (a
    `<<EOF` recognized while some OTHER construct opened earlier on the
    same line remains unresolved past it is not this scanner's concern).

    `unresolved_since` tracks the line index where the currently-still-open
    frame stack OR heredoc most recently transitioned from resolved to
    unresolved (reset to `None` the moment both fully resolve). If the file
    ends with EITHER still unresolved, EVERY line from `unresolved_since`
    through the last line is marked UNDECIDABLE -- not just the final line:
    a construct that never resolves by end of file is a shape this lexer's
    "subset that matters" does not fully model (unmatched, an unterminated
    heredoc, or a bash feature outside its scope, e.g. backtick command
    substitution or `$'...'` ANSI-C quoting), and NONE of the lines inside
    that unresolved span -- not only the last one -- can be trusted to have
    been read as real, resolved code."""
    stack: list[_Frame] = []
    out: list[_LineLex] = []
    heredoc_term: str | None = None
    unresolved_since: int | None = None
    for idx, line in enumerate(lines):
        if heredoc_term is not None:
            if line.strip() == heredoc_term:
                heredoc_term = None
                unresolved_since = None
            code, undecidable, _ = _lex_one_line(line, [])
            out.append(_LineLex(code=code, undecidable=undecidable, in_heredoc_body=True))
            continue
        was_open = bool(stack)
        code, undecidable, heredoc_open = _lex_one_line(line, stack)
        now_open = bool(stack)
        if not was_open and now_open:
            unresolved_since = idx
        elif was_open and not now_open:
            unresolved_since = None
        if heredoc_open is not None and not stack:
            heredoc_term = heredoc_open
            if unresolved_since is None:
                unresolved_since = idx
        out.append(_LineLex(code=code, undecidable=undecidable))
    if (stack or heredoc_term is not None) and out and unresolved_since is not None:
        for i in range(unresolved_since, len(out)):
            prev = out[i]
            out[i] = _LineLex(code=prev.code, undecidable=True, in_heredoc_body=prev.in_heredoc_body)
    return out


def _trigger_text(raw_line: str, lex: _LineLex) -> str:
    """Widest-scope reading, for a "does this line REFERENCE token X" test
    (a knob's use sites, a `jammi-bench` binary-path assignment): an
    UNDECIDABLE line still counts its full RAW text, so a real use this
    lexer could not fully resolve is never silently dropped."""
    return raw_line if lex.undecidable else lex.code


def _guard_text(lex: _LineLex) -> str:
    """Narrowest-scope reading, for a "does this line SATISFY a guard or
    cross-check shape" test: an UNDECIDABLE line contributes NO text at
    all, so a shape this lexer could not fully resolve can never forge a
    passing guard or cross-check."""
    return "" if lex.undecidable else lex.code


def _independent_lex_stream(lines: list[str]) -> list[_LineLex]:
    """A second, independently-written IMPLEMENTATION of the exact same
    frame model `_lex_stream` implements -- SQ/DQ quoting plus `$(`/`$((`/
    `${` nesting, the `ARITH`/`PARAM` `#`-is-never-a-comment exemption, and
    a top-level-only `<<[-]TERM` heredoc opener that is never the `<<<`
    here-string operator -- coded as a TOKEN-DRIVEN scan
    (`_INDEPENDENT_TOKEN_RE.finditer`, jumping from match to match) instead
    of the primary's character-by-character `while i < n` walk. This is
    NOT a conceptually different or deliberately-simplified lexer -- an
    earlier version of this docstring claimed one (two plain quote
    booleans, no frame typing, no `ARITH`/`PARAM` exemption at all), but
    that variant does not match the code below (which has always tracked
    `B`/`A`/`P` frames and the `ARITH`/`PARAM` exemption -- see `top in
    ("A", "P")` below) and, run for real, disagrees with the primary on 21
    real corpus lines. What this function actually is: a hand-transliterated
    re-implementation of the SAME rules via a genuinely different
    mechanism (regex-token jumps rather than a per-character `if`/`elif`
    chain), so a TRANSCRIPTION bug in either implementation -- a typo, a
    dropped case, an off-by-one -- shows up as a corpus disagreement,
    rather than testing two different CONCEPTS of what the rules should
    be. It shares the word-initial delimiter set with the primary (a
    settled, independently-obvious constant -- what a shell-word delimiter
    is -- not part of the comment-decision logic under test) but has its
    OWN, separately-written heredoc-opener detection (see `_INDEPENDENT_
    TOKEN_RE`'s `<<<`/`<<-?` alternatives and `_independent_heredoc_term`
    below) -- no shared regex constant with the primary's `_lex_one_line`,
    because a SHARED heredoc-detection constant
    is exactly where a bug in one implementation can hide from this
    cross-check."""
    stack: list[list] = []
    out: list[_LineLex] = []
    heredoc_term: str | None = None
    unresolved_since: int | None = None
    for idx, line in enumerate(lines):
        if heredoc_term is not None:
            if line.strip() == heredoc_term:
                heredoc_term = None
                unresolved_since = None
            code, undecidable, _ = _independent_lex_one_line(line, [])
            out.append(_LineLex(code=code, undecidable=undecidable, in_heredoc_body=True))
            continue
        was_open = bool(stack)
        code, undecidable, heredoc_open = _independent_lex_one_line(line, stack)
        now_open = bool(stack)
        if not was_open and now_open:
            unresolved_since = idx
        elif was_open and not now_open:
            unresolved_since = None
        if heredoc_open is not None and not stack:
            heredoc_term = heredoc_open
            if unresolved_since is None:
                unresolved_since = idx
        out.append(_LineLex(code=code, undecidable=undecidable))
    if (stack or heredoc_term is not None) and out and unresolved_since is not None:
        for i in range(unresolved_since, len(out)):
            prev = out[i]
            out[i] = _LineLex(code=prev.code, undecidable=True, in_heredoc_body=prev.in_heredoc_body)
    return out


# Every character this scanner treats specially, as ONE alternation instead
# of the primary's per-character `if`/`elif` chain: an escape pair, a
# quote, a bracket opener (`$((`, tried before the shorter `$(` so the
# arithmetic form is never mis-tokenized as a command substitution plus a
# stray `(`), the `<<<` here-string operator (tried before the shorter
# `<<-?` so a genuine here-string's leading two `<` are never themselves
# mis-tokenized as a heredoc opener), a heredoc opener candidate (`<<`/
# `<<-`), a bare paren (nested grouping inside `$(...)`/`$((...))`), a
# bracket closer, or a `#`. Everything between matches is ordinary text
# this scanner does not need to look at at all. Deliberately its OWN
# alternation, sharing no heredoc-recognizing pattern with the primary's
# `_HEREDOC_OPEN_AT_RE` -- see `_independent_lex_stream`'s own doc for why.
_INDEPENDENT_TOKEN_RE = re.compile(r"\\.|\\\Z|'|\"|\$\(\(|\$\(|\$\{|<<<|<<-?|\(|\)|\}|#")


def _independent_heredoc_term(line: str, after: int) -> tuple[str, int] | None:
    """Manually scans `line[after:]` (the text immediately following a
    recognized `<<`/`<<-` token) for a heredoc terminator: optional
    leading whitespace, an optional quote character, a bare identifier,
    and an optional matching quote -- returning `(term, end_index)`, or
    `None` if no identifier follows. Written as plain character indexing
    rather than a regex, deliberately unlike the primary's own anchored-
    regex opener parse (`_HEREDOC_OPEN_AT_RE`) -- this lexer's heredoc-
    opener recognition shares no code with the primary's beyond the
    settled, independently-obvious `_WORD_INITIAL_DELIMS` constant."""
    n = len(line)
    j = after
    while j < n and line[j] in " \t":
        j += 1
    quote = line[j] if j < n and line[j] in "'\"" else None
    if quote is not None:
        j += 1
    start = j
    while j < n and (line[j].isalpha() or line[j] == "_" or (j > start and line[j].isdigit())):
        j += 1
    if j == start:
        return None
    term = line[start:j]
    if quote is not None and j < n and line[j] == quote:
        j += 1
    return term, j


def _independent_lex_one_line(line: str, stack: list[list]) -> tuple[str, bool, str | None]:
    """`stack` holds a `[kind, paren_depth]` PAIR per open construct
    (`kind` one of `S` single-quote, `D` double-quote, `B` `$(...)`
    closing on `)`, `A` `$((...))` closing on `))`, `P` `${...}` closing
    on `}`; `paren_depth` counts bare `(`/`)` nesting inside a `B`/`A`
    frame so an inner grouping paren -- a subshell inside a command
    substitution, a parenthesized arithmetic sub-expression -- does not
    prematurely close it), threaded across lines exactly like the
    primary's `_lex_one_line`. Returns `(code, undecidable, heredoc_open)`,
    `heredoc_open` being a terminator name recognized at genuine top level
    (`top is None`, stack empty) via this lexer's OWN token/parse pair
    (`<<<`/`<<-?` tokens plus `_independent_heredoc_term`), never the
    primary's `_HEREDOC_OPEN_AT_RE`."""
    comment_at: int | None = None
    undecidable = False
    heredoc_open: str | None = None
    pos = 0
    for m in _INDEPENDENT_TOKEN_RE.finditer(line):
        if m.start() < pos:
            continue  # inside a token already consumed (an escape pair)
        tok = m.group(0)
        frame = stack[-1] if stack else None
        top = frame[0] if frame else None
        if tok.startswith("\\"):
            pos = m.end()
            continue
        if top == "S":
            # A bare `'` is the ONLY character with any special meaning
            # inside a single-quoted region -- not even `"`/`$(`/`${`.
            if tok == "'":
                stack.pop()
            pos = m.end()
            continue
        if top == "D":
            # Inside a double-quoted region, a bare `'` is ordinary,
            # literal text (real bash: single quotes carry NO special
            # meaning inside double quotes) -- only the closing `"` and a
            # NESTED substitution/expansion opener matter.
            if tok == '"':
                stack.pop()
            elif tok == "$((":
                stack.append(["A", 0])
            elif tok == "$(":
                stack.append(["B", 0])
            elif tok == "${":
                stack.append(["P", 0])
            pos = m.end()
            continue
        # TOP / B / A / P -- code-like contexts: every token is live.
        if tok == "'":
            stack.append(["S", 0])
            pos = m.end()
            continue
        if tok == '"':
            stack.append(["D", 0])
            pos = m.end()
            continue
        if tok == "$((":
            stack.append(["A", 0])
            pos = m.end()
            continue
        if tok == "$(":
            stack.append(["B", 0])
            pos = m.end()
            continue
        if tok == "${":
            stack.append(["P", 0])
            pos = m.end()
            continue
        if tok == "<<<":
            # A `<<<` here-string operand is never a heredoc opener, in
            # any frame -- this token consumes all three characters, so
            # its own inner `<<` is never re-examined as one either.
            pos = m.end()
            continue
        if tok in ("<<", "<<-") and top is None:
            parsed = _independent_heredoc_term(line, m.end())
            if parsed is not None:
                term, end_idx = parsed
                if heredoc_open is None:
                    heredoc_open = term
                pos = end_idx
            else:
                pos = m.end()
            continue
        if tok == "(" and top in ("B", "A"):
            frame[1] += 1
            pos = m.end()
            continue
        if tok == ")" and top == "B":
            if frame[1] > 0:
                frame[1] -= 1
            else:
                stack.pop()
            pos = m.end()
            continue
        if tok == ")" and top == "A":
            if frame[1] > 0:
                frame[1] -= 1
                pos = m.end()
                continue
            # The FIRST of the closing `))`: only pop once two consecutive
            # `)` tokens have been seen; a lone `)` at depth 0 with no
            # second `)` right behind it is treated as the (malformed, not
            # expected in this corpus) close anyway, same as the primary.
            if line[m.end() : m.end() + 1] == ")":
                stack.pop()
                pos = m.end() + 1
            else:
                stack.pop()
                pos = m.end()
            continue
        if tok == "}" and top == "P":
            stack.pop()
            pos = m.end()
            continue
        if tok == "#":
            if top in ("A", "P"):
                pos = m.end()
                continue
            prev = line[m.start() - 1] if m.start() > 0 else None
            if m.start() == 0 or prev in _WORD_INITIAL_DELIMS:
                comment_at = m.start()
                break
            undecidable = True
            break
        pos = m.end()
    code = line[:comment_at] if comment_at is not None else line
    if stack and stack[-1][0] == "S":
        undecidable = True
    return code, undecidable, heredoc_open


def _if_taken_branch_extent(
    lines: list[str], lex: list[_LineLex], start_idx: int, max_scan: int, heredoc_aware: bool
) -> tuple[int, int]:
    """Shared walker behind `_guard_block_lines` and `_heredoc_aware_block_
    extent` (see their own docs for the two callers' distinct purposes,
    caps, and heredoc handling): the inclusive `(start, end)` line-index
    range of ONLY the TAKEN branch — the `then` arm — of the `if [...];
    then ... [else ...|elif ...] fi` block whose OWN condition line is
    `lines[start_idx]`.

    A `bash`-`if`/`fi` DEPTH walk (nested `if`s inside the taken branch
    both increment and later decrement the same counter, so a nested
    conditional does not prematurely end the scan), but one that ALSO stops
    the instant it sees an `else`/`elif` token at the guard's OWN depth
    (`depth == 1` — still directly inside the opening `if`, never a nested
    one: a nested `if`'s own `else`/`elif` sits at `depth >= 2` and is
    correctly ignored). A guard's condition selects exactly ONE branch to
    execute; code sitting in the sibling `else`/`elif` arm never runs when
    the guard fires, so it must never be credited as though it did — an
    `exit` living only in a dead `else` arm does not refuse anything (a
    read living only in a dead `else`/`elif` arm is likewise never
    "contained" by a `= "1"` guard that never took that branch).

    Bounded at `max_scan` lines so a malformed/never-closed `if` cannot make
    this loop unbounded — an unterminated block still returns everything up
    to the cap, which keeps every caller's own "`exit`/read must be found
    inside this window" check fail-closed. `if`/`fi`/`else`/`elif` tokens
    are counted off `_guard_text(lex[i])`, never the raw line: a comment's
    own bare "if"/"else" (this codebase's own prose uses all of these
    constantly) cannot perturb the depth walk, and an UNDECIDABLE line
    contributes no tokens either (the same fail-closed reading every guard/
    satisfaction test in this module uses). When `heredoc_aware` is set, a
    line `_lex_stream` already marked `in_heredoc_body` is opaque data —
    never inspected for any of these tokens, exactly as `_heredoc_aware_
    block_extent`'s own doc explains."""
    depth = 0
    end = start_idx
    limit = min(len(lines), start_idx + max_scan)
    for i in range(start_idx, limit):
        if heredoc_aware and lex[i].in_heredoc_body:
            end = i
            continue
        stop = False
        for tok in _IF_FI_ELSE_ELIF_TOKEN_RE.findall(_guard_text(lex[i])):
            if tok == "if":
                depth += 1
            elif tok == "fi":
                depth -= 1
            elif depth == 1:  # "else" or "elif" at the guard's own depth
                stop = True
                break
        if stop:
            # The else/elif line itself is not part of the taken branch —
            # exclude it (unless it shares the opener's own line, a rare
            # single-line `if X; then Y; else Z; fi` shape this line-level
            # walk cannot split further; the window then still includes at
            # least the opener line).
            end = i - 1 if i > start_idx else start_idx
            break
        end = i
        if depth <= 0:
            break
    return start_idx, end


def _guard_block_lines(lines: list[str], lex: list[_LineLex], start_idx: int) -> tuple[int, int]:
    """The inclusive `(start, end)` line-index range of ONLY the TAKEN
    (`then`) branch of the `if [...]; then ... fi` block whose OWN
    condition line is `lines[start_idx]` — see `_if_taken_branch_extent`
    for the full depth-walk/else-elif-stop rule this delegates to. Bounded
    at `_GUARD_BLOCK_MAX_SCAN` lines; not heredoc-aware (callers of this
    function inspect a guard's own condition/body, never a heredoc payload
    at this call site)."""
    return _if_taken_branch_extent(lines, lex, start_idx, _GUARD_BLOCK_MAX_SCAN, heredoc_aware=False)


def check_fake_knob_inertness(path: Path) -> list[str]:
    """(A) — see module doc. Operates on the file's CODE lines only (a
    knob only ever named in a module-doc comment above its real guard is
    never mistaken for an unguarded use), and every remaining token/regex
    test below runs against `_trigger_text`/`_guard_text` (see module doc,
    "Comment handling"), never the raw line. `guard_idx` also excludes any
    line `_lex_stream` marked `in_heredoc_body`, symmetrically with
    `_dry_run_true_block_intervals`'s own opener skip: a heredoc payload
    that happens to contain the literal text of a real refusal guard
    (`<PREFIX>_DRY_RUN`, `!=`, `"1"`) is opaque data, not this script's own
    top-level guard code, and must never be credited as one."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lex = _lex_stream(lines)
    trigger = [_trigger_text(lines[i], lex[i]) for i in range(len(lines))]
    guard = [_guard_text(lex[i]) for i in range(len(lines))]
    fake_vars = sorted(set(FAKE_VAR_RE.findall(text)))
    findings: list[str] = []
    for var in fake_vars:
        code_use_idx = [i for i in range(len(lines)) if var in trigger[i]]
        if not code_use_idx:
            continue  # only ever named in comments/docs — nothing live to guard
        guard_idx = [
            i
            for i in code_use_idx
            if not lex[i].in_heredoc_body
            and DRY_RUN_VAR_RE.search(guard[i])
            and "!=" in guard[i]
            and '"1"' in guard[i]
        ]
        if not guard_idx:
            findings.append(
                f"{path}: `{var}` is referenced in code but no refusal guard (a line combining "
                f"`{var}`, a *DRY_RUN* variable, and `!= \"1\"`) was found — inert-unless-dry-run "
                "is not provable"
            )
            continue
        first_guard = min(guard_idx)
        block_start, block_end = _guard_block_lines(lines, lex, first_guard)
        guard_window = "\n".join(guard[i] for i in range(block_start, block_end + 1))
        if "exit" not in guard_window:
            findings.append(
                f"{path}:{first_guard + 1}: `{var}`'s guard line does not `exit` — a guard that "
                "does not refuse is not a refusal"
            )
        earlier = [i for i in code_use_idx if i < first_guard]
        if earlier:
            findings.append(
                f"{path}: `{var}` used at line(s) {[i + 1 for i in earlier]} BEFORE its refusal "
                f"guard at line {first_guard + 1} — a use preceding the guard cannot be covered by it"
            )
    return findings


def check_producer_parity(path: Path) -> list[str]:
    """(B) — see module doc. Only applies to scripts that actually name a
    jammi-bench BINARY PATH (`.../jammi-bench`, never the source-tree
    `crates/jammi-bench/...`) in CODE.

    Scope inclusion (does this script name a binary path at all) is
    computed off `_trigger_text` — the widest reading, so this scanner
    never fails to notice a script that genuinely invokes the binary.
    Cross-check satisfaction (`provenance`/`build_sha` present) is computed
    off `_guard_text` — the narrowest reading, so a comment (whole-line or
    trailing) can never forge or dodge this gate's verdict without a
    single real line of code backing it."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lex = _lex_stream(lines)
    trigger_text = "\n".join(_trigger_text(lines[i], lex[i]) for i in range(len(lines)))
    if not BIN_ASSIGN_RE.search(trigger_text):
        return []
    guard_text = "\n".join(_guard_text(lex[i]) for i in range(len(lines)))
    missing = [tok for tok in ("provenance", "build_sha") if tok not in guard_text]
    if missing:
        return [
            f"{path}: names a jammi-bench binary path but is missing {missing} in CODE (a "
            "comment mentioning the token does not count) — every producer that runs a "
            "jammi-bench binary must cross-check `$BIN provenance`'s build_sha before writing "
            "a GREEN leg (unification contract C5.1)"
        ]
    return []


def _governing_toggle(var: str) -> str:
    """`<PREFIX>_DRY_RUN_<SUFFIX>` -> `<PREFIX>_DRY_RUN` — the toggle whose
    `= "1"` truth gates every legitimate read site of `var` (see (C))."""
    idx = var.index("_DRY_RUN_")
    return var[:idx] + "_DRY_RUN"


def _is_self_default_assignment(line: str, var: str) -> bool:
    """`VAR="${VAR:-default}"` (or an empty default `${VAR:-}`) — the
    ordinary bash env-var-with-default idiom. Captures the ambient value (or
    a fallback) into a same-named local with no consequence of its own, so
    it is never counted as a "read site" for (C)'s containment/refusal
    check — see this module's own doc for why every real knob in this file
    declares one of these up front."""
    pattern = re.compile(r'^\s*' + re.escape(var) + r'\s*=\s*"\$\{' + re.escape(var) + r':-[^}]*\}"\s*$')
    return bool(pattern.match(line))


def _heredoc_aware_block_extent(lines: list[str], lex: list[_LineLex], start_idx: int) -> tuple[int, int]:
    """The `(start, end)` inclusive line-index range of ONLY the TAKEN
    (`then`) branch of the bash `if ... fi` block whose OPENING line is
    `lines[start_idx]` — see `_if_taken_branch_extent` for the full
    depth-walk/else-elif-stop rule this delegates to — treating every
    heredoc body between a `<<[-]TERM` opener and its bare-`TERM`
    terminator line as OPAQUE DATA — see this module's own doc for why a
    depth counter that does not skip heredoc bodies runs past its real
    closing `fi` on both `profile_421_legs.sh` and `lora_bias_ab.sh`
    (their DRY_RUN stub heredocs embed a `python3 -c '...'` payload whose
    own `if`/`else` statements never close with a bash `fi`).

    Heredoc bodies are identified purely by reading `lex[i].in_heredoc_
    body` — the span `_lex_stream` already computed once for the whole
    file — never by re-detecting the opener with a fresh regex scan here:
    a second, independent heredoc-opener detector living in this function
    (sharing `_lex_stream`'s own heredoc regex constant) is exactly the
    kind of duplication that hides a bug (the false heredoc opened by `_HEREDOC_OPEN_RE`
    matching inside an already-closed quote, or inside a `<<<` here-string,
    on a raw-line re-scan that ignored the lexer's own position). Note that
    `start_idx` itself must never be a heredoc-body line — the caller
    (`_dry_run_true_block_intervals`) is responsible for skipping those when
    it looks for an opener in the first place, so this walker is never
    asked to open a block from inside a heredoc payload's own text.

    `if`/`fi`/`else`/`elif` tokens are recognized off `_guard_text(lex[i])`,
    never the raw line — a comment's own bare "if" (this codebase's own
    prose uses the word constantly) cannot perturb the depth walk, and an
    UNDECIDABLE line contributes nothing either.

    A DIFFERENT residual gap, disclosed rather than papered over with a
    refusal this scanner cannot actually back up: every regex in this
    module (`FAKE_VAR_RE`, `DRY_RUN_VAR_RE`, `DRY_RUN_KNOB_RE`) matches a
    LITERAL identifier appearing in the script's own text. A knob name
    built at RUNTIME -- bash indirect expansion (`${!name}`), `eval
    "$name=..."`, or a name assembled by string concatenation
    (`"${PREFIX}_DRY_RUN_${SUFFIX}"`) -- never appears as that literal
    substring anywhere in the file, so neither this containment/refusal
    walker nor (A)'s inertness check can see it at all: not a false
    negative on a knob it inspected and misjudged, but a knob it never
    knew existed. Not hit by any tracked producer today (every real knob
    in this file is a bash-conventional literal `NAME="${NAME:-default}"`
    declaration, never an indirect/constructed one) -- this scanner
    staying grep-shaped for the failure class it DOES catch is the same
    tradeoff its own module doc already states for (A)/(B) ("mechanical...
    not a semantic understanding"), not a gap this file should try to
    close by refusing the pattern outright (a producer author who
    genuinely needs indirect expansion for an unrelated reason would then
    be blocked by a check that cannot actually evaluate whether THEIR
    specific use is safe).
    """
    return _if_taken_branch_extent(lines, lex, start_idx, _BLOCK_MAX_SCAN, heredoc_aware=True)


def _dry_run_true_block_intervals(lines: list[str], lex: list[_LineLex], governing: str) -> list[tuple[int, int]]:
    """Every `if [ "$<governing>" = "1" ]`-guarded region in `lines` (the
    guard may be one ANDed clause of a larger compound condition — e.g.
    `profile_421_legs.sh`'s `if [ "$leg_status" = "ok" ] && [
    "$PROFILE_421_LEGS_DRY_RUN" = "1" ] && [ -n "..." ]; then` — the
    equality test appearing ANYWHERE on the `if` line is what matters, not
    that it is the line's only clause). Matched against `_guard_text`, so a
    trailing comment that merely LOOKS like this guard shape (or an
    UNDECIDABLE line) can never open a phantom "covered" interval that
    hides a genuinely unguarded read site elsewhere in the file. A
    heredoc-body line (`lex[i].in_heredoc_body`) is skipped BEFORE the
    opener regex even runs: a stub script's own heredoc payload can
    legitimately contain the literal text of a real bash `if [ "$X" = "1"
    ]; then` (a producer that writes ITS OWN dry-run guard into a generated
    stub script), and that payload text is opaque data from this file's own
    top-level perspective, not a real, top-level guard opening a real
    interval — `_heredoc_aware_block_extent` already treats heredoc bodies
    as opaque for its OWN `if`/`fi` counting, so the opener search here must
    apply that same rule symmetrically, or a phantom interval opened from
    inside heredoc text can swallow a genuinely unguarded read that follows
    the heredoc's close."""
    guard_re = re.compile(r'\bif\b.*\[\s*"\$' + re.escape(governing) + r'"\s*=\s*"1"\s*\]')
    intervals: list[tuple[int, int]] = []
    for i, _line in enumerate(lines):
        if lex[i].in_heredoc_body:
            continue
        if guard_re.search(_guard_text(lex[i])):
            intervals.append(_heredoc_aware_block_extent(lines, lex, i))
    return intervals


def _line_in_any_interval(idx: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= idx <= end for start, end in intervals)


def check_dry_run_knob_containment(path: Path) -> list[str]:
    """(C) — see module doc. Widens (A)'s `*FAKE*`-only name filter to any
    `<PREFIX>_DRY_RUN_<SUFFIX>` test knob, admissible by EITHER containment
    (every read site inside an `if [ "$<PREFIX>_DRY_RUN" = "1" ]`-guarded
    region) or (A)'s own preflight-refusal shape, generalized off the
    `FAKE` name requirement. Read-site detection runs against
    `_trigger_text` (the widest reading — a real use this lexer could not
    fully parse is never dropped); guard-shape and self-default-exemption
    checks run against `_guard_text` (the narrowest reading — see module
    doc, "Comment handling"). `guard_idx`'s preflight-refusal detection also
    excludes any line `_lex_stream` marked `in_heredoc_body`, symmetrically
    with `_dry_run_true_block_intervals`'s own opener skip — a heredoc
    payload's own text can never forge this script's own top-level
    preflight-refusal guard."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lex = _lex_stream(lines)
    trigger = [_trigger_text(lines[i], lex[i]) for i in range(len(lines))]
    guard = [_guard_text(lex[i]) for i in range(len(lines))]
    knob_vars = sorted(set(DRY_RUN_KNOB_RE.findall(text)))
    findings: list[str] = []
    for var in knob_vars:
        governing = _governing_toggle(var)
        code_use_idx = [
            i
            for i in range(len(lines))
            if var in trigger[i] and not _is_self_default_assignment(guard[i], var)
        ]
        if not code_use_idx:
            continue  # only ever named in comments/docs, or only ever self-defaulted

        governing_re = re.compile(r"\b" + re.escape(governing) + r"\b")
        guard_idx = [
            i
            for i in code_use_idx
            if not lex[i].in_heredoc_body
            and governing_re.search(guard[i])
            and "!=" in guard[i]
            and '"1"' in guard[i]
        ]
        if guard_idx:
            first_guard = min(guard_idx)
            block_start, block_end = _guard_block_lines(lines, lex, first_guard)
            guard_window = "\n".join(guard[i] for i in range(block_start, block_end + 1))
            earlier = [i for i in code_use_idx if i < first_guard]
            if "exit" in guard_window and not earlier:
                continue  # admissible via preflight refusal (mode 2)

        intervals = _dry_run_true_block_intervals(lines, lex, governing)
        uncovered = [i for i in code_use_idx if not _line_in_any_interval(i, intervals)]
        if uncovered:
            findings.append(
                f"{path}: `{var}` is read at line(s) {[i + 1 for i in uncovered]} outside any "
                f'`if [ "${governing}" = "1" ]`-guarded region, and no preflight refusal (a line '
                f'combining `{var}`, `{governing}`, `!=`, and `"1"`, that `exit`s, before every '
                f"other use) covers it either — this DRY_RUN-only test knob is not provably inert "
                "in a real run."
            )
    return findings


# CI incident (run 33230050451, main, "Guard (arch validation freshness
# self-test)"), same class here: `shutil.rmtree` during a `tempfile.
# TemporaryDirectory`'s teardown can hit `OSError: [Errno 39] Directory not
# empty: '.git'` — a race between tempdir cleanup and a background `git
# maintenance`/`gc --auto` process the scratch repo `self_test` builds below
# can spawn. `-c gc.auto=0 -c gc.autoDetach=false -c maintenance.auto=false`
# kills the background writer AT THE SOURCE.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _scratch_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=cwd, check=True)


def _self_test_lexer() -> list[str]:
    """Direct, fixture-free assertions on `_lex_one_line`/`_lex_stream`
    themselves -- the shared primitive every arm's forged-GREEN self-test
    fixtures below exercise only indirectly through a full `check_*` call.
    Returns a list of failure descriptions (empty when everything
    passes)."""
    failures: list[str] = []

    def lex_one(text: str) -> list[_LineLex]:
        return _lex_stream(text.splitlines())

    def check(got, expected, label: str) -> None:
        if got != expected:
            failures.append(f"lexer self-check FAILED ({label}): got {got!r}, expected {expected!r}")

    # Trailing comment on an otherwise-real code line is stripped.
    check(lex_one('BIN="$X"  # provenance build_sha')[0], _LineLex('BIN="$X"  ', False), "trailing comment")
    # A whole-line comment strips to nothing but leading whitespace.
    check(lex_one("# just a comment")[0], _LineLex("", False), "whole-line comment")
    check(lex_one("  # indented comment")[0], _LineLex("  ", False), "indented whole-line comment")
    # A `#` inside a double-quoted string is NOT a comment marker.
    check(lex_one('echo "value#1"')[0], _LineLex('echo "value#1"', False), "# inside double quotes")
    # A `#` inside a single-quoted string is NOT a comment marker.
    check(lex_one("echo 'a#b'")[0], _LineLex("echo 'a#b'", False), "# inside single quotes")
    # `${#name}` (bash length expansion) is NOT a comment marker.
    check(lex_one("n=${#myvar}")[0], _LineLex("n=${#myvar}", False), "${#var} length expansion")
    # `$#` (positional-parameter count): `#` is preceded by `$`, never a
    # delimiter, so it is mid-word -- UNDECIDABLE, not silently "not a
    # comment, keep going": this lexer never guesses on a shape it has not
    # been specifically taught to recognize.
    check(lex_one("echo $# arguments")[0], _LineLex("echo $# arguments", True), "$# positional count")
    # A REAL comment following a closed quoted string is still stripped.
    check(lex_one('echo "ok" # trailing')[0], _LineLex('echo "ok" ', False), "comment after a closed quote")
    # A line with no `#` at all is returned unchanged.
    check(lex_one('echo "no hash here"')[0], _LineLex('echo "no hash here"', False), "no hash at all")
    # `$((10#$N))`: the base-literal `#` sits inside ARITH -- never a
    # comment, decidable, code unchanged (false-strip case,
    # `runpod_lib.sh:155`/`:173`'s real shape).
    check(
        lex_one("RP_SSH_WAIT_SECS=$((10#$RP_SSH_WAIT_SECS))")[0],
        _LineLex("RP_SSH_WAIT_SECS=$((10#$RP_SSH_WAIT_SECS))", False),
        "arithmetic base-literal # (runpod_lib.sh:155 shape)",
    )
    # `${var#pattern}`: the strip-operator `#` sits inside PARAM -- never a
    # comment, decidable, code unchanged (false-strip case,
    # `test_pod_substrate.sh:1170`'s real shape).
    check(
        lex_one('rsync ${rsync_flags_line#rsync } "$X"')[0],
        _LineLex('rsync ${rsync_flags_line#rsync } "$X"', False),
        "parameter-expansion strip-operator # (test_pod_substrate.sh:1170 shape)",
    )
    # A single-quoted region genuinely spanning multiple physical lines: a
    # line that CLOSES a quote opened several lines earlier, with real
    # code and a real trailing comment after the close, must have that
    # comment correctly stripped -- the fail-open case
    # (`pod_push_stamp.sh:353`'s real shape: `python3 -c '...'` piped
    # across several lines, the closing line reading `' "$stamp"
    # 2>/dev/null)" # trailing comment`).
    multi = lex_one(
        "x=\"$(python3 -c '\n"
        "print(1)\n"
        "' \"$y\" 2>/dev/null)\" # trailing comment\n"
    )
    check(multi[0].undecidable, True, "multiline SQ opener is undecidable (ends with open SQ)")
    check(multi[1].undecidable, True, "multiline SQ body line is undecidable")
    check(multi[2], _LineLex('\' "$y" 2>/dev/null)" ', False), "multiline SQ closing line strips its own comment")
    # A double-quoted `"$(...)"` command substitution spanning multiple
    # physical lines is ORDINARY, fully-tracked real code -- never flagged
    # UNDECIDABLE just for spanning lines (the false-regression this
    # design specifically avoids: `top == "SQ"` is the only "ends open"
    # trigger, never a bare non-empty stack).
    multiline_cmd = lex_one(
        'x="$(FOO="a" \\\n'
        '  BAR="b" \\\n'
        '  cmd)" # trailing comment\n'
    )
    check(multiline_cmd[0].undecidable, False, "multiline $(...) opener is decidable")
    check(multiline_cmd[2], _LineLex('  cmd)" ', False), "multiline $(...) closing line strips its own comment")
    # A heredoc body's own prose apostrophes must never leak cross-line
    # quote state into the real code that follows the heredoc's close.
    heredoc = lex_one(
        "cat > x <<'EOF'\n"
        "# gpu-dev.sh's own doesn't-close-a-real-quote comment\n"
        "EOF\n"
        'BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"  # SWEEP_DRY_RUN != "1"; exit 2\n'
    )
    check(heredoc[3], _LineLex('BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"  ', False), "code after a heredoc lexes cleanly")
    return failures


def _self_test_corpus_lexer_agreement(repo_root: Path) -> list[str]:
    """Runs `_lex_stream` and `_independent_lex_stream` (see their own
    docs) over EVERY tracked `.sh` line under `ci/scripts/` and asserts
    zero disagreements — a differential proof that the primary lexer's
    comment-boundary decision is not merely tuned to this file's own
    hand-picked fixtures, and a regression detector for either
    implementation going forward. The five real lines a
    differential scan found disagreeing with the OLD (pre-this-file)
    hand-rolled `_strip_comment` are pinned as explicit, named
    expectations below, not just implicitly covered by the general scan."""
    failures: list[str] = []
    paths = _tracked_sh_under(repo_root, "ci/scripts/")
    if not paths:
        return ["corpus lexer agreement self-test FAILED: zero tracked .sh files under ci/scripts/ -- vacuous scan"]
    disagreements = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        primary = _lex_stream(lines)
        independent = _independent_lex_stream(lines)
        for i, (p, ind) in enumerate(zip(primary, independent, strict=True)):
            if p.undecidable != ind.undecidable or (not p.undecidable and not ind.undecidable and p.code != ind.code):
                disagreements += 1
                if disagreements <= 20:
                    failures.append(
                        f"corpus lexer disagreement at {path}:{i + 1}: "
                        f"primary={p!r} independent={ind!r} line={lines[i]!r}"
                    )
    if disagreements > 20:
        failures.append(f"... and {disagreements - 20} more disagreements (truncated)")

    # The five real lines pinned by name (differential corpus scan):
    # two live fail-opens (a closing quote misread as opening one, so a
    # real trailing comment was never stripped) and three false-strips (a
    # non-comment `#` misread as a comment marker, truncating real code).
    named_cases = [
        ("ci/scripts/pod_push_stamp.sh", 353, False, '\' "$stamp" 2>/dev/null)" '),
        (
            "ci/scripts/test_pod_substrate.sh",
            1118,
            False,
            '  bsha_reverted_value="$(bash -c "${bsha_reverted_text}; printf \'%s\' \\"\\${JAMMI_BUILD_SHA:-}\\"" '
            '2>/dev/null)" ',
        ),
        ("ci/scripts/runpod_lib.sh", 155, False, "RP_SSH_WAIT_SECS=$((10#$RP_SSH_WAIT_SECS))"),
        ("ci/scripts/runpod_lib.sh", 173, False, "RP_INACTIVITY=$((10#$RP_INACTIVITY))"),
        (
            "ci/scripts/test_pod_substrate.sh",
            1170,
            False,
            '  rsync ${rsync_flags_line#rsync } "${EXCLUDE_ARGS[@]}" "$SRC_REPO/" "$TREE_DEST/" > '
            '"$SANDBOX/i_push.out" 2>&1',
        ),
    ]
    for rel, lineno, expect_undecidable, expect_code in named_cases:
        path = repo_root / rel
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        got = _lex_stream(lines)[lineno - 1]
        if got.undecidable != expect_undecidable or got.code != expect_code:
            failures.append(
                f"named case FAILED: {rel}:{lineno} expected _LineLex(code={expect_code!r}, "
                f"undecidable={expect_undecidable}), got {got!r}"
            )
    return failures


def run_gate(perf_dir: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    for path in _tracked_sh_under(repo_root, "ci/scripts/"):
        findings += check_fake_knob_inertness(path)
        findings += check_dry_run_knob_containment(path)
    for path in _tracked_sh_under(repo_root, "ci/scripts/perf/"):
        findings += check_producer_parity(path)
    return findings


def self_test() -> int:
    failures: list[str] = []
    failures += _self_test_lexer()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        repo = Path(tmp)
        _scratch_git(["init", "-q"], repo)
        _scratch_git(["config", "user.email", "test@example.com"], repo)
        _scratch_git(["config", "user.name", "Test"], repo)
        perf = repo / "ci" / "scripts" / "perf"
        perf.mkdir(parents=True)

        def commit_and_check(rel: str, text: str, check_fn, expect_hit: str | None):
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            _scratch_git(["add", "-A"], repo)
            _scratch_git(["commit", "-q", "-m", rel], repo)
            got = check_fn(p)
            if expect_hit is None:
                if got:
                    failures.append(f"self-test FAILED: {rel} expected clean, got {got}")
            elif not any(expect_hit in g for g in got):
                failures.append(f"self-test FAILED: {rel} expected a finding containing {expect_hit!r}, got {got}")

        # (A) RED: a knob referenced with no guard at all.
        commit_and_check(
            "ci/scripts/perf/bad_no_guard.sh",
            '#!/usr/bin/env bash\nif [ -n "${SWEEP_FAKE_BIN_SHA:-}" ]; then BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"; fi\n',
            check_fake_knob_inertness,
            "no refusal guard",
        )

        # (A) RED: a knob used BEFORE its own guard.
        commit_and_check(
            "ci/scripts/perf/bad_use_before_guard.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo refuse; exit 2; fi\n'
            ),
            check_fake_knob_inertness,
            "BEFORE its refusal",
        )

        # (A) RED: a guard line that never exits (not a real refusal).
        commit_and_check(
            "ci/scripts/perf/bad_guard_no_exit.sh",
            '#!/usr/bin/env bash\nif [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo warn; fi\n',
            check_fake_knob_inertness,
            "does not `exit`",
        )

        # (A) RED, still: a THREE physical-line guard block that genuinely
        # never `exit`s (warns and falls through) — proves the widened
        # if/fi-depth window is not a rubber stamp: it reads the WHOLE
        # enclosing block, not "any 3+ lines", and still fires when that
        # block really has no `exit` anywhere in it.
        commit_and_check(
            "ci/scripts/perf/bad_guard_no_exit_multiline.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::refusing" >&2\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            "does not `exit`",
        )

        # (A) GREEN control: `stacked_sweep.sh`'s OWN real guard shape — a
        # THREE physical-line `if`/`echo`/`exit`/`fi` block, `exit` on the
        # guard's third line — proves the block-extent walk reads the
        # WHOLE enclosing block rather than a fixed short window.
        commit_and_check(
            "ci/scripts/perf/good_guard_three_line.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::SWEEP_FAKE_BIN_SHA is set but SWEEP_DRY_RUN != 1" >&2\n'
                '  exit 2\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            None,
        )

        # (A) GREEN control: the real stacked_sweep.sh shape — guard first,
        # dry-run-only use after.
        commit_and_check(
            "ci/scripts/perf/good_guard.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ] && [ "$SWEEP_DRY_RUN" != "1" ]; then echo refuse >&2; exit 2; fi\n'
                'if [ "$SWEEP_DRY_RUN" = "1" ]; then\n'
                '  if [ -n "${SWEEP_FAKE_BIN_SHA:-}" ]; then BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"; fi\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            None,
        )

        # (A) GREEN control: knob named ONLY in a comment (e.g. a module doc)
        # — never a live code use, so nothing to guard.
        commit_and_check(
            "ci/scripts/perf/good_comment_only.sh",
            '#!/usr/bin/env bash\n# SWEEP_FAKE_BIN_SHA is documented elsewhere; not read by this script.\necho hi\n',
            check_fake_knob_inertness,
            None,
        )

        # (A) RED: a knob use whose ONLY guard-shaped text (`SWEEP_DRY_RUN`,
        # `!=`, `"1"`, and `exit`) sits inside a TRAILING comment on the
        # very same line as the use. A lexer that read the raw line here
        # would see the whole line as both its own "use" and its own
        # "guard" — a comment alone forging a full GREEN. With the comment
        # correctly excluded from the guard text, the use survives but the
        # (fake) guard vanishes, correctly reporting "no refusal guard".
        commit_and_check(
            "ci/scripts/perf/bad_fake_knob_guard_forged_by_trailing_comment.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"  # SWEEP_DRY_RUN != "1"; exit 2 if triggered\n'
            ),
            check_fake_knob_inertness,
            "no refusal guard",
        )

        # (A) RED, robustness control: the SAME trailing-comment forgery,
        # with an ordinary, fully-balanced, unrelated `echo "ok"` statement
        # inserted immediately before the comment on the same line — a
        # hand-rolled comment lexer with incomplete quote/escape handling
        # can be perturbed by nearby, unrelated real code; a complete
        # state machine must reach the identical verdict regardless.
        commit_and_check(
            "ci/scripts/perf/bad_fake_knob_guard_forged_by_trailing_comment_with_echo_ok.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN_PROV_SHA="$SWEEP_FAKE_BIN_SHA"; echo "ok"  # SWEEP_DRY_RUN != "1"; exit 2 if triggered\n'
            ),
            check_fake_knob_inertness,
            "no refusal guard",
        )

        # (B) RED: names a jammi-bench binary path, invokes it, but never
        # cross-checks provenance.
        commit_and_check(
            "ci/scripts/perf/bad_no_provenance.sh",
            '#!/usr/bin/env bash\nBIN="$TARGET_DIR/release/jammi-bench"\n"$BIN" finetune-step --batch 1\n',
            check_producer_parity,
            "missing",
        )

        # (B) GREEN control: names the binary AND carries both tokens.
        commit_and_check(
            "ci/scripts/perf/good_provenance.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN="$TARGET_DIR/release/jammi-bench"\n'
                'J="$("$BIN" provenance)"\n'
                'S="$(python3 -c \'import json,sys;print(json.load(sys.stdin)["build_sha"])\' <<<"$J")"\n'
            ),
            check_producer_parity,
            None,
        )

        # (B) GREEN control: a script that never names a jammi-bench BINARY
        # path (only the crate source tree) is out of scope entirely.
        commit_and_check(
            "ci/scripts/perf/good_out_of_scope.sh",
            '#!/usr/bin/env bash\n# see crates/jammi-bench/reference/torch_finetune_step.py\necho hi\n',
            check_producer_parity,
            None,
        )

        # (B) GREEN control: the ONLY mention of a jammi-bench BINARY path
        # is inside a COMMENT — must never pull the script into scope (a
        # comment naming another producer's binary path is not this script
        # invoking one), so this is out-of-scope, not a RED for missing
        # provenance/build_sha.
        commit_and_check(
            "ci/scripts/perf/good_comment_only_bin_path.sh",
            (
                '#!/usr/bin/env bash\n'
                '# for comparison, see: $TARGET_DIR/release/jammi-bench\n'
                'echo hi\n'
            ),
            check_producer_parity,
            None,
        )

        # (B) RED: names a REAL binary path in code and invokes it, but
        # `provenance`/`build_sha` appear ONLY inside a comment — a comment
        # must never satisfy the cross-check on a script with no real code
        # backing it.
        commit_and_check(
            "ci/scripts/perf/bad_comment_only_provenance.sh",
            (
                '#!/usr/bin/env bash\n'
                '# TODO: cross-check "$BIN" provenance build_sha before shipping\n'
                'BIN="$TARGET_DIR/release/jammi-bench"\n'
                '"$BIN" finetune-step --batch 1\n'
            ),
            check_producer_parity,
            "missing",
        )

        # (B) RED: the binary-path line itself carries the ONLY mention of
        # `provenance`/`build_sha`, as a TRAILING comment on that very line
        # (never a whole comment line). A lexer that read the raw line
        # here would let the trailing comment's tokens satisfy the
        # cross-check with no real code backing it — a comment alone
        # forging GREEN.
        commit_and_check(
            "ci/scripts/perf/bad_provenance_satisfied_only_by_trailing_comment.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN="$TARGET_DIR/release/jammi-bench"  # would check "$BIN" provenance build_sha but skipped\n'
                '"$BIN" finetune-step --batch 1\n'
            ),
            check_producer_parity,
            "missing",
        )

        # (B) RED, robustness control: the SAME trailing-comment forgery,
        # with an unrelated, fully-balanced `echo "ok"` statement inserted
        # on the binary-path line before the comment.
        commit_and_check(
            "ci/scripts/perf/bad_provenance_satisfied_only_by_trailing_comment_with_echo_ok.sh",
            (
                '#!/usr/bin/env bash\n'
                'BIN="$TARGET_DIR/release/jammi-bench"; echo "ok"  '
                '# would check "$BIN" provenance build_sha but skipped\n'
                '"$BIN" finetune-step --batch 1\n'
            ),
            check_producer_parity,
            "missing",
        )

        # (C) RED: a `_DRY_RUN_` knob read completely unguarded — no
        # containment, no preflight refusal.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_unguarded.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'echo "${FOO_DRY_RUN_BAR:-}"\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) RED, and the actual regression this class's heredoc-aware
        # extent walker exists to catch: a knob read on the line
        # IMMEDIATELY AFTER a DRY_RUN=1 block's real closing `fi`, where
        # that block's own heredoc body embeds a `python3 -c` payload whose
        # `if`/`else` never close with a bash `fi` (the exact shape both
        # `profile_421_legs.sh` and `lora_bias_ab.sh` use). A depth counter
        # that does NOT skip heredoc bodies never reaches depth<=0 at the
        # real `fi` (the two dangling python `if`s leave it short), so it
        # keeps scanning to the end of the file and wrongly reports the
        # LAST line as still "inside" the block — silently passing an
        # uncontained, unrefused knob. `FOO_DRY_RUN_BAZ` here sits outside
        # the block on a real (non-heredoc) line, so a correct,
        # heredoc-aware walker must still flag it.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_after_heredoc_with_unmatched_if.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ "$FOO_DRY_RUN" = "1" ]; then\n'
                '  cat > /tmp/stub.sh <<STUBEOF\n'
                'python3 -c "\n'
                'if True:\n'
                '    print(1)\n'
                'else:\n'
                '    print(2)\n'
                '"\n'
                'STUBEOF\n'
                'fi\n'
                'echo "${FOO_DRY_RUN_BAZ:-}"\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) RED: the knob's use line carries NO real guard code of its
        # own, but a TRAILING comment on that same line spells out the full
        # preflight shape (`FOO_DRY_RUN`, `!=`, `"1"`, `exit`). A lexer
        # that read the raw line here would admit the knob via preflight
        # refusal with zero real guarding code — a forged GREEN.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_guard_forged_by_trailing_comment.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'echo "${FOO_DRY_RUN_BAR:-}"  # FOO_DRY_RUN != "1"; exit 2 if triggered\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) RED, robustness control: the SAME trailing-comment forgery,
        # with an unrelated, fully-balanced `echo "ok"` statement inserted
        # on the use line before the comment.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_guard_forged_by_trailing_comment_with_echo_ok.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'echo "${FOO_DRY_RUN_BAR:-}"; echo "ok"  # FOO_DRY_RUN != "1"; exit 2 if triggered\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) GREEN control: containment — the knob is only ever read
        # inside a heredoc body written while `FOO_DRY_RUN=1`.
        commit_and_check(
            "ci/scripts/perf/good_dry_run_knob_heredoc_containment.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ "$FOO_DRY_RUN" = "1" ]; then\n'
                '  cat > /tmp/stub.sh <<STUBEOF\n'
                'echo "${FOO_DRY_RUN_BAR:-}"\n'
                'STUBEOF\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            None,
        )

        # (C) GREEN control: preflight refusal — `profile_421_legs.sh`'s
        # own `PROFILE_421_LEGS_DRY_RUN_TRUNCATE_CORPUS_VAR` shape,
        # generalized off the `FAKE` name requirement.
        commit_and_check(
            "ci/scripts/perf/good_dry_run_knob_preflight_refusal.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ -n "${FOO_DRY_RUN_TRUNCATE_VAR:-}" ] && [ "$FOO_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::refusing" >&2\n'
                '  exit 2\n'
                'fi\n'
                'if [ "$FOO_DRY_RUN" = "1" ] && [ -n "${FOO_DRY_RUN_TRUNCATE_VAR:-}" ]; then\n'
                '  echo "$FOO_DRY_RUN_TRUNCATE_VAR"\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            None,
        )

        # (C) GREEN control: a knob only ever named in a comment, or only
        # ever self-defaulted (`VAR="${VAR:-...}"`) — neither is a live
        # read site.
        commit_and_check(
            "ci/scripts/perf/good_dry_run_knob_comment_and_self_default_only.sh",
            (
                '#!/usr/bin/env bash\n'
                '# FOO_DRY_RUN_BAR is documented elsewhere; not read by this script.\n'
                'FOO_DRY_RUN_BAR="${FOO_DRY_RUN_BAR:-}"\n'
                'echo hi\n'
            ),
            check_dry_run_knob_containment,
            None,
        )

        # (C) RED — F1, else-arm read: `FOO_DRY_RUN_EVIL`'s only read site
        # sits in the `else` arm of `if [ "$FOO_DRY_RUN" = "1" ]`, which
        # never runs while DRY_RUN=1 (the only time this containment guard
        # is satisfied at all) — a knob read exclusively in the sibling
        # dead arm must never be credited as contained.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_read_in_else_arm.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ "$FOO_DRY_RUN" = "1" ]; then\n'
                '  echo dry\n'
                'else\n'
                '  rm -rf "$FOO_DRY_RUN_EVIL"\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) RED — F1, elif-arm read: same class, the dead arm is an
        # `elif` rather than a bare `else`.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_read_in_elif_arm.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ "$FOO_DRY_RUN" = "1" ]; then\n'
                '  echo dry\n'
                'elif [ "$OTHER_TOGGLE" = "1" ]; then\n'
                '  rm -rf "$FOO_DRY_RUN_EVIL2"\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) GREEN control — F1 positive control: the same else-arm shape,
        # but the knob is read ONLY inside the taken (`then`) branch — an
        # else arm being present at all must not perturb a genuinely
        # contained read.
        commit_and_check(
            "ci/scripts/perf/good_dry_run_knob_read_in_then_arm_with_else_present.sh",
            (
                '#!/usr/bin/env bash\n'
                'FOO_DRY_RUN="${FOO_DRY_RUN:-0}"\n'
                'if [ "$FOO_DRY_RUN" = "1" ]; then\n'
                '  echo "${FOO_DRY_RUN_BAR:-}"\n'
                'else\n'
                '  echo no-op\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            None,
        )

        # (A) RED — F1, dead exit in a dead else arm: `X_FAKE_THING`'s
        # refusal guard's `then` arm only `echo`s; the ONLY `exit` in the
        # block lives inside a NESTED `if false; then exit 9; fi` sitting in
        # the guard's OWN `else` arm — an arm that never runs when the
        # refusal condition (`X_DRY_RUN != "1"`) is true. That dead `exit`
        # must never be credited as a real refusal.
        commit_and_check(
            "ci/scripts/perf/bad_fake_knob_exit_only_in_dead_else_arm.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${X_FAKE_THING:-}" ] && [ "$X_DRY_RUN" != "1" ]; then\n'
                '  echo warn\n'
                'else\n'
                '  echo not-exiting\n'
                '  if false; then\n'
                '    exit 9\n'
                '  fi\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            "does not `exit`",
        )

        # (A) GREEN control — F1 positive control: the same refusal-guard
        # shape, `exit` correctly placed in the TAKEN (`then`) arm, with an
        # unrelated `else` arm present — the else arm's mere presence must
        # not stop the block-extent walk from finding the real, taken-arm
        # `exit`.
        commit_and_check(
            "ci/scripts/perf/good_fake_knob_exit_in_taken_arm_with_else_present.sh",
            (
                '#!/usr/bin/env bash\n'
                'if [ -n "${X_FAKE_THING:-}" ] && [ "$X_DRY_RUN" != "1" ]; then\n'
                '  echo "::error::refusing" >&2\n'
                '  exit 2\n'
                'else\n'
                '  echo "not taken: a real run always refuses above"\n'
                'fi\n'
            ),
            check_fake_knob_inertness,
            None,
        )

        # (C) RED — F2, phantom opener inside a heredoc body: the ONLY text
        # matching `if [ "$WUP_DRY_RUN" = "1" ]` sits inside a heredoc
        # payload (a stub script this producer writes out, whose own
        # generated content happens to contain that literal guard shape).
        # `WUP_DRY_RUN_EVIL` is read at TOP LEVEL, after the heredoc closes,
        # with no real guard anywhere in the file — the opener match must
        # never fire on heredoc-body text, or this read is wrongly credited
        # as contained by a phantom interval opened from inside opaque data.
        commit_and_check(
            "ci/scripts/perf/bad_dry_run_knob_phantom_opener_in_heredoc.sh",
            (
                '#!/usr/bin/env bash\n'
                'WUP_DRY_RUN="${WUP_DRY_RUN:-0}"\n'
                'cat > /tmp/stub.sh <<'"'"'EOS'"'"'\n'
                'echo start\n'
                'if [ "$WUP_DRY_RUN" = "1" ]; then\n'
                '  echo fake\n'
                'fi\n'
                'EOS\n'
                'curl "$WUP_DRY_RUN_EVIL"\n'
            ),
            check_dry_run_knob_containment,
            "outside any",
        )

        # (C) GREEN control — F2 positive control: a REAL, top-level opener
        # appearing AFTER a heredoc closes must still open a real interval
        # (proves the fix is the heredoc-body skip specifically, not a
        # blanket "ignore every opener" regression).
        commit_and_check(
            "ci/scripts/perf/good_dry_run_knob_real_opener_after_heredoc.sh",
            (
                '#!/usr/bin/env bash\n'
                'WUP_DRY_RUN="${WUP_DRY_RUN:-0}"\n'
                'cat > /tmp/stub.sh <<'"'"'EOS'"'"'\n'
                'echo just data, no guard shape here\n'
                'EOS\n'
                'if [ "$WUP_DRY_RUN" = "1" ]; then\n'
                '  echo "${WUP_DRY_RUN_EVIL:-}"\n'
                'fi\n'
            ),
            check_dry_run_knob_containment,
            None,
        )

    # Non-vacuousness control (the actual bug this round fixes): a wrong
    # `REPO_ROOT` (previously `parents[2]`, resolving to `<repo>/ci` instead
    # of `<repo>`) makes `git ls-files ci/scripts/` run with the WRONG `cwd`
    # look for `<repo>/ci/ci/scripts/**`, which never exists — zero files,
    # zero findings, a PASS that enforced nothing. Assert BOTH tracked-file
    # scans this gate depends on see a REAL, nonzero count on the actual
    # repo tree, so a future regression of `REPO_ROOT` (or of the
    # `Cargo.toml` guard above being weakened/removed) cannot silently
    # revert to scanning nothing while still printing PASS.
    real_sh_under_scripts = _tracked_sh_under(REPO_ROOT, "ci/scripts/")
    real_sh_under_perf = _tracked_sh_under(REPO_ROOT, "ci/scripts/perf/")
    if not real_sh_under_scripts:
        failures.append(
            "self-test FAILED: `git ls-files ci/scripts/` under the real REPO_ROOT found ZERO "
            "`.sh` files -- the scan is vacuous (REPO_ROOT is almost certainly wrong)"
        )
    if not real_sh_under_perf:
        failures.append(
            "self-test FAILED: `git ls-files ci/scripts/perf/` under the real REPO_ROOT found "
            "ZERO `.sh` files -- the scan is vacuous (REPO_ROOT is almost certainly wrong)"
        )

    # End-to-end: the REAL tree, both checks, must be clean today.
    real_findings = run_gate(PERF_DIR, REPO_ROOT)
    if real_findings:
        failures.append(f"self-test FAILED: real tree is not clean: {real_findings}")

    # The full corpus lexer-agreement scan (see its own doc): every
    # tracked line, two independent implementations, zero disagreements —
    # including the five real lines the differential corpus scan named.
    failures += _self_test_corpus_lexer_agreement(REPO_ROOT)

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("check-producer-provenance-gates self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "check-producer-provenance-gates self-test: OK — (A) FAKE-knob inertness "
        "(no-guard / use-before-guard / guard-without-exit / trailing-comment-forgery, plain "
        "and with an added echo \"ok\", plus an `exit` living only in a dead `else` arm, all "
        "RED; a real guard, a comment-only mention, and a real taken-arm `exit` with an "
        "unrelated `else` arm present, all GREEN), (B) producer parity (a jammi-bench-binary "
        "producer missing provenance/build_sha, including via a trailing-comment forgery plain "
        "and with an added echo \"ok\", is RED; one carrying both, or one that never names a "
        "binary path at all, is GREEN), and (C) *_DRY_RUN_* knob containment (unguarded, a "
        "knob past a heredoc-embedded unmatched-if block's real `fi`, a trailing-comment "
        "forgery plain and with an added echo \"ok\", a read confined to a dead `else`/`elif` "
        "arm, and a knob covered only by a phantom guard opener whose text lives inside a "
        "heredoc payload, all RED; heredoc containment, preflight refusal, "
        "comment/self-default-only, a taken-arm read with an unrelated `else` arm present, and "
        "a real opener appearing after a heredoc closes, all GREEN) all bite on throwaway "
        "fixtures; the real tree is clean; and the lexer agrees with an independently-written "
        "second implementation on every tracked ci/scripts/ line, including the five real lines "
        "a differential audit scan found the pre-existing hand-rolled comment stripper "
        "misreading."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    findings = run_gate(PERF_DIR, REPO_ROOT)
    if findings:
        print("check-producer-provenance-gates: FAIL", file=sys.stderr)
        for msg in findings:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\ncheck-producer-provenance-gates: {len(findings)} finding(s).", file=sys.stderr)
        return 1
    print(
        "check-producer-provenance-gates: PASS — every FAKE-shaped and every "
        "*_DRY_RUN_*-shaped test knob under ci/scripts/ is inert unless its own governing "
        "toggle is 1 (contained or preflight-refused), and every ci/scripts/perf/*.sh naming a "
        "jammi-bench binary path cross-checks its provenance build_sha."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
