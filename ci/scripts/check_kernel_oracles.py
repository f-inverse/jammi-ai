#!/usr/bin/env python3
"""Kernel-acceptance-oracle standard gate (v1) — hermetic, static, no build, no GPU.

Mechanizes five of the eight `KO-1`..`KO-8` ids `docs/maintainer/cuda-kernel-guide.md`
§3's conformance list reports (`check_doc_parity.py` binds that list's set to this
file's `STABLE_IDS`, the same way it binds every other guide↔code enumeration —
see that module's `KernelOracleStandardIds` binding). The other three (`KO-1`,
`KO-6`, `KO-8`) are auditor-only by design — each requires running code or a
semantic judgment a static scan cannot make; the guide states, for each, the one
sentence explaining why.

## KO-7 — unrun-is-RED (TOTAL, enforced on every scanned file today)

A `#[test]` fn that silently `return;`s partway through (the `let Some(dev) =
cuda_device() else { return; };` skip idiom, and its brace-tail-expression sibling
`else { return }`) still reports green — the oracle never actually ran. That is
only acceptable when the skip is GATED: dominated, textually, by a call to a
REGISTERED require-gate helper — a fn, found anywhere in the scanned tree, whose
body reads a `JAMMI_REQUIRE_*` environment variable AND calls `panic!` (so a CI
lane that sets the var gets a hard failure instead of a silent skip). Helpers are
DISCOVERED by this shape, never hardcoded by name — `cuda_device` (`tests/
cuda_parity.rs`, `tests/flash_smoke.rs`) and `growth_oracle_cuda_device`
(`jammi-encoders/src/modernbert.rs`) are today's instances, found the same way a
fourth helper with a different name would be. "Dominated" here is a TEXTUAL
check (this is a lexical scanner, not a control-flow analyzer): within the SAME
`#[test]` fn body, a registered helper's call token must appear BEFORE the
return-skip token. Scope: RUNTIME skips only (an early `return` inside the test
body) — this rule says nothing about `#[ignore]`/`#[cfg(feature = ...)]`
compile-time gating, which `check_cuda_run_artifacts.py` rule (c) already
verifies per-artifact.

Scanned files: `crates/jammi-kernels/tests/*.rs`, `crates/jammi-encoders/src/*.rs`
(test modules live inline in the latter; scanning the whole file costs nothing —
non-test code simply carries no `#[test]` fn to check). KO-7 is TOTAL over this
file set today: every `return;` (or brace-tail `return}`) inside every `#[test]`
fn in every scanned file is checked, not just new/changed ones — no diff-base
input is needed in v1 for this reason. Fail-closed: any ungated skip is a
non-zero exit naming file:line.

## KO-2 / KO-5 — marker-scoped (apply only where a marker exists)

A test/const author who wants a marked oracle-cell writes ONE `//!` line inside
the file, near the relevant control fn:

    //! oracle-cell: op=<key> leg=<name> dtype=<bf16|f32> bounds=<IDENT,...> \
        control=<test_fn>|none:<reason> derived-on=<seed_list>|none \
        asserted-on=<seed_list>|none

  - `op` — an admission key from the SHIPPED op-domain (see below); must resolve
    or the marker itself is a finding (mirrors `check_gpu_parity_matrix.py`'s
    "a reviewed entry naming an unknown identifier is a non-zero exit").
  - `leg`/`dtype` — free identifiers naming which comparison leg this cell covers
    (e.g. `leg=fwd_parity dtype=bf16`); not independently validated in v1 beyond
    `dtype` being `bf16` or `f32`.
  - `bounds` — comma-separated identifiers (the tolerance/floor constants this
    cell's control fn is claimed to assert against).
  - `control` — the `#[test]` fn (or any fn) that performs the check, OR
    `none:<reason>` (underscore-joined, no spaces — this is a single
    whitespace-delimited marker line) when the cell is genuinely uncontrolled
    (e.g. a documentation-only cell).
  - `derived-on`/`asserted-on` — comma-separated seed lists (or the literal
    `none`) naming which seeds a bound was CALIBRATED from vs. VALIDATED
    against.

**KO-2 (bound coverage parity).** When `control` names a real fn, that fn's
body (resolved in the marker's OWN file only, brace-balanced) must contain
EVERY identifier listed in `bounds` used INSIDE an assertion or comparison — an
`assert!`-family macro argument list, or a statement carrying a comparison
operator (`<`, `<=`, `>`, `>=`) — either directly in the control fn, or (one
level of indirection) in a same-file fn the control calls. A bare declaration
or passing mention does not count. `control=none:<reason>` opts a cell out of
KO-2 entirely — it becomes DECLARED_UNCONTROLLED (see reconciliation below),
never silently passed.

**KO-5 (off-sample bounds).** When BOTH `derived-on` and `asserted-on` are real
seed lists (neither is the literal `none`), their intersection — after
normalizing every token to an integer (`42` == `042` == `0x2a`; a non-numeric
token compares as its raw string) — must be EMPTY: a bound calibrated on seed
42 and then "validated" by asserting against seed 42 again is circular, not
evidence (the pressure-test design rule in the guide: a bound's noise floor
must be measured independently of where it is enforced).

Both rules are MARKER-SCOPED: an unmarked oracle file is reported PENDING in the
reconciliation, never silently passed OR silently failed. **v1 ships with ZERO
markers added to any numerics-owned file** (`crates/jammi-kernels/src`,
`crates/jammi-encoders/src` production code, and their oracle test files) —
that domain is explicitly out of scope for this PR; every op in the
reconciliation below is honestly PENDING today, not COVERED. Closing a PENDING
op means adding a real `oracle-cell` marker AND removing the (implicit,
computed) PENDING entry in the same PR — same closure shape as
`check_gpu_parity_matrix.py`'s PENDING debt.

## The op domain (SHIPPED set)

There is no single `enum` of admission keys — `crate::admission::admit`'s `op`
argument is a `&'static str` literal threaded through call sites across
multiple crates. The one place every real op key surfaces as a literal is
`crate::admission::counters_for("<key>")` — every fused op's `DispatchCounters`
handle is obtained this way (`jammi-encoders/src/{modernbert,layer_norm}.rs`,
`jammi-lora/src/lora_linear.rs`, `jammi-ai/src/fine_tune/adamw.rs`,
`jammi-kernels/tests/cuda_parity.rs`). `load_shipped_ops()` below is a static
scan of every TRACKED `.rs` file for that call shape, EXCLUDING
`crates/jammi-kernels/src/admission.rs` itself (the definition site, whose own
test module calls `counters_for` with synthetic non-op names like
`"registry_test_op_a"` — including it would pollute the SHIPPED set with
fixture names that are not real admission keys).

## Fail-closed contract

  - Any ungated `#[test]`-body runtime skip (KO-7) is a non-zero exit naming
    file:line.
  - Any marker whose `control` fn does not cover every `bounds` identifier
    (KO-2), or whose `derived-on`/`asserted-on` seed sets intersect (KO-5), is
    a non-zero exit naming the marker's file:line.
  - Any marker naming an `op` outside the SHIPPED set, or a malformed marker
    line, is a non-zero exit (mirrors `check_gpu_parity_matrix.py`).
  - Any op claimed by more than one of COVERED / STRUCTURALLY_EXCLUDED /
    DECLARED_UNCONTROLLED is a non-zero exit (PENDING is computed as the
    complement, so it cannot conflict with the other three by construction).
  - A parse failure (missing scan root, uncomputable op domain) is a non-zero
    exit naming what could not be resolved.

## Round-2 (adversarial-audit) fixes

  - **Comment/string laundering.** Every scan (KO-7's skip/helper-call search,
    KO-2's bound-in-assertion search) runs over `_strip_comments_and_strings`
    output, never raw source — a helper name mentioned only in a `//` comment,
    or a `panic!`/`JAMMI_REQUIRE_*` token sitting only inside a string literal
    (an assertion MESSAGE, not real control flow), must never register a
    fn as a require-gate helper or launder an ungated skip.
  - **Helper registration by mechanism, not vibes.** A fn registers as a
    require-gate helper only when its stripped body contains BOTH a real env
    READ (`std::env::var(`/`env::var(`/`std::env::var_os(`/`env::var_os(`/
    `option_env!(`) whose string-literal argument starts with
    `JAMMI_REQUIRE_`, AND a reachable `panic!(`/`unreachable!(`/`.expect(` —
    a fn that merely returns `Option`/`bool` without ever being able to
    panic is not a require-GATE at all.
  - **KO-7 per-skip windowed domination.** An EARLIER helper call in a
    `#[test]` fn no longer launders every LATER skip in the same fn — each
    `return` is checked against only the window since the PREVIOUS skip (or
    fn start), so a gated CUDA-device check followed by an unrelated,
    ungated `if !FLASH_COMPILED { return; }` further down the same fn still
    reds. `RETURN_SKIP_RE` also matches `return Ok(...)`/`return Err(...)`.
  - **KO-2 is file-scoped and assertion-context-aware.** A marker's
    `control` fn is resolved ONLY within the marker's OWN file (never a
    same-named fn elsewhere in the tree); a bound must appear inside an
    `assert!`-family macro call or a comparison-operator-bearing statement
    IN that control fn, or (one level of indirection) in a same-file helper
    fn the control calls — a bare declaration or a passing mention no
    longer counts as coverage.
  - **A `control=none:<reason>` marker is DECLARED_UNCONTROLLED**, a FOURTH
    reconciliation category, printed with its reason — it can never move an
    op into COVERED. `PENDING = SHIPPED − COVERED − STRUCTURALLY_EXCLUDED −
    DECLARED_UNCONTROLLED`.
  - **KO-5 seed tokens are normalized to integers** (`42` == `042` ==
    `0x2a`) before the disjointness check — a non-numeric token still
    compares as its raw string.

Run: `python3 ci/scripts/check_kernel_oracles.py`
Suite: `python3 ci/scripts/test_check_kernel_oracles.py`
Hermetic: reads only files in the working tree (or in-memory/tempdir synthetic
data under the test suite); no network, no build, no GPU.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_TESTS_DIR = REPO_ROOT / "crates" / "jammi-kernels" / "tests"
ENCODERS_SRC_DIR = REPO_ROOT / "crates" / "jammi-encoders" / "src"
ADMISSION_DEFINITION_FILE = "crates/jammi-kernels/src/admission.rs"

# The `KO-1`..`KO-8` stable ids — bound to `docs/maintainer/cuda-kernel-guide.md`
# §3's `<!-- BEGIN KERNEL-ORACLE-STANDARD-IDS -->` marked list by
# `check_doc_parity.py`'s `KernelOracleStandardIds` binding (set-equality, code
# side = this Python list constant, not a Rust enum — see that binding's own
# comment for why a second leaf-level parser, not a second mechanism, was
# needed). Order is cosmetic; the binding is set-equality.
STABLE_IDS: tuple[str, ...] = (
    "KO-1",
    "KO-2",
    "KO-3",
    "KO-4",
    "KO-5",
    "KO-6",
    "KO-7",
    "KO-8",
)

# The subset this script mechanically enforces today. `KO-3` lives in
# `check_cuda_run_artifacts.py` (the `oracle_separation` artifact block) and
# `KO-4` lives in `check_doc_numbers_have_producers.py` (the floor-cites-a-
# producer trigger) — both documented here for completeness, neither
# re-implemented in this file (one definition per rule). Bound to the guide's
# per-id "mechanical"/"auditor-only" labels by `check_doc_parity.py`'s
# `KernelOracleMechanicalIds`/`KernelOracleAuditorOnlyIds` bindings (round-2
# audit item 7) — these constants are not dead weight; a stale label drift
# between this file and the guide's prose now reds the doc-parity gate.
MECHANICAL_HERE_IDS = ("KO-2", "KO-5", "KO-7")
MECHANICAL_ELSEWHERE_IDS = ("KO-3", "KO-4")
AUDITOR_ONLY_IDS = ("KO-1", "KO-6", "KO-8")


class OracleError(Exception):
    """Uncomputable input (parse failure, missing dir) — fails closed."""


# --------------------------------------------------------------------------- #
# op domain (SHIPPED) — static scan for `counters_for("<key>")` call sites.
# --------------------------------------------------------------------------- #
COUNTERS_FOR_RE = re.compile(r'counters_for\(\s*"([a-z0-9_]+)"\s*\)')


def _git_ls_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "--", "*.rs"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise OracleError(f"`git ls-files` failed: {proc.stderr.strip()}")
    return proc.stdout.splitlines()


def load_shipped_ops(repo_root: Path = REPO_ROOT) -> set[str]:
    """Every real admission key: every `counters_for("<key>")` call-site
    literal in a TRACKED `.rs` file, excluding `ADMISSION_DEFINITION_FILE`
    itself (whose own test module uses synthetic, non-op literal names).
    """
    ops: set[str] = set()
    for rel in _git_ls_files(repo_root):
        if rel == ADMISSION_DEFINITION_FILE:
            continue
        path = repo_root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in COUNTERS_FOR_RE.finditer(text):
            ops.add(m.group(1))
    if not ops:
        raise OracleError(
            "load_shipped_ops: zero `counters_for(\"...\")` call sites found in the "
            "tracked tree — did the admission-key registration convention change?"
        )
    return ops


def shipped_ops_from_sources(sources: dict[str, str]) -> set[str]:
    """Pure variant of `load_shipped_ops` over an in-memory {relpath: text}
    map — the self-test seam (`load_shipped_ops` itself is a thin `git
    ls-files` + filesystem wrapper around this).
    """
    ops: set[str] = set()
    for rel, text in sources.items():
        if rel == ADMISSION_DEFINITION_FILE:
            continue
        for m in COUNTERS_FOR_RE.finditer(text):
            ops.add(m.group(1))
    return ops


# --------------------------------------------------------------------------- #
# marker parsing — `//! oracle-cell: op=... leg=... dtype=... bounds=... \
#   control=... derived-on=... asserted-on=...`
# --------------------------------------------------------------------------- #
MARKER_PREFIX_RE = re.compile(r"^\s*//!\s*oracle-cell:\s*(?P<rest>\S.*)$")
REQUIRED_MARKER_KEYS = ("op", "leg", "dtype", "bounds", "control", "derived-on", "asserted-on")


@dataclass(frozen=True)
class OracleCellMarker:
    file: str
    line_no: int
    op: str
    leg: str
    dtype: str
    bounds: tuple[str, ...]
    control: str  # a fn name, or "none:<reason>"
    derived_on: tuple[str, ...] | None  # None means the literal `none`
    asserted_on: tuple[str, ...] | None

    @property
    def is_control_none(self) -> bool:
        return self.control.startswith("none:")


def _split_seed_list(value: str) -> tuple[str, ...] | None:
    if value == "none":
        return None
    return tuple(s for s in value.split(",") if s)


def parse_marker_line(line: str, file_label: str, line_no: int) -> OracleCellMarker:
    m = MARKER_PREFIX_RE.match(line)
    if m is None:
        raise OracleError(f"{file_label}:{line_no}: not an `oracle-cell:` marker line")
    rest = m.group("rest")
    fields: dict[str, str] = {}
    for tok in rest.split():
        if "=" not in tok:
            raise OracleError(
                f"{file_label}:{line_no}: oracle-cell marker token `{tok}` has no `=` "
                "— every field must be `key=value`"
            )
        k, v = tok.split("=", 1)
        fields[k] = v
    missing = [k for k in REQUIRED_MARKER_KEYS if k not in fields]
    if missing:
        raise OracleError(
            f"{file_label}:{line_no}: oracle-cell marker missing field(s) "
            f"{', '.join(missing)}: {line!r}"
        )
    if fields["dtype"] not in ("bf16", "f32"):
        raise OracleError(
            f"{file_label}:{line_no}: oracle-cell marker dtype must be `bf16` or `f32`, "
            f"got {fields['dtype']!r}"
        )
    return OracleCellMarker(
        file=file_label,
        line_no=line_no,
        op=fields["op"],
        leg=fields["leg"],
        dtype=fields["dtype"],
        bounds=tuple(b for b in fields["bounds"].split(",") if b),
        control=fields["control"],
        derived_on=_split_seed_list(fields["derived-on"]),
        asserted_on=_split_seed_list(fields["asserted-on"]),
    )


def parse_markers(text: str, file_label: str) -> list[OracleCellMarker]:
    markers: list[OracleCellMarker] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if "oracle-cell:" in line:
            markers.append(parse_marker_line(line, file_label, i))
    return markers


# --------------------------------------------------------------------------- #
# ONE canonical stripper (round-3 audit fix — "change the approach, not just
# the cases"): every downstream pass (fn discovery, fn-body brace counting,
# KO-7, KO-2) reads ONLY this function's output, applied ONCE per file.
# Handles, all by depth/state tracking rather than a naive quote-toggle:
#   - `//` line comments
#   - `/* ... */` block comments, NESTED (a `/* /* */ */` doesn't end at the
#     first `*/`)
#   - string literals `"..."`, byte strings `b"..."`, raw strings
#     `r"..."`/`r#"..."#`/..., raw byte strings `br"..."`/`br#"..."#`/...
#   - char literals `'x'`, `'\''`, `'"'`, `'\\'`, `'\u{...}'`, byte-char
#     literals `b'x'` — WITHOUT this, `'"'` (a char literal whose content
#     is a literal double-quote) is indistinguishable from the START of a
#     new string to a naive scanner, which silently desyncs quote-tracking
#     for the REST of the file (the live instance this round's audit found:
#     `feature_table.rs:45`'s `.trim_matches('"')` blinded everything after
#     it — the gate printed `0 runtime skip(s), 0 ungated` for a file that
#     actually has real, ungated skips). A bare `'a` LIFETIME is left alone
#     (not consumed as an unterminated char literal): only `'<escape-or-one-
#     char>'` shapes are recognized as char literals at all.
#   - a `JAMMI_REQUIRE_[A-Z0-9_]*` run INSIDE a string literal's content is
#     the ONE exception kept VISIBLE (not blanked) — this is DATA the
#     registration mechanism must read (the env-var name IS the string
#     argument), not prose; every other character of every string's content
#     is still blanked, so an assertion MESSAGE or doc string cannot launder
#     a `panic!`/helper-name mention. This is why ONE stripper, used
#     everywhere, is enough — no second "comments-only" variant is needed
#     anymore.
# Output is the SAME LENGTH as the input (every stripped character becomes a
# space, newlines kept as newlines) so every downstream offset/line-number
# computation is unaffected.
# --------------------------------------------------------------------------- #
_RAW_STR_OPEN_RE = re.compile(r'b?r(#*)"')
_JAMMI_REQUIRE_RUN_RE = re.compile(r"JAMMI_REQUIRE_[A-Z0-9_]*")


def _blank_string_content(content: str) -> str:
    """Blanks `content` (a string literal's interior) EXCEPT any
    `JAMMI_REQUIRE_[A-Z0-9_]*` run, which survives verbatim — see the
    module comment above. Length-preserving.
    """
    out: list[str] = []
    pos = 0
    for m in _JAMMI_REQUIRE_RUN_RE.finditer(content):
        out.append("".join("\n" if c == "\n" else " " for c in content[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append("".join("\n" if c == "\n" else " " for c in content[pos:]))
    return "".join(out)


def _strip_rust(source: str) -> str:
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        two = source[i : i + 2]

        if two == "//":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue

        if two == "/*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth > 0:
                if source[i : i + 2] == "/*":
                    depth += 1
                    out.append("  ")
                    i += 2
                    continue
                if source[i : i + 2] == "*/":
                    depth -= 1
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            continue

        m = _RAW_STR_OPEN_RE.match(source, i)
        if m:
            hashes = m.group(1)
            opener = m.group(0)
            out.append(opener)  # delimiter chars (b/r/#/") kept VERBATIM
            i = m.end()
            closer = '"' + hashes
            end = source.find(closer, i)
            if end == -1:
                end = n
            out.append(_blank_string_content(source[i:end]))
            i = end
            if i < n:
                out.append(closer)
                i += len(closer)
            continue

        if two == 'b"' or source[i] == '"':
            prefix_len = 2 if two == 'b"' else 1
            out.append(source[i : i + prefix_len])  # b/" delimiter kept VERBATIM
            i += prefix_len
            content_start = i
            while i < n and source[i] != '"':
                if source[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            out.append(_blank_string_content(source[content_start:i]))
            if i < n:
                out.append('"')  # closing quote kept VERBATIM
                i += 1
            continue

        # char / byte-char literal, OR a lifetime (`'a`, `'static`) — only a
        # genuine `'<escaped-or-one-char>'` shape is consumed as a literal;
        # anything else (a lifetime) leaves just the quote character alone.
        if source[i] == "'" or two == "b'":
            prefix_len = 2 if two == "b'" else 1
            start = i
            j = i + prefix_len
            consumed_to = None
            if j < n and source[j] == "\\":
                k = j + 1
                if k < n and source[k] == "u" and k + 1 < n and source[k + 1] == "{":
                    k += 2
                    while k < n and source[k] != "}":
                        k += 1
                    if k < n:
                        k += 1
                else:
                    k += 1
                if k < n and source[k] == "'":
                    consumed_to = k + 1
            elif j < n and source[j : j + 1] != "'" and source[j + 1 : j + 2] == "'":
                consumed_to = j + 2
            if consumed_to is not None:
                out.append(" " * (consumed_to - start))
                i = consumed_to
                continue
            out.append(source[i])
            i += 1
            continue

        out.append(source[i])
        i += 1
    return "".join(out)


def _strip_comments_only_independent(source: str) -> str:
    """Comments-only (nested `/* */` + `//`), strings/chars left INTACT — a
    deliberately SEPARATE, simpler implementation (no string/char state at
    all) from `_strip_rust`, used ONLY by the desync check below so a bug
    specific to `_strip_rust`'s string/char tracking cannot also be present
    in its cross-reference.
    """
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        two = source[i : i + 2]
        if two == "//":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth > 0:
                if source[i : i + 2] == "/*":
                    depth += 1
                    out.append("  ")
                    i += 2
                    continue
                if source[i : i + 2] == "*/":
                    depth -= 1
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            continue
        out.append(source[i])
        i += 1
    return "".join(out)


FN_HEAD_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
_ATTR_LINE_RE = re.compile(r"^\s*#!?\[")


# Round-3b (lead probe of the class): a fail-closed check with NO review/
# escape mechanism is a check no conforming file can ever satisfy once a
# real fn-shaped string literal legitimately exists (the live instance:
# crates/jammi-kernels/tests/stateful_op_discipline.rs's own grep-
# discipline fixture, `let attention_block_text = "pub(crate) fn foo() {
# ... }";` — a real, reviewed, intentional test string, not a bug). Mirrors
# this repo's own `no-producer: <reason>` opt-out idiom: fail-closed by
# default, escapable ONLY by a reviewed, per-line marker — never a whole-
# file or whole-repo suppression.
KERNEL_ORACLES_FN_IN_LITERAL_MARKER_RE = re.compile(
    r"//\s*kernel-oracles:\s*fn-in-literal reviewed:\s*(?P<reason>.+?)\s*$"
)


def check_fn_desync(source: str, file_label: str) -> None:
    """PER-LINE fail-closed tripwire: for each line, the count of `fn `
    keyword matches in the FULL stripper's output must equal the count in
    the independent comments-only stripper's output. A line where they
    disagree (a string/char literal on that line contains a `fn <name>(`-
    shaped substring) must carry a `// kernel-oracles: fn-in-literal
    reviewed: <reason>` marker on ITS OWN raw text or the line DIRECTLY
    ABOVE it (raw text, since the marker is itself a `//` comment both
    strippers would otherwise blank) — an unmarked desync line still fails
    closed exactly as before. A marker present on a line that covers NO
    desync (neither its own line nor the line below) is itself a FAIL —
    mirrors the doc-number allowlist's "only shrinks" discipline: a marker
    that no longer corresponds to a real desync must be REMOVED, never
    accumulate as dead weight nobody can tell is still load-bearing.
    """
    lines_raw = source.splitlines()
    n = len(lines_raw)
    lines_full = _strip_rust(source).splitlines()
    lines_comments_only = _strip_comments_only_independent(source).splitlines()
    # length-preserving strippers keep the same line COUNT; defensively pad
    # in case a trailing no-newline partial line differs by one empty entry.
    while len(lines_full) < n:
        lines_full.append("")
    while len(lines_comments_only) < n:
        lines_comments_only.append("")

    desynced = [
        len(FN_HEAD_RE.findall(lines_full[i])) != len(FN_HEAD_RE.findall(lines_comments_only[i]))
        for i in range(n)
    ]
    marked = [bool(KERNEL_ORACLES_FN_IN_LITERAL_MARKER_RE.search(lines_raw[i])) for i in range(n)]

    unmarked_desync = [
        i + 1
        for i in range(n)
        if desynced[i] and not (marked[i] or (i > 0 and marked[i - 1]))
    ]
    stale_markers = [
        i + 1
        for i in range(n)
        if marked[i] and not (desynced[i] or (i + 1 < n and desynced[i + 1]))
    ]

    if unmarked_desync or stale_markers:
        parts = []
        if unmarked_desync:
            parts.append(
                f"unmarked fn-keyword desync at line(s) {', '.join(map(str, unmarked_desync))} "
                "(a string/char literal on that line contains a fn-shaped substring — add "
                "`// kernel-oracles: fn-in-literal reviewed: <reason>` on that line or the "
                "line directly above it, or fix the stripper if this is a real bug)"
            )
        if stale_markers:
            parts.append(
                f"stale `fn-in-literal reviewed` marker at line(s) {', '.join(map(str, stale_markers))} "
                "(no fn-keyword desync on that line or the line below it anymore — remove the marker)"
            )
        raise OracleError(f"{file_label}: " + "; ".join(parts))


def _extract_balanced_block(text: str, open_brace_idx: int) -> tuple[str, int, int]:
    """Returns (block_text_including_braces, open_idx, close_idx) — brace-
    balanced from `open_brace_idx` (which MUST point at a `{`). Generic:
    used for both fn bodies and `if`-block bodies.
    """
    depth = 0
    for i in range(open_brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx : i + 1], open_brace_idx, i
    return text[open_brace_idx:], open_brace_idx, len(text) - 1


@dataclass(frozen=True)
class FnRecord:
    name: str
    file: str
    body: str  # RAW slice (same offsets as body_stripped; comments/strings intact)
    body_stripped: str  # the ONE canonical stripped body (same length as `body`)
    body_start_idx: int  # offset of the fn's own `{` within `source`
    is_test: bool


def _has_test_attr(lines: list[str], fn_line_idx: int) -> bool:
    j = fn_line_idx - 1
    while j >= 0 and _ATTR_LINE_RE.match(lines[j]):
        if "test" in lines[j].lower():
            return True
        j -= 1
    return False


def find_fns(source: str, file_label: str) -> list[FnRecord]:
    """Every `fn <name>(...) { ... }` in `source`. `check_fn_desync` runs
    FIRST (fail-closed, once per file). Both fn-BOUNDARY detection
    (`FN_HEAD_RE`) and fn-BODY brace counting now run entirely on the ONE
    stripped text (round-3 audit item 2 — a `}` character sitting merely
    inside a string literal, e.g. `println!("brace }} }} literal")`, can no
    longer truncate a fn body early, which previously swallowed every
    subsequent fn in the file into one bogus mega-body). `body`/`body_
    stripped` are sliced from the SAME [start, end) range out of `source`
    and the stripped text respectively — valid because stripping is
    length-preserving.
    """
    check_fn_desync(source, file_label)
    stripped_full = _strip_rust(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def line_idx_of(pos: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    records: list[FnRecord] = []
    plain_lines = [line.rstrip("\n") for line in lines]
    for m in FN_HEAD_RE.finditer(stripped_full):
        name = m.group(1)
        fn_line_idx = line_idx_of(m.start())
        is_test = _has_test_attr(plain_lines, fn_line_idx)
        brace_start = stripped_full.find("{", m.end())
        if brace_start == -1:
            continue
        body_stripped, body_start, body_end = _extract_balanced_block(stripped_full, brace_start)
        body_raw = source[body_start : body_end + 1]
        records.append(
            FnRecord(
                name=name,
                file=file_label,
                body=body_raw,
                body_stripped=body_stripped,
                body_start_idx=body_start,
                is_test=is_test,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# KO-7 — unrun-is-RED
# --------------------------------------------------------------------------- #
# Round-3 audit item 6: registration requires the panic to be on the ENV
# READ'S OWN CONTROL PATH — only an `if <cond containing the env-read> {
# ... panic!/unreachable!/.expect( ... }` shape registers, checked entirely
# on the ONE stripped text (the env-var literal survives stripping — see
# `_blank_string_content`'s `JAMMI_REQUIRE_` exception above). This closes
# the class where an env-read result is DISCARDED (`let _ = option_env!
# (...)`, never read at all) or read but not on any panic's own path (`if
# ... { eprintln!(...); }` followed by an UNRELATED `.expect(` elsewhere in
# the fn) — neither shape is a real require-gate.
IF_ENV_READ_RE = re.compile(
    r"\bif\b[^{]*?\b(?:std::env::var_os|env::var_os|std::env::var|env::var|option_env!)"
    r'\s*\(\s*"(JAMMI_REQUIRE_[A-Z0-9_]*)"[^{]*\{'
)
PANIC_REACHABLE_RE = re.compile(r"\b(?:panic!|unreachable!)\s*\(|\.expect\s*\(")
# `return;` / brace-tail `return}`, plus `return Ok(...)`/`return Err(...)`
# — a #[test] fn returning `Result<(), E>` that early-exits via either
# shape is just as much a silent skip as a bare `return;`. Only the START
# of the statement is matched (sufficient for textual ordering/windowing
# below); the rest of a multi-token `Ok(...)` expression is not consumed.
RETURN_SKIP_RE = re.compile(r"\breturn\b\s*(?:;|\}|Ok\s*\(|Err\s*\()")


def find_require_gate_helpers(all_fns: list[FnRecord]) -> set[str]:
    """A fn is a REGISTERED require-gate helper iff its stripped body
    contains an `if <env-read-of-JAMMI_REQUIRE_...> { ... }` shape whose
    OWN block body (brace-balanced, extracted from that SAME `if`'s `{`)
    contains a reachable `panic!(`/`unreachable!(`/`.expect(` — discovered
    by shape, never a hardcoded name list (see module doc).
    """
    helpers: set[str] = set()
    for fn in all_fns:
        for m in IF_ENV_READ_RE.finditer(fn.body_stripped):
            brace_idx = m.end() - 1
            if_block, _, _ = _extract_balanced_block(fn.body_stripped, brace_idx)
            if PANIC_REACHABLE_RE.search(if_block):
                helpers.add(fn.name)
                break
    return helpers


@dataclass(frozen=True)
class UngatedSkip:
    file: str
    fn_name: str
    line_no: int


def check_ko7(all_fns: list[FnRecord], helper_names: set[str], source_texts: dict[str, str]) -> list[UngatedSkip]:
    """Every `return;`/`return}`/`return Ok(`/`return Err(` inside every
    `#[test]` fn's STRIPPED body must be textually dominated by a call to a
    name in `helper_names` — PER SKIP: the dominance window for a given
    `return` is `[end of the PREVIOUS skip (or fn start), this skip's
    start)`, so an early helper call gates only the skip(s) immediately
    downstream of it, never every later skip in the same fn unconditionally
    — a gated CUDA-device check followed by an UNRELATED, ungated
    `if !FLASH_COMPILED { return; }` further down the same fn still reds.
    """
    helper_call_res = {name: re.compile(rf"\b{re.escape(name)}\s*\(") for name in helper_names}
    findings: list[UngatedSkip] = []
    for fn in all_fns:
        if not fn.is_test:
            continue
        skip_matches = sorted(RETURN_SKIP_RE.finditer(fn.body_stripped), key=lambda m: m.start())
        if not skip_matches:
            continue
        helper_positions = sorted(
            m.start() for cre in helper_call_res.values() for m in cre.finditer(fn.body_stripped)
        )
        window_start = 0
        for m in skip_matches:
            pos = m.start()
            gated = any(window_start <= hp < pos for hp in helper_positions)
            if not gated:
                full = source_texts[fn.file]
                abs_pos = fn.body_start_idx + pos
                line_no = full.count("\n", 0, abs_pos) + 1
                findings.append(UngatedSkip(file=fn.file, fn_name=fn.name, line_no=line_no))
            window_start = m.end()
    return findings


# --------------------------------------------------------------------------- #
# KO-2 — bound coverage parity (marker-scoped)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ko2Finding:
    marker: OracleCellMarker
    missing_bounds: tuple[str, ...] = ()
    control_not_found: bool = False


ASSERT_MACRO_RE = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne)!\s*\("
)
# Round-3 audit item 7: a bare `<`/`>` in Rust is FAR more often a generic
# bracket (`Vec<f32>`, `foo::<f32>(...)`) than a comparison — those never
# carry surrounding whitespace, while a real comparison (this repo's own
# rustfmt convention) does. `<=`/`>=`/`==` are unambiguous either way (no
# valid generic-bracket shape produces them), so only the bare-`<`/`>` forms
# require the space padding.
_COMPARISON_OP_RE_KO2 = re.compile(r"<=|>=|==| < | > ")


def _extract_paren_balanced(text: str, open_paren_idx: int) -> str:
    depth = 0
    for i in range(open_paren_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx : i + 1]
    return text[open_paren_idx:]


def _bound_in_assertion_context(body_stripped: str, bound_name: str) -> bool:
    """`bound_name` (an identifier) appears EITHER inside an assert!-family
    macro call's argument list, OR DIRECTLY ADJACENT to a comparison
    operator (immediately before or after it, only whitespace in between) —
    round-3 audit fix: "same statement chunk contains both, anywhere" was
    too permissive (`let f = compute(TOL) as u32 > 0;` — TOL feeds a value
    that is later compared, but TOL itself is never an operand of that
    comparison — must MISS); direct adjacency requires TOL to be one of the
    comparison's own two sides.
    """
    name_re = re.compile(rf"\b{re.escape(bound_name)}\b")

    for m in ASSERT_MACRO_RE.finditer(body_stripped):
        args = _extract_paren_balanced(body_stripped, m.end() - 1)
        if name_re.search(args):
            return True

    name_before_op_re = re.compile(rf"\b{re.escape(bound_name)}\b\s*(?:<=|>=|==| < | > )")
    op_before_name_re = re.compile(rf"(?:<=|>=|==| < | > )\s*{re.escape(bound_name)}\b")
    if name_before_op_re.search(body_stripped) or op_before_name_re.search(body_stripped):
        return True

    return False


def check_ko2(markers: list[OracleCellMarker], all_fns: list[FnRecord]) -> list[Ko2Finding]:
    """The `control` fn is resolved ONLY within the marker's OWN FILE — a
    same-named fn elsewhere in the scanned tree is never consulted. A bound
    is covered iff `_bound_in_assertion_context` holds directly on the
    control fn, or (one level of indirection) on a SAME-FILE fn the control
    calls.
    """
    by_file_name: dict[str, dict[str, list[FnRecord]]] = {}
    for fn in all_fns:
        by_file_name.setdefault(fn.file, {}).setdefault(fn.name, []).append(fn)

    findings: list[Ko2Finding] = []
    for marker in markers:
        if marker.is_control_none:
            continue
        same_file_fns = by_file_name.get(marker.file, {})
        candidates = same_file_fns.get(marker.control)
        if not candidates:
            findings.append(Ko2Finding(marker=marker, control_not_found=True))
            continue

        missing: list[str] = []
        for bound in marker.bounds:
            if any(_bound_in_assertion_context(c.body_stripped, bound) for c in candidates):
                continue
            via_helper = False
            for c in candidates:
                for callee_name, callee_fns in same_file_fns.items():
                    if callee_name == marker.control:
                        continue
                    if not re.search(rf"\b{re.escape(callee_name)}\s*\(", c.body_stripped):
                        continue
                    if any(_bound_in_assertion_context(cf.body_stripped, bound) for cf in callee_fns):
                        via_helper = True
                        break
                if via_helper:
                    break
            if not via_helper:
                missing.append(bound)
        if missing:
            findings.append(Ko2Finding(marker=marker, missing_bounds=tuple(missing)))
    return findings


# --------------------------------------------------------------------------- #
# KO-5 — off-sample bounds (marker-scoped, round-2 audit item 8)
# --------------------------------------------------------------------------- #
def _normalize_seed_token(tok: str) -> int | str:
    """`42` == `042` == `0x2a` — a seed is compared as its VALUE, not its
    spelling. A non-numeric token (a named seed like `seedA`) falls back to
    comparing as its raw string.
    """
    t = tok.strip()
    try:
        if t.lower().startswith("0x"):
            return int(t, 16)
        return int(t, 10)
    except ValueError:
        return t


def check_ko5(markers: list[OracleCellMarker]) -> list[OracleCellMarker]:
    """Markers whose `derived-on` and `asserted-on` seed sets are BOTH real
    lists (neither the literal `none`) and intersect, AFTER normalizing
    every token to an integer where possible — circular calibration.
    """
    findings: list[OracleCellMarker] = []
    for marker in markers:
        if marker.derived_on is None or marker.asserted_on is None:
            continue
        derived_norm = {_normalize_seed_token(s) for s in marker.derived_on}
        asserted_norm = {_normalize_seed_token(s) for s in marker.asserted_on}
        if derived_norm & asserted_norm:
            findings.append(marker)
    return findings


# --------------------------------------------------------------------------- #
# reconciliation — op domain × marker coverage (round-2 audit item 3 adds
# DECLARED_UNCONTROLLED as a fourth category)
# --------------------------------------------------------------------------- #
# Reviewed, in-script. Empty in v1 — no op has been reviewed off the op
# domain yet; kept as a dict (not omitted) so a future round has somewhere
# to add a reviewed exclusion without inventing a second structure.
STRUCTURALLY_EXCLUDED_OPS: dict[str, str] = {}


def reconcile_ops(
    shipped: set[str],
    covered: dict[str, list[OracleCellMarker]],
    excluded: dict[str, str],
    declared_uncontrolled: dict[str, list[OracleCellMarker]],
) -> tuple[dict[str, str], list[str]]:
    """Returns (pending_with_reason, failures). PENDING is COMPUTED (the
    complement of COVERED ∪ EXCLUDED ∪ DECLARED_UNCONTROLLED within
    SHIPPED), never hand-curated — v1 has no CONTROLLED markers, so every
    un-excluded, un-declared op is honestly PENDING; a reviewer only ever
    ADDS to `STRUCTURALLY_EXCLUDED_OPS`, lands a real controlled marker, or
    declares `control=none:<reason>`, never edits a PENDING dict directly.
    A `control=none:<reason>` marker can NEVER move its op into COVERED —
    `covered` here is pre-filtered by the caller (`run_gate`) to markers
    with a REAL control only.
    """
    failures: list[str] = []

    for op in covered:
        if op not in shipped:
            failures.append(f"COVERED marker references unknown op `{op}` — not in the SHIPPED admission-key set")
    for op in excluded:
        if op not in shipped:
            failures.append(f"STRUCTURALLY_EXCLUDED_OPS entry `{op}` references unknown op — not in the SHIPPED admission-key set")
    for op in declared_uncontrolled:
        if op not in shipped:
            failures.append(f"DECLARED_UNCONTROLLED entry `{op}` references unknown op — not in the SHIPPED admission-key set")

    pairs = [
        (set(covered), set(excluded), "COVERED and STRUCTURALLY_EXCLUDED"),
        (set(covered), set(declared_uncontrolled), "COVERED and DECLARED_UNCONTROLLED"),
        (set(excluded), set(declared_uncontrolled), "STRUCTURALLY_EXCLUDED and DECLARED_UNCONTROLLED"),
    ]
    for left, right, label in pairs:
        for op in sorted(left & right):
            failures.append(f"op `{op}` is claimed both {label} — remove one")

    pending_reason = (
        "shipped admission key with no CONTROLLED `oracle-cell` marker yet — v1 scope excludes "
        "every numerics-owned file (crates/jammi-kernels, crates/jammi-encoders); tracked debt, "
        "see docs/maintainer/cuda-kernel-guide.md §3's kernel-acceptance-oracle standard."
    )
    accounted = set(covered) | set(excluded) | set(declared_uncontrolled)
    pending = {op: pending_reason for op in sorted(shipped - accounted)}
    return pending, failures


def print_reconciliation(
    shipped: set[str],
    covered: dict[str, list[OracleCellMarker]],
    excluded: dict[str, str],
    declared_uncontrolled: dict[str, list[OracleCellMarker]],
    pending: dict[str, str],
) -> None:
    print("kernel-oracle op reconciliation:")
    for op in sorted(shipped):
        if op in covered:
            legs = ", ".join(f"{m.file}:{m.line_no}" for m in covered[op])
            print(f"    COVERED                {op}  <- {legs}")
        elif op in excluded:
            print(f"    STRUCTURALLY_EXCL      {op}  — {excluded[op]}")
        elif op in declared_uncontrolled:
            reasons = sorted(
                {m.control.split(":", 1)[1] if ":" in m.control else m.control for m in declared_uncontrolled[op]}
            )
            print(f"    DECLARED_UNCONTROLLED  {op}  — {', '.join(reasons)}")
        elif op in pending:
            print(f"    PENDING                {op}")
        else:
            print(f"    !!!! UNACCOUNTED !!!!  {op}")
    print(
        f"\nSummary: {len(covered)} COVERED, {len(excluded)} STRUCTURALLY_EXCLUDED, "
        f"{len(declared_uncontrolled)} DECLARED_UNCONTROLLED, {len(pending)} PENDING "
        f"out of {len(shipped)} SHIPPED admission keys."
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def scan_files() -> dict[str, str]:
    if not KERNELS_TESTS_DIR.is_dir():
        raise OracleError(f"scan root not found: {KERNELS_TESTS_DIR}")
    if not ENCODERS_SRC_DIR.is_dir():
        raise OracleError(f"scan root not found: {ENCODERS_SRC_DIR}")
    texts: dict[str, str] = {}
    for path in sorted(KERNELS_TESTS_DIR.glob("*.rs")):
        rel = str(path.relative_to(REPO_ROOT))
        texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
    for path in sorted(ENCODERS_SRC_DIR.glob("*.rs")):
        rel = str(path.relative_to(REPO_ROOT))
        texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def run_gate(source_texts: dict[str, str], shipped_ops: set[str]) -> tuple[
    list[UngatedSkip],
    list[Ko2Finding],
    list[OracleCellMarker],
    dict[str, list[OracleCellMarker]],
    dict[str, list[OracleCellMarker]],
    dict[str, str],
    list[str],
]:
    """Pure orchestration over `{file: text}` + the SHIPPED op set — the
    self-test seam. Returns (ko7_findings, ko2_findings, ko5_findings,
    covered, declared_uncontrolled, pending, reconciliation_failures).
    """
    all_fns: list[FnRecord] = []
    all_markers: list[OracleCellMarker] = []
    for file_label, text in source_texts.items():
        all_fns.extend(find_fns(text, file_label))
        all_markers.extend(parse_markers(text, file_label))

    helper_names = find_require_gate_helpers(all_fns)
    ko7 = check_ko7(all_fns, helper_names, source_texts)
    ko2 = check_ko2(all_markers, all_fns)
    ko5 = check_ko5(all_markers)

    markers_by_op: dict[str, list[OracleCellMarker]] = {}
    for marker in all_markers:
        markers_by_op.setdefault(marker.op, []).append(marker)

    covered: dict[str, list[OracleCellMarker]] = {}
    declared_uncontrolled: dict[str, list[OracleCellMarker]] = {}
    for op, markers_for_op in markers_by_op.items():
        controlled = [m for m in markers_for_op if not m.is_control_none]
        if controlled:
            covered[op] = controlled
        else:
            declared_uncontrolled[op] = markers_for_op

    pending, recon_failures = reconcile_ops(shipped_ops, covered, STRUCTURALLY_EXCLUDED_OPS, declared_uncontrolled)

    return ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures


def main() -> int:
    try:
        source_texts = scan_files()
        shipped_ops = load_shipped_ops()
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures = run_gate(source_texts, shipped_ops)
    except OracleError as exc:
        print(f"kernel-oracles: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    print_reconciliation(shipped_ops, covered, STRUCTURALLY_EXCLUDED_OPS, declared_uncontrolled, pending)

    ko7_by_file: dict[str, list[UngatedSkip]] = {}
    for f in ko7:
        ko7_by_file.setdefault(f.file, []).append(f)
    print("\nKO-7 (unrun-is-RED) per scanned file:")
    for file_label, text in source_texts.items():
        fns_in_file = find_fns(text, file_label)
        n_skips = sum(
            len(RETURN_SKIP_RE.findall(fn.body_stripped)) for fn in fns_in_file if fn.is_test
        )
        n_ungated = len(ko7_by_file.get(file_label, []))
        print(f"    {file_label}: {n_skips} runtime skip(s), {n_ungated} ungated")

    failures: list[str] = list(recon_failures)
    for f in ko7:
        failures.append(f"KO-7: {f.file}:{f.line_no} — ungated runtime skip in `#[test] fn {f.fn_name}` (no registered require-gate helper called in the window since the previous skip)")
    for f in ko2:
        if f.control_not_found:
            failures.append(f"KO-2: {f.marker.file}:{f.marker.line_no} — oracle-cell control fn `{f.marker.control}` not found in the marker's own file")
        else:
            failures.append(f"KO-2: {f.marker.file}:{f.marker.line_no} — control fn `{f.marker.control}` does not assert against bound(s) {', '.join(f.missing_bounds)} inside an assertion/comparison")
    for f in ko5:
        derived_norm = {_normalize_seed_token(s) for s in (f.derived_on or ())}
        asserted_norm = {_normalize_seed_token(s) for s in (f.asserted_on or ())}
        overlap = sorted(str(s) for s in (derived_norm & asserted_norm))
        failures.append(f"KO-5: {f.file}:{f.line_no} — derived-on/asserted-on share seed(s) {', '.join(overlap)} (circular calibration)")

    if failures:
        print("\nkernel-oracles: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\nkernel-oracles: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        "\nkernel-oracles: PASS — every scanned #[test] runtime skip is gated per-skip (KO-7); "
        "every marked oracle-cell's bounds are asserted/compared by its file-scoped control (KO-2) "
        "and derived/asserted on disjoint (integer-normalized) seeds (KO-5); reconciliation is "
        "fully accounted across all four categories (v1: 0 controlled markers, every SHIPPED op "
        "honestly PENDING or DECLARED_UNCONTROLLED)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
