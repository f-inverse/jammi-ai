#!/usr/bin/env python3
"""Kernel-acceptance-oracle standard gate (v1) — hermetic, static, no build, no GPU.

Mechanizes five of the eight `KO-1`..`KO-8` ids `docs/maintainer/cuda-kernel-guide.md`
§3's conformance list reports (`check_doc_parity.py` binds that list's set to this
file's `STABLE_IDS`, the same way it binds every other guide↔code enumeration —
see that module's `KernelOracleStandardIds` binding). The other three (`KO-1`,
`KO-6`, `KO-8`) are auditor-only by design — each requires running code or a
semantic judgment a static scan cannot make; the guide states, for each, the one
sentence explaining why.

## KO-7 — unrun-is-RED (TOTAL over the RECOGNIZED skip shapes, enforced on every
## scanned file today)

A `#[test]` fn that silently skips partway through still reports green — the
oracle never actually ran. `RETURN_SKIP_RE` enumerates the RECOGNIZED skip
shapes (round-5 audit NF-1 — this list, not "every way Rust can early-return",
is what "TOTAL" means below): a bare `return` immediately followed by `;`
(`let Some(dev) = cuda_device() else { return; };`), `}` (a brace-tail
expression, `else { return }`), `,` (a match-arm tail expression, `None =>
return,`), `Ok(...)`, or `Err(...)`; or a `process::exit(...)`/`std::process::
exit(...)` call. That is only acceptable when
the skip is GATED: dominated, textually, by a call to a name in the REVIEWED
registry `ci/kernel-oracle-helpers.txt` (`load_helper_registry`) — never by
scanning any fn's body for a shape that merely LOOKS like a require-gate. A
regex/lexical scanner cannot establish "this fn IS a genuine, reviewed
require-gate helper" as a syntactic fact from shape alone — any shape a real
helper has, a decoy can imitate (an unrelated `.expect(` inside the if-block, an
uncalled closure's `panic!`, a conforming gate written as `match` instead of
`if`, a shadowed import). So gating asks "is this NAME in the reviewed
registry?", never "does this look like a gate?" — a decoy can share a real
helper's exact shape and it still never gates anything unless its OWN name is
registered.

A registry entry is not a rubber stamp: `verify_helper_registry` re-checks each
one has the CANONICAL shape — a real env-read (`std::env::var_os`/`var`/
`option_env!`) of a `JAMMI_REQUIRE_*` name, via `if` OR `match` (both accepted),
whose taken-when-set branch consists of EXACTLY ONE statement — a
`panic!(...)`/`unreachable!(...)` invocation, nothing else (no `.expect(`, no
closures, no additional statements) — or the gate is a REGISTRY FAIL naming the
file/fn and why. Registering a new helper is a reviewed PR diff, exactly like
adding a citation or a `no-producer:` opt-out; the registry may only grow by
entries the shape check accepts.

"Dominated" is a TEXTUAL, PER-SKIP check (this is a lexical scanner, not a
control-flow analyzer — it cannot prove REACHABILITY, so a registered helper's
call sitting in a genuinely dead branch, e.g. `if false { cuda_device(); }`,
still counts as "called before the skip" — a documented, out-of-scope class
limitation, not a bug this gate closes): within the SAME `#[test]` fn body, each
`return`-shaped skip's dominance window runs from the END of the PREVIOUS skip
(or fn start) to THAT skip's own start — an earlier helper call gates only the
skip(s) immediately downstream of it, never every later skip in the same fn
unconditionally.

Fn discovery (`find_fns`) and `is_test` classification both run over the ONE
canonical stripped text (`_strip_rust` — comments, including NESTED block
comments, and string/char-literal CONTENTS all blanked; a `JAMMI_REQUIRE_*` run
inside a string survives, since that is data the registry's shape check must
read, not prose). An attribute (`#[test]`, `#[tokio::test]`, a multi-line
`#[cfg_attr(...)]`, ...) is associated with the fn item DIRECTLY below it — only
whitespace in between, which the stripped text already reduces every doc
comment/line comment/blank line to — walking BACK through the whole CONTIGUOUS
chain of stacked attributes and past any `pub`/`pub(...)`/`async`/`unsafe`/
`const`/`extern ".."` modifier keyword between the last attribute and the `fn`
keyword itself. An attribute's own leading PATH decides test-ness (`test`, or
anything ending `::test`) — never a bare substring match, which would (and once
did) misread `#[cfg(test)]` — TEST-BUILD-ONLY code, of any kind, not itself a
`#[test]` function — as a test attribute. `check_test_attr_totality` is a
SEPARATE, fail-closed cross-check run after classification: the count of
`#[test]`-shaped attribute TOKENS in the file must exactly equal the number of
fns classified `is_test=True` — a genuine mismatch (an orphaned test attribute
attached to nothing, or two stacked before one fn) makes the file UNCOMPUTABLE,
never silently mis-counted.

`check_fn_desync` (run first, per file) is a SEPARATE, PER-LINE fail-closed
tripwire: for each line, the count of `fn ` keyword matches in the fully
stripped text must equal the count in an INDEPENDENTLY-implemented comments-only
stripper's text — a string/char literal containing a `fn <name>(`-shaped
substring (e.g. a grep-discipline test fixture whose OWN content is a literal
Rust snippet) desyncs the two counts on that one line. A desync line is
escapable ONLY by a reviewed `// kernel-oracles: fn-in-literal reviewed:
<reason>` marker on that line or the line directly above it (mirrors this
repo's own `no-producer: <reason>` opt-out idiom); an unmarked desync line still
fails closed, and a marker covering NO desync (on its own line or the line
below) is itself a FAIL — a marker that no longer corresponds to a real desync
must be removed, never accumulate as unreviewable dead weight.

Scanned files: `crates/jammi-kernels/tests/**/*.rs`, `crates/jammi-encoders/
src/**/*.rs` (recursive under both roots; test modules live inline in the
latter; scanning the whole file costs nothing — non-test code simply carries
no `#[test]` fn to check). KO-7 is TOTAL, over EVERY recognized skip shape
(above) inside EVERY `#[test]` fn in EVERY scanned file, not just new/changed
ones — no diff-base input is needed in v1 for this reason. This is a
deliberately narrower claim than "every possible way a Rust fn can exit
early": a skip idiom not on the recognized-shapes list above is not currently
detectable by this lexical scanner — a genuinely new shape is a gap to close
(widen `RETURN_SKIP_RE` and add a regression case), not something this claim
already covers. Fail-closed: any ungated skip, any registry entry that does
not verify, or any desync/totality mismatch is a non-zero exit naming
file:line (or file:fn for a registry failure). Scope: RUNTIME skips only (an
early `return`/`process::exit` inside the test body) — this rule says nothing
about `#[ignore]`/`#[cfg(feature = ...)]` compile-time gating, which
`check_cuda_run_artifacts.py` rule (c) already verifies per-artifact.

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

**KO-2 (bound coverage parity).** When `control` names a real fn, that fn — and
ONLY that fn's file (never a same-named fn elsewhere in the tree) — is resolved,
and a bound is covered iff, checked in this order:

  1. It is the `epsilon =`/`max_relative =` OPERAND of an `approx::
     assert_relative_eq!`/`assert_abs_diff_eq!` call anywhere in that call's
     argument list (these macros carry no literal comparison operator at all —
     the named parameter itself IS the bound operand).
  2. It is PRESENT in the FIRST top-level-comma-delimited argument (the
     CONDITION) of any other `assert!`-family macro call — message/format
     arguments are NEVER inspected, so `assert!(d < 1e-3, "tol was {}", TOL)`
     does not cover `TOL`.
  3. With no enclosing assert! call at all (a bare `if`/boolean-return-
     expression control, e.g. `within_bound`), it is DIRECTLY ADJACENT — only
     whitespace in between — to one of `<=`/`>=`/`==`/`<`/`>`'s own two operand
     positions, anywhere in the fn body. Adjacency (not mere co-occurrence in
     the same statement) is what keeps `let f = compute(TOL) as u32 > 0;`
     correctly UNCOVERED (`TOL` feeds a value that is later compared, but is
     never itself an operand).

A bare declaration or passing mention never counts; one level of same-file
helper-fn indirection is followed (a fn the control CALLS, checked the same
way). `control=none:<reason>` opts a cell out of KO-2 entirely — it becomes
DECLARED_UNCONTROLLED (see reconciliation below), never silently passed.

Known, out-of-scope class limitations (documented, not silently claimed fixed):
a bound compared to ITSELF (`assert!(TOL == TOL)`), a bound used inside a
provably-dead branch (`if false { assert!(d < TOL); } `), a bound SHADOWED by
a later same-name local (`let TOL = 1e9; assert!(d < TOL);`), and (round-4
audit — the H3 fix's own honest cost, dropping round-3's space-padding
requirement to stop it from rejecting a real unpadded `if d<TOL {...}`) a
GENERIC BRACKET (`let v: Vec<TOL> = ..;`, `foo::<TOL>(..)`) all still read as
COVERED — none of these are syntactic facts a lexical scanner can refute;
each needs real value/reachability/scope/type analysis this gate does not
attempt. A false COVERED is a silent miss in the SAFE direction for KO-2's
own stated job (KO-2 only ever flags MISSING coverage), not a new class of
false alarm — narrowing adjacency back to reject `Vec<TOL>` without also
reintroducing the H3 regression is future work, not attempted this round.

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
  - Any registry entry (`ci/kernel-oracle-helpers.txt`) whose file/fn does not
    resolve, or whose body does not have the canonical require-gate shape, is a
    non-zero exit naming the entry and why.
  - Any per-line fn-keyword desync with no reviewed marker, or a marker
    covering no real desync, is a non-zero exit naming the file and line(s).
  - Any file whose `#[test]`-attribute-token count does not exactly match its
    classified-test-fn count is a non-zero exit naming both counts.
  - Any marker whose `control` fn does not cover every `bounds` identifier
    (KO-2), or whose `derived-on`/`asserted-on` seed sets intersect (KO-5), is
    a non-zero exit naming the marker's file:line.
  - Any marker naming an `op` outside the SHIPPED set, or a malformed marker
    line, is a non-zero exit (mirrors `check_gpu_parity_matrix.py`).
  - Any op claimed by more than one of COVERED / STRUCTURALLY_EXCLUDED /
    DECLARED_UNCONTROLLED is a non-zero exit (PENDING is computed as the
    complement, so it cannot conflict with the other three by construction).
  - A parse failure (missing scan root, uncomputable op domain, missing
    registry file) is a non-zero exit naming what could not be resolved.

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


# Round-5 audit advisory A3: a reviewed marker (`// kernel-oracles: ...`)
# matched against RAW source text is fooled by a STRING LITERAL that
# merely CONTAINS marker-shaped text (`let b = "// kernel-oracles: ...
# reviewed: fake";` — no real comment there at all) into treating a
# genuine, unmarked desync/mismatch as reviewed. A marker only counts if
# it survives THIS stripper — string/char CONTENT blanked (reusing the
# exact same string/char-tracking logic as `_strip_rust`, so a bug in one
# cannot diverge from the other), comments and code left VERBATIM (the
# opposite trade of `_strip_rust`, which blanks comments too).
def _strip_strings_only(source: str) -> str:
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        two = source[i : i + 2]

        m = _RAW_STR_OPEN_RE.match(source, i)
        if m:
            hashes = m.group(1)
            opener = m.group(0)
            out.append(opener)
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
            out.append(source[i : i + prefix_len])
            i += prefix_len
            content_start = i
            while i < n and source[i] != '"':
                if source[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            out.append(_blank_string_content(source[content_start:i]))
            if i < n:
                out.append('"')
                i += 1
            continue

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


# Round-4 audit item 3: a raw identifier (`fn r#match(`, legal Rust for a
# name that collides with a keyword) or a macro_rules! metavariable
# (`fn $name(`, the standard "generate N #[test] fns from one template"
# idiom) previously matched NEITHER alternative, so the fn was invisible to
# `find_fns` while its `#[test]` attribute was still counted by the
# attribute scan — a totality mismatch with no remedy. Both are accepted as
# valid "name" shapes now, closing the class at its source instead of only
# via the marker escape hatch below (`check_test_attr_totality`).
FN_HEAD_RE = re.compile(r"\bfn\s+((?:r#)?[A-Za-z_][A-Za-z0-9_]*|\$[A-Za-z_][A-Za-z0-9_]*)\s*[(<]")


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
    # Round-5 audit A3: the marker is matched against strings-blanked text
    # (comments/code left verbatim) — NOT raw text — so a marker-shaped
    # SUBSTRING that only exists inside some OTHER string literal on the
    # line (`let b = "// kernel-oracles: fn-in-literal reviewed: fake";`)
    # cannot masquerade as a real reviewed comment.
    lines_marker_view = _strip_strings_only(source).splitlines()
    # length-preserving strippers keep the same line COUNT; defensively pad
    # in case a trailing no-newline partial line differs by one empty entry.
    while len(lines_full) < n:
        lines_full.append("")
    while len(lines_comments_only) < n:
        lines_comments_only.append("")
    while len(lines_marker_view) < n:
        lines_marker_view.append("")

    desynced = [
        len(FN_HEAD_RE.findall(lines_full[i])) != len(FN_HEAD_RE.findall(lines_comments_only[i]))
        for i in range(n)
    ]
    marked = [bool(KERNEL_ORACLES_FN_IN_LITERAL_MARKER_RE.search(lines_marker_view[i])) for i in range(n)]

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


# --------------------------------------------------------------------------- #
# Round-4 (lead probe of the class, item N1/item 2): a regex/lexical scanner
# cannot establish a syntactic fact — attribute-to-item ASSOCIATION — by
# walking RAW lines backward through a CONTIGUOUS run of `#[...]`-shaped
# lines. That breaks on (proven by fixtures G1-G5): `#[test]` on the SAME
# line as `fn` (G1); a `///`/`//` comment OR a blank line between the
# attribute and the `fn` (G2-G4, both legal, common Rust style); a
# MULTI-LINE attribute — e.g. `#[cfg_attr(\n    not(feature = "cuda"),\n
# ignore\n)]` — stacked above `#[test]` (G5). Every one of these silently
# sets `is_test=False`, making that fn's skips INVISIBLE to KO-7 — the
# EXACT "unrun reads green" failure this whole standard exists to prevent,
# just one level up (the test itself, not its skip, goes dark).
#
# Fixed by working on the ONE stripped text (comments/blank-content are
# ALREADY whitespace there, so G2-G4 are transparently bridged) and by
# discovering every `#[...]`/`#![...]` ATTRIBUTE as its own bracket-
# balanced SPAN (so a multi-line attribute, G5, is one unit, not "not an
# attribute line"), then associating a `fn` with the CONTIGUOUS (only
# whitespace between) chain of attribute spans immediately preceding it —
# walking that chain, not merely the ONE span closest to the item.
#
# A single substring check (`"test" in line`) is ALSO wrong the other
# direction: `#[cfg(test)]` (test-BUILD-only code — any kind of fn, not a
# `#[test]` test function) contains "test" as a substring and was being
# misread as a test attribute (found on `crates/jammi-encoders/src/
# attention.rs`'s `in_proj_weight`, a `#[cfg(test)]`-gated ACCESSOR, not a
# test — a genuine false POSITIVE this round's own audit tooling surfaced
# independently). `_ATTR_PATH_RE` reads the attribute's own leading path
# (`test`, `tokio::test`, `cfg`, `cfg_attr`, ...) so only a path that IS
# `test` or ENDS `::test` counts.
_ATTR_SPAN_START_RE = re.compile(r"#!?\[")
_ATTR_PATH_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_:]*)")


def _find_attribute_spans(stripped: str) -> list[tuple[int, int, str]]:
    """Every `#[...]`/`#![...]` in `stripped`, bracket-balanced (so a
    multi-line attribute is ONE span), as `(start, end, path)` — `end` is
    EXCLUSIVE (one past the closing `]`); `path` is the attribute's own
    leading identifier path, used to decide test-ness (`_is_test_attr_
    path`) without a substring match.
    """
    spans: list[tuple[int, int, str]] = []
    for m in _ATTR_SPAN_START_RE.finditer(stripped):
        bracket_start = m.end() - 1  # position of the '['
        depth = 0
        end = None
        for j in range(bracket_start, len(stripped)):
            if stripped[j] == "[":
                depth += 1
            elif stripped[j] == "]":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            continue
        inner = stripped[bracket_start + 1 : end - 1]
        path_m = _ATTR_PATH_RE.match(inner)
        path = path_m.group(1) if path_m else ""
        spans.append((m.start(), end, path))
    return spans


def _is_test_attr_path(path: str) -> bool:
    return path == "test" or path.endswith("::test")


# An attribute attaches to the WHOLE item, not literally the `fn` token —
# `async fn`/`pub fn`/`pub(crate) fn`/`const fn`/`unsafe fn`/`extern "C"
# fn` all put a modifier keyword BETWEEN the attribute and `fn` itself.
# Anchored with `$` and searched over `stripped[:fn_kw_start]` so it finds
# the item's TRUE start (every alternative optional, so a bare `fn` with
# no modifiers correctly returns `fn_kw_start` itself unchanged).
_ITEM_PREFIX_RE = re.compile(
    r"(?:\bpub(?:\([^()]*\))?\s+)?(?:\bconst\s+)?(?:\basync\s+)?(?:\bunsafe\s+)?"
    r'(?:\bextern\s+"[^"]*"\s+)?$'
)


def _item_start_before_fn(stripped: str, fn_kw_start: int) -> int:
    m = _ITEM_PREFIX_RE.search(stripped[:fn_kw_start])
    return m.start() if m else fn_kw_start


def _has_test_attr(stripped: str, attr_spans: list[tuple[int, int, str]], fn_kw_start: int) -> bool:
    """True iff the WHOLE item at `fn_kw_start` (a `fn` keyword's position
    in `stripped`, walked back past any `pub`/`async`/`unsafe`/`const`/
    `extern ".."` modifier — see `_item_start_before_fn`) is immediately
    preceded — only whitespace in between, which the STRIPPED text already
    reduces every comment/blank line to — by a CONTIGUOUS chain of
    attribute spans, at least one of which is `#[test]`/`#[<path>::test]`.
    Walking the whole chain (not just the nearest span) is what makes G5
    (a multi-line `#[cfg_attr(...)]` stacked ABOVE `#[test]`) resolve
    correctly.
    """
    pos = _item_start_before_fn(stripped, fn_kw_start)
    found_test = False
    spans_by_end = sorted(attr_spans, key=lambda s: s[1])
    while True:
        candidate = None
        for s, e, path in spans_by_end:
            if e <= pos and not stripped[e:pos].strip():
                if candidate is None or e > candidate[1]:
                    candidate = (s, e, path)
        if candidate is None:
            break
        s, e, path = candidate
        if _is_test_attr_path(path):
            found_test = True
        pos = s
    return found_test


# Round-4 audit item 3: the SAME lesson round-3b learned for `check_fn_
# desync` (:602) applies here — a fail-closed check with NO review/escape
# mechanism is a check no conforming file can ever satisfy once a genuine,
# lexically-unresolvable shape exists (a proc-macro-attribute-generated
# test, or any other totality-breaking shape the `FN_HEAD_RE` widening
# above doesn't close). Mirrors the SAME `// kernel-oracles: <check>
# reviewed: <reason>` idiom, with the same "a marker that no longer
# corresponds to a real discrepancy is itself a FAIL" symmetry — a marker
# never accumulates as unverifiable dead weight. Totality is a whole-file
# SCALAR count (not a single line), so the marker's cardinality — not its
# exact line — is what's checked: the file must carry EXACTLY `|delta|`
# reviewed markers, no more, no fewer, each a deliberate, individually
# reviewed acknowledgment (never a blanket one-marker-covers-all escape).
KERNEL_ORACLES_TEST_ATTR_MARKER_RE = re.compile(
    r"//\s*kernel-oracles:\s*test-attr reviewed:\s*(?P<reason>.+?)\s*$"
)


def check_test_attr_totality(
    stripped: str,
    attr_spans: list[tuple[int, int, str]],
    records: list,
    file_label: str,
    marker_view_lines: list[str] | None = None,
) -> None:
    """TOTALITY CROSS-CHECK (round-4, item 2), fail-closed: the count of
    `#[test]`/`#[<path>::test]`-shaped ATTRIBUTE TOKENS in the stripped
    text must equal the number of fns this gate classified `is_test=True`
    — if `_has_test_attr` (above) is correct, these are the SAME count by
    construction; a genuine mismatch (a test attribute orphaned — not
    immediately followed, across only whitespace/other attributes, by any
    `fn` at all — or two attributes stacked before one fn) means this
    file is UNCOMPUTABLE, not silently under- or over-counted — UNLESS the
    file carries exactly `|delta|` `// kernel-oracles: test-attr reviewed:
    <reason>` markers (round-4 audit item 3), each a deliberate sign-off on
    one lexically-unresolvable shape. `marker_view_lines` is `_strip_
    strings_only(source).splitlines()` (round-5 audit A3) — NOT raw
    lines — so a marker-shaped SUBSTRING sitting inside some OTHER string
    literal on the line cannot masquerade as a real reviewed comment.
    """
    attr_test_count = sum(1 for _s, _e, path in attr_spans if _is_test_attr_path(path))
    fn_test_count = sum(1 for r in records if r.is_test)
    delta = attr_test_count - fn_test_count
    marker_lines = [
        i + 1
        for i, line in enumerate(marker_view_lines or [])
        if KERNEL_ORACLES_TEST_ATTR_MARKER_RE.search(line)
    ]
    if delta == 0:
        if marker_lines:
            raise OracleError(
                f"{file_label}: stale `test-attr reviewed` marker(s) at line(s) "
                f"{', '.join(map(str, marker_lines))} (attribute/fn totality already balances "
                "exactly — remove the marker(s))"
            )
        return
    if len(marker_lines) != abs(delta):
        raise OracleError(
            f"{file_label}: test-attribute totality mismatch — {attr_test_count} `#[test]`-shaped "
            f"attribute(s) in the stripped text vs {fn_test_count} fn(s) this gate classified as "
            "tests. Totality here means exact PARITY between attribute-token count and classified-"
            "test-fn count — a genuine mismatch means a test attribute is not attached to exactly "
            "one fn (orphaned, stacked with another, or a proc-macro-attribute-generated shape "
            f"this lexical scanner cannot resolve). Found {len(marker_lines)} `// kernel-oracles: "
            f"test-attr reviewed: <reason>` marker(s); need exactly {abs(delta)} (one per "
            "unresolvable shape) to accept this file as reviewed rather than uncomputable."
        )


def find_fns(source: str, file_label: str) -> list[FnRecord]:
    """Every `fn <name>(...) { ... }` in `source`. `check_fn_desync` runs
    FIRST (fail-closed, once per file), then `check_test_attr_totality`
    (fail-closed, once per file) AFTER classification. Both fn-BOUNDARY
    detection (`FN_HEAD_RE`) and fn-BODY brace counting run entirely on
    the ONE stripped text (a `}` character sitting merely inside a string
    literal, e.g. `println!("brace }} }} literal")`, can no longer
    truncate a fn body early). `body`/`body_stripped` are sliced from the
    SAME [start, end) range out of `source` and the stripped text
    respectively — valid because stripping is length-preserving.
    """
    check_fn_desync(source, file_label)
    stripped_full = _strip_rust(source)
    attr_spans = _find_attribute_spans(stripped_full)

    records: list[FnRecord] = []
    for m in FN_HEAD_RE.finditer(stripped_full):
        name = m.group(1)
        is_test = _has_test_attr(stripped_full, attr_spans, m.start())
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
    check_test_attr_totality(
        stripped_full, attr_spans, records, file_label, _strip_strings_only(source).splitlines()
    )
    return records


# --------------------------------------------------------------------------- #
# KO-7 — unrun-is-RED
# --------------------------------------------------------------------------- #
# Round-4 (lead probe): DECLARED helpers, not DISCOVERED (item 1). Three
# rounds in a row found a NEW hole in "scan any fn's body for a shape that
# looks like a require-gate" — an unrelated `.expect(` or an uncalled
# closure's `panic!` still registers (F1, F2 — "prior item 3 STANDS"); a
# conforming gate written as `match env_var { Some(_) => panic!(..), None
# => .. }` instead of `if` does NOT register under the if-only regex (F3,
# a round-3 REGRESSION versus round 2); a shadowed import
# (`use std::env::var as getenv;`) correctly stays unregistered only by
# accident of the literal-call-syntax match, not by design. A regex/
# lexical scanner cannot establish "this fn IS a real, reviewed require-
# gate helper" as a syntactic fact — no shape-based heuristic closes the
# class, because ANY shape a real helper has, a decoy can imitate.
#
# So KO-7's "gated" predicate no longer asks "does this look like a
# gate?" — it asks "is this NAME in the reviewed registry?" `ci/kernel-
# oracle-helpers.txt` is a committed, human-reviewed list of `<file>::
# <fn_name>` pairs (comment lines `#`-prefixed). A skip is gated iff its
# window contains a call to a name FROM THAT LIST — nothing else, however
# panic-shaped, ever gates anything. Registering a helper is a reviewed
# PR diff, exactly like adding a citation or a `no-producer:` opt-out; the
# registry may only GROW by lines the shape-check below accepts (never a
# bare rename with no re-verification).
#
# The shape-check itself stays real (a registry entry is not a rubber
# stamp): each registered helper must have the CANONICAL shape — a real
# env-read (`std::env::var_os`/`var`/`option_env!`) of a `JAMMI_REQUIRE_*`
# name, via `if` OR `match` (both accepted — this is what closes F3), and
# the branch taken when the var IS SET (the `if`-block; or, for `match`,
# an arm whose pattern reads `Some`/`Ok`) must consist of EXACTLY ONE
# statement — a `panic!(...)`/`unreachable!(...)` invocation, nothing
# else (no `.expect(`, no closures, no additional statements) — or the
# gate is a REGISTRY FAIL naming the offending file/fn.
HELPERS_REGISTRY_PATH = REPO_ROOT / "ci" / "kernel-oracle-helpers.txt"

# Round-4 audit advisory: `option_env!` reads a value baked in AT COMPILE
# TIME (the macro is expanded by the compiler that BUILDS this test binary,
# not by the process that RUNS it) — a `JAMMI_REQUIRE_*` gate is a runtime
# enforcement switch (the pod lane exports it before `cargo test` runs, not
# before it was compiled), so `option_env!` can never observe it and a
# helper gated only that way silently never fires. Runtime reads
# (`std::env::var_os`/`var`) only.
ENV_READ_CALL_ALTERNATION = r"(?:std::env::var_os|env::var_os|std::env::var|env::var)"
# Round-4 audit advisory: the ORIGINAL `[^{]*?` wildcards (both before AND
# after the env-read call, up to the opening `{`) accepted ANY surrounding
# condition text — `... .is_some() && false { panic!(..) }` (a conjunct
# that makes the guard never actually fire) and `... .is_none() {
# panic!(..) }` (an INVERTED gate — panics when the flag is NOT set) both
# satisfied the old regex. The `if` condition must now be EXACTLY the
# env-read call followed by `.is_some()` or `.is_ok()` — nothing before it
# in the condition, nothing after it but the opening `{`.
IF_ENV_READ_RE = re.compile(
    rf'\bif\s+{ENV_READ_CALL_ALTERNATION}\s*\(\s*"(JAMMI_REQUIRE_[A-Z0-9_]*)"\s*\)'
    r"\s*\.\s*(?:is_some|is_ok)\s*\(\s*\)\s*\{"
)
MATCH_ENV_READ_RE = re.compile(
    rf'\bmatch\b[^{{]*?\b{ENV_READ_CALL_ALTERNATION}\s*\(\s*"(JAMMI_REQUIRE_[A-Z0-9_]*)"[^{{]*\{{'
)
# `return;` / brace-tail `return}`, plus `return Ok(...)`/`return Err(...)`,
# plus `process::exit(`/`std::process::exit(` (round-4 audit F15 — a
# #[test] fn that skips by TERMINATING THE PROCESS instead of returning is
# just as invisible to KO-7 without this) — a #[test] fn early-exiting via
# any of these shapes is just as much a silent skip as a bare `return;`.
# Only the START of the statement is matched (sufficient for textual
# ordering/windowing below). A bare, imported `exit(` (round-4 audit
# advisory) is handled separately by `BARE_EXIT_RE`, gated on the file
# actually importing it (`_file_imports_exit`) — an unqualified `exit(`
# with no import in scope is at least as likely a local/shadowed name as
# `std::process::exit`. A trailing `,` (round-5 audit NF-1 — `match
# cuda_device() { Some(d) => d, None => return, }`, `return` as a
# match-ARM tail expression, terminated by the arm's own comma rather than
# a `;`/`}`) is recognized alongside `;`/`}` — the RECOGNIZED shapes this
# regex covers, listed here and in the module doc above (round-5 audit
# NF-1): a bare `return` followed by `;`, `}`, `,`, `Ok(`, or `Err(`, or a
# `process::exit(`/`std::process::exit(` call. KO-7's totality claim is
# over THESE recognized shapes — a `return` shape not on this list (were
# one to exist) is not currently detectable; the module doc states this
# as a limitation, not "every possible skip idiom."
RETURN_SKIP_RE = re.compile(
    r"\breturn\b\s*(?:;|\}|,|Ok\s*\(|Err\s*\()|\b(?:std::)?process::exit\s*\("
)
BARE_EXIT_RE = re.compile(r"(?<![.:])\bexit\s*\(")
_EXIT_IMPORT_RE = re.compile(r"\buse\s+std::process::(?:exit\b|\{[^}]*\bexit\b[^}]*\})")


def _file_imports_exit(source_text: str) -> bool:
    return bool(_EXIT_IMPORT_RE.search(source_text))


# Round-4 audit advisory: a `return`/`process::exit` inside a CLOSURE
# (`v.iter().for_each(|x| { if .. { return; } })`) or a nested `fn` ITEM
# defined inside the test body (`fn helper() { .. return; .. }
# helper();`) does not skip the TEST — it is a different control-flow
# scope entirely. `body_stripped` starts at the test fn's own opening `{`
# (no head text before it), so any `FN_HEAD_RE` match found INSIDE it is
# necessarily a NESTED fn; any `|params| { .. }`/`|params| -> Ty { .. }`
# closure head found INSIDE it opens a nested scope the same way. Round-5
# audit advisory A1 widens this to an `async`/`async move` BLOCK
# (`let f = async move { return; }; drop(f);`) — its `return` returns from
# the generated future's own poll body, exactly as a closure's does, not
# from the enclosing #[test] fn.
_CLOSURE_HEAD_END_RE = re.compile(
    r"\|[^|{}]*\|\s*(?:->\s*[^{;]+?)?\s*\{$|\basync\s+(?:move\s+)?\{$"
)


def _nested_scope_open_braces(body_stripped: str) -> set[int]:
    opens: set[int] = set()
    for m in FN_HEAD_RE.finditer(body_stripped):
        brace = body_stripped.find("{", m.end())
        if brace != -1:
            opens.add(brace)
    for i, c in enumerate(body_stripped):
        if c != "{":
            continue
        if _CLOSURE_HEAD_END_RE.search(body_stripped[max(0, i - 200) : i + 1]):
            opens.add(i)
    return opens


def _top_level_skip_matches(body_stripped: str, extra_res: tuple[re.Pattern, ...] = ()) -> list[re.Match]:
    """Every `RETURN_SKIP_RE` (plus any `extra_res`, e.g. `BARE_EXIT_RE`)
    match that sits at the test fn's OWN body depth — not inside a nested
    closure or `fn` item (see module note above)."""
    opens = _nested_scope_open_braces(body_stripped)
    skips_by_pos: dict[int, re.Match] = {}
    for m in RETURN_SKIP_RE.finditer(body_stripped):
        skips_by_pos[m.start()] = m
    for cre in extra_res:
        for m in cre.finditer(body_stripped):
            skips_by_pos.setdefault(m.start(), m)
    stack = [False]
    result: list[re.Match] = []
    for i, c in enumerate(body_stripped):
        if i in skips_by_pos and not stack[-1]:
            result.append(skips_by_pos[i])
        if c == "{":
            stack.append(stack[-1] or i in opens)
        elif c == "}" and len(stack) > 1:
            stack.pop()
    return sorted(result, key=lambda m: m.start())


def _extract_call_and_rest(text: str) -> tuple[str, str]:
    """`text` starts with `name(`. Returns (call_text_incl_parens, rest_after)."""
    depth = 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[: i + 1], text[i + 1 :]
    return text, ""


def _is_exactly_one_panic_stmt(text: str) -> bool:
    """`text` (already the "branch taken when the var is set" body,
    braces stripped if present) is EXACTLY one `panic!(...)`/
    `unreachable!(...)` statement — no `.expect(`, no closures, nothing
    else before or after (an optional trailing `;` is allowed).
    """
    text = text.strip()
    if text.startswith("{"):
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    text = text[1:i].strip()
                    break
    m = re.match(r"(?:panic!|unreachable!)\s*\(", text)
    if not m:
        return False
    _call, rest = _extract_call_and_rest(text)
    return rest.strip().rstrip(";").strip() == ""


def _split_match_arms(match_body: str) -> list[tuple[str, str]]:
    """`match_body` is the text between a match's own `{`...`}` (braces
    NOT included). Returns `[(pattern, body), ...]`, splitting on
    TOP-LEVEL `,` (depth-balanced over `()[]{}` so a `,` inside a
    pattern's own tuple/struct destructure, or inside a body's own block,
    never splits an arm early).
    """
    arms_raw: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(match_body):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            arms_raw.append(match_body[start:i])
            start = i + 1
    if match_body[start:].strip():
        arms_raw.append(match_body[start:])
    result: list[tuple[str, str]] = []
    for arm in arms_raw:
        m = re.search(r"=>", arm)
        if not m:
            continue
        result.append((arm[: m.start()].strip(), arm[m.end() :].strip()))
    return result


def helper_shape_ok(fn_body_stripped: str) -> tuple[bool, str]:
    """The canonical require-gate shape (see module doc above): `if`-form
    or `match`-form, both accepted; whichever it is, the branch taken when
    the env var is set is EXACTLY one panic!/unreachable! statement.
    Returns (ok, reason-if-not).
    """
    for m in IF_ENV_READ_RE.finditer(fn_body_stripped):
        block, _, _ = _extract_balanced_block(fn_body_stripped, m.end() - 1)
        if _is_exactly_one_panic_stmt(block):
            return True, ""
    for m in MATCH_ENV_READ_RE.finditer(fn_body_stripped):
        block, _, _ = _extract_balanced_block(fn_body_stripped, m.end() - 1)
        inner = block[1:-1]
        for pattern, body in _split_match_arms(inner):
            # Round-4 audit advisory: a MATCH GUARD (`Some(v) if <cond> =>
            # panic!(..)`) makes the panic conditional on more than "the
            # env var is set" — a guard that can never be true (or is
            # merely narrower than the bare pattern) would otherwise let
            # this arm satisfy verification while never actually firing.
            # Reject any guarded arm outright; only a bare `Some(..)`/
            # `Ok(..)` pattern (no `if`) counts.
            if re.search(r"\bif\b", pattern):
                continue
            if re.search(r"\b(?:Some|Ok)\b", pattern) and _is_exactly_one_panic_stmt(body):
                return True, ""
    return (
        False,
        "no canonical if/match RUNTIME env-read (std::env::var_os/var of a JAMMI_REQUIRE_* name — "
        "option_env! is a COMPILE-TIME read, never accepted) whose taken-when-set branch is "
        "EXACTLY one panic!/unreachable! statement (no .expect(, no closures, no other "
        "statements, no match-arm guard)",
    )


def load_helper_registry(path: Path = HELPERS_REGISTRY_PATH) -> list[tuple[str, str]]:
    """`[(file_rel_path, fn_name), ...]` from `ci/kernel-oracle-helpers.txt`
    — `#`-prefixed and blank lines ignored. Fails closed on a malformed
    line (never silently skips one), and on an exact DUPLICATE `<file>::
    <fn_name>` entry (round-4 audit advisory — a repeated line is either a
    copy-paste mistake or dead weight nobody would notice is redundant;
    the registry is a reviewed list, and a duplicate was never itself
    reviewed as a second, distinct fact).
    """
    if not path.is_file():
        raise OracleError(f"helper registry not found: {path}")
    entries: list[tuple[str, str]] = []
    seen: dict[tuple[str, str], int] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if "::" not in stripped_line:
            raise OracleError(
                f"{path.name}:{line_no}: malformed line (expected `<file>::<fn_name>`): {line!r}"
            )
        file_part, fn_part = stripped_line.rsplit("::", 1)
        key = (file_part, fn_part)
        if key in seen:
            raise OracleError(
                f"{path.name}:{line_no}: duplicate registry entry {file_part}::{fn_part} "
                f"(already present at line {seen[key]} — remove the repeat)"
            )
        seen[key] = line_no
        entries.append(key)
    return entries


def verify_helper_registry(
    entries: list[tuple[str, str]], source_texts: dict[str, str]
) -> tuple[set[tuple[str, str]], list[str]]:
    """Returns `(verified, failures)` — `verified` is a set of `(file,
    fn_name)` PAIRS, not a flat name set (round-4 audit item 1: a flat name
    set lets an UNREGISTERED same-named fn in a DIFFERENT file "borrow"
    another file's review by name alone — `flash_op_oracles.rs` defining
    its own, never-reviewed `cuda_device()` gated silently because
    `cuda_device` was a registered NAME for `cuda_parity.rs`'s unrelated
    fn. Same-NAME, different-FILE helpers, e.g. both `cuda_parity.rs` and
    `flash_smoke.rs` each defining their own `cuda_device`, remain the
    ordinary case — each is verified, and gates, independently against its
    OWN file).

    Each entry's `file` must resolve among the scanned `source_texts`; the
    named `fn` must exist in it EXACTLY ONCE — two fns sharing that name in
    one file (this lexical scanner does not track `mod` boundaries, so
    "the real one" vs. a same-named decoy elsewhere in the file is
    undecidable) is a REGISTRY FAIL, never a best-effort pick of whichever
    candidate happens to pass; and that one candidate's body must pass
    `helper_shape_ok` — else a REGISTRY FAIL naming the file:line and why.
    """
    verified: set[tuple[str, str]] = set()
    failures: list[str] = []
    for file_rel, fn_name in entries:
        text = source_texts.get(file_rel)
        if text is None:
            failures.append(f"{file_rel}::{fn_name}: file not found among scanned files")
            continue
        candidates = [f for f in find_fns(text, file_rel) if f.name == fn_name]
        if not candidates:
            failures.append(f"{file_rel}::{fn_name}: fn not found in file")
            continue
        if len(candidates) > 1:
            lines = sorted(text.count("\n", 0, c.body_start_idx) + 1 for c in candidates)
            failures.append(
                f"{file_rel}::{fn_name}: {len(candidates)} fns named {fn_name!r} in this one "
                f"file (at lines {', '.join(map(str, lines))}) — which one this entry reviews is "
                "undecidable; rename or restructure so exactly one fn in the file has this name"
            )
            continue
        candidate = candidates[0]
        line_no = text.count("\n", 0, candidate.body_start_idx) + 1
        shape_ok, reason = helper_shape_ok(candidate.body_stripped)
        if not shape_ok:
            failures.append(f"{file_rel}:{line_no}::{fn_name}: {reason}")
            continue
        verified.add((file_rel, fn_name))
    return verified, failures


@dataclass(frozen=True)
class UngatedSkip:
    file: str
    fn_name: str
    line_no: int


def _is_bare_call(text: str, pos: int) -> bool:
    """True iff the identifier match starting at `pos` in `text` is called
    in the file's OWN local scope — not as a method (`self.name(`), a
    qualified path (`Other::name(`), or a `fn name(` DEFINITION (round-4
    audit item 4 — all three textually contain `name(` and would otherwise
    "gate" a skip without ever having called the file's OWN reviewed
    helper: a nested `fn cuda_device() {..}` DEFINITION inside the test
    body, never called, previously satisfied the bare regex; so did an
    explicitly-qualified `a::cuda_device()` reaching a DIFFERENT module's
    same-named decoy).
    """
    return not re.search(r"(?:\.\s*|::\s*|\bfn\s+)$", text[:pos])


def check_ko7(
    all_fns: list[FnRecord], verified: set[tuple[str, str]], source_texts: dict[str, str]
) -> list[UngatedSkip]:
    """Every top-level `return;`/`return}`/`return Ok(`/`return Err(`/
    `process::exit(`/gated bare `exit(` inside every `#[test]` fn's
    STRIPPED body (round-4 audit advisory: NOT one sitting inside a nested
    closure or `fn` item — see `_top_level_skip_matches`) must be textually
    dominated by a BARE call (see `_is_bare_call`) to a name registered FOR
    THIS FN'S OWN FILE (round-4 audit item 1 — `verified` is `(file,
    fn_name)`-scoped, never a flat cross-file name set) — PER SKIP: the
    dominance window for a given skip is `[end of the PREVIOUS skip (or fn
    start), this skip's start)`, so an early helper call gates only the
    skip(s) immediately downstream of it, never every later skip in the
    same fn unconditionally — a gated CUDA-device check followed by an
    UNRELATED, ungated `if !FLASH_COMPILED { return; }` further down the
    same fn still reds.
    """
    verified_by_file: dict[str, set[str]] = {}
    for file_rel, fn_name in verified:
        verified_by_file.setdefault(file_rel, set()).add(fn_name)
    call_res: dict[str, re.Pattern] = {}
    findings: list[UngatedSkip] = []
    for fn in all_fns:
        if not fn.is_test:
            continue
        extra_res: tuple[re.Pattern, ...] = ()
        if _file_imports_exit(source_texts[fn.file]):
            extra_res = (BARE_EXIT_RE,)
        skip_matches = _top_level_skip_matches(fn.body_stripped, extra_res)
        if not skip_matches:
            continue
        local_names = verified_by_file.get(fn.file, set())
        helper_positions = []
        for name in local_names:
            cre = call_res.setdefault(name, re.compile(rf"\b{re.escape(name)}\s*\("))
            for m in cre.finditer(fn.body_stripped):
                if _is_bare_call(fn.body_stripped, m.start()):
                    helper_positions.append(m.start())
        helper_positions.sort()
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


# Round-4 audit item 3. `assert_relative_eq!`/`assert_abs_diff_eq!` (the
# `approx` crate — H1) added: these take `epsilon =`/`max_relative =`
# NAMED parameters instead of a literal comparison operator at all, so
# they get their own bound-extraction rule below rather than forcing them
# through the operator-adjacency shape the other macros use.
ASSERT_MACRO_RE = re.compile(
    r"\b(assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|"
    r"assert_relative_eq|assert_abs_diff_eq)!\s*\("
)
_APPROX_EQ_MACRO_NAMES = frozenset({"assert_relative_eq", "assert_abs_diff_eq"})
# Round-4 audit fix (H3 regression): round 3's space-padded ` < `/` > `
# requirement (meant to exclude a generic bracket like `Vec<f32>`) also
# excluded a real, unpadded comparison (`if d<TOL { ... }`, legal rustfmt
# output inside some contexts) — dropped. This is an honest, KNOWN cost,
# not a wash: `let v: Vec<TOL> = ..;` DOES now read COVERED (TOL is
# directly touching the bracket's own `<`/`>`, same adjacency shape as a
# real comparison operand — a lexical scanner cannot tell "generic
# bracket" from "comparison" apart without a type grammar). See the
# module docstring's KO-2 "known, out-of-scope class limitations"
# paragraph — a false COVERED here is a silent miss in the safe
# direction (KO-2 only ever flags MISSING coverage), traded deliberately
# against H3's real comparisons going undetected, never both closed at
# once by an adjacency-only rule.
_COMPARISON_OP_RE_KO2 = re.compile(r"<=|>=|==|<|>")


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


def _first_top_level_arg(args_with_parens: str) -> str:
    """`args_with_parens` includes the outer parens, e.g. `(a, b, "msg")`.
    Returns the FIRST top-level-comma-delimited argument's text (parens
    excluded) — for an assert!-family call, this is the CONDITION; round-4
    audit fix (H7): a bound mentioned only in a later `format!` MESSAGE
    argument (`assert!(d < 1e-3, "tol was {}", TOL)`) must never count as
    coverage — message args are never even looked at.
    """
    inner = args_with_parens[1:-1]
    depth = 0
    for i, c in enumerate(inner):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            return inner[:i]
    return inner


def _adjacent_to_comparison(text: str, bound_name: str) -> bool:
    """`bound_name` is DIRECTLY adjacent (only whitespace in between) to a
    comparison operator's OWN operand position — the bound must be one of
    the operator's two SIDES, not merely present somewhere nearby (`let f
    = compute(TOL) as u32 > 0;` — TOL feeds a value that is later
    compared, but is never an operand of that comparison — must MISS).
    """
    name_before_op_re = re.compile(rf"\b{re.escape(bound_name)}\b\s*(?:<=|>=|==|<|>)")
    op_before_name_re = re.compile(rf"(?:<=|>=|==|<|>)\s*{re.escape(bound_name)}\b")
    return bool(name_before_op_re.search(text) or op_before_name_re.search(text))


def _bound_in_assertion_context(body_stripped: str, bound_name: str) -> bool:
    """`bound_name` (an identifier) is covered EITHER by an `assert!`-
    family macro call — the `approx`-style epsilon macros via their own
    `epsilon =`/`max_relative =` operand (H1); every other assert!-family
    macro via mere PRESENCE in the FIRST top-level argument (the
    condition) ONLY — message args never count (H7) — OR, with no
    enclosing assert! call at all, by DIRECT adjacency to a comparison
    operator anywhere in the fn body (the bare-`if`/boolean-return-
    expression shape, e.g. `within_bound`).
    """
    name_re = re.compile(rf"\b{re.escape(bound_name)}\b")

    for m in ASSERT_MACRO_RE.finditer(body_stripped):
        macro_name = m.group(1)
        args = _extract_paren_balanced(body_stripped, m.end() - 1)
        if macro_name in _APPROX_EQ_MACRO_NAMES:
            if re.search(rf"\b(?:epsilon|max_relative)\s*=\s*{re.escape(bound_name)}\b", args):
                return True
            continue
        first_arg = _first_top_level_arg(args)
        if name_re.search(first_arg):
            return True

    if _adjacent_to_comparison(body_stripped, bound_name):
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
    # Round-4 audit advisory: RECURSIVE under the two roots (not additional
    # roots — `crates/jammi-encoders/src/context/*.rs` is a real,
    # previously-silently-unscanned subdirectory of the already-in-scope
    # `crates/jammi-encoders/src/`; scope.py's 19 files elsewhere are a
    # DIFFERENT crate entirely and stay out of scope, documented, not
    # pulled in).
    texts: dict[str, str] = {}
    for path in sorted(KERNELS_TESTS_DIR.rglob("*.rs")):
        rel = str(path.relative_to(REPO_ROOT))
        texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
    for path in sorted(ENCODERS_SRC_DIR.rglob("*.rs")):
        rel = str(path.relative_to(REPO_ROOT))
        texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def run_gate(
    source_texts: dict[str, str], shipped_ops: set[str], registry_entries: list[tuple[str, str]]
) -> tuple[
    list[UngatedSkip],
    list[Ko2Finding],
    list[OracleCellMarker],
    dict[str, list[OracleCellMarker]],
    dict[str, list[OracleCellMarker]],
    dict[str, str],
    list[str],
    list[str],
]:
    """Pure orchestration over `{file: text}` + the SHIPPED op set + the
    helper REGISTRY entries — the self-test seam. Returns (ko7_findings,
    ko2_findings, ko5_findings, covered, declared_uncontrolled, pending,
    reconciliation_failures, registry_failures).
    """
    all_fns: list[FnRecord] = []
    all_markers: list[OracleCellMarker] = []
    for file_label, text in source_texts.items():
        all_fns.extend(find_fns(text, file_label))
        all_markers.extend(parse_markers(text, file_label))

    verified_helpers, registry_failures = verify_helper_registry(registry_entries, source_texts)
    ko7 = check_ko7(all_fns, verified_helpers, source_texts)
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

    return ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures, registry_failures


def main() -> int:
    try:
        source_texts = scan_files()
        shipped_ops = load_shipped_ops()
        registry_entries = load_helper_registry()
        ko7, ko2, ko5, covered, declared_uncontrolled, pending, recon_failures, registry_failures = run_gate(
            source_texts, shipped_ops, registry_entries
        )
    except OracleError as exc:
        print(f"kernel-oracles: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    print_reconciliation(shipped_ops, covered, STRUCTURALLY_EXCLUDED_OPS, declared_uncontrolled, pending)

    print(f"\nHelper registry: {len(registry_entries)} entr(y/ies), {len(registry_failures)} registry FAIL(s).")

    ko7_by_file: dict[str, list[UngatedSkip]] = {}
    for f in ko7:
        ko7_by_file.setdefault(f.file, []).append(f)
    print("\nKO-7 (unrun-is-RED) per scanned file:")
    for file_label, text in source_texts.items():
        fns_in_file = find_fns(text, file_label)
        extra_res: tuple[re.Pattern, ...] = (BARE_EXIT_RE,) if _file_imports_exit(text) else ()
        n_skips = sum(
            len(_top_level_skip_matches(fn.body_stripped, extra_res))
            for fn in fns_in_file
            if fn.is_test
        )
        n_ungated = len(ko7_by_file.get(file_label, []))
        print(f"    {file_label}: {n_skips} runtime skip(s), {n_ungated} ungated")

    failures: list[str] = list(recon_failures) + [f"REGISTRY: {msg}" for msg in registry_failures]
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
