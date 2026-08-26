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
body (found anywhere in the scanned tree, brace-balanced) must contain EVERY
identifier listed in `bounds` as a token — a marker cannot claim a bound its own
control never touches. `control=none:<reason>` opts a cell out of KO-2 entirely
(reported, never silently passed).

**KO-5 (off-sample bounds).** When BOTH `derived-on` and `asserted-on` are real
seed lists (neither is the literal `none`), their intersection must be EMPTY — a
bound calibrated on seed 42 and then "validated" by asserting against seed 42
again is circular, not evidence (the pressure-test design rule in the guide:
a bound's noise floor must be measured independently of where it is enforced).

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
  - Any op claimed by more than one of COVERED / STRUCTURALLY_EXCLUDED is a
    non-zero exit (PENDING is computed as the complement, so it cannot
    conflict with the other two by construction).
  - A parse failure (missing scan root, uncomputable op domain) is a non-zero
    exit naming what could not be resolved.

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
# re-implemented in this file (one definition per rule).
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
# brace-balanced fn-body extraction — shared by KO-2's control-fn lookup and
# KO-7's #[test]-fn / require-gate-helper scan.
# --------------------------------------------------------------------------- #
FN_HEAD_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(<]")
_ATTR_LINE_RE = re.compile(r"^\s*#!?\[")


def _extract_fn_body(source: str, fn_kw_start: int) -> tuple[str, int, int]:
    """Returns (body_text_including_braces, brace_open_idx, brace_close_idx)."""
    brace_start = source.find("{", fn_kw_start)
    if brace_start == -1:
        return "", -1, -1
    depth = 0
    for i in range(brace_start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start : i + 1], brace_start, i
    return source[brace_start:], brace_start, len(source) - 1


@dataclass(frozen=True)
class FnRecord:
    name: str
    file: str
    body: str
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
    """Every `fn <name>(...) { ... }` in `source`, brace-balanced, tagged
    with whether its contiguous attribute block above the `fn` line contains
    any attribute whose text mentions "test" (covers `#[test]`,
    `#[tokio::test]`, ... — same convention as `check_doc_numbers_have_
    producers.py`'s `build_test_fn_index`).
    """
    lines = source.splitlines(keepends=True)
    # Precompute the character offset each line starts at, so a regex match
    # position in `source` maps back to a line index.
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def line_idx_of(pos: int) -> int:
        # binary-search-free linear scan is fine at this file size; kept
        # simple over premature optimisation.
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
    for m in FN_HEAD_RE.finditer(source):
        name = m.group(1)
        fn_line_idx = line_idx_of(m.start())
        is_test = _has_test_attr(plain_lines, fn_line_idx)
        body, body_start, _ = _extract_fn_body(source, m.end())
        if not body:
            continue
        records.append(FnRecord(name=name, file=file_label, body=body, body_start_idx=body_start, is_test=is_test))
    return records


# --------------------------------------------------------------------------- #
# KO-7 — unrun-is-RED
# --------------------------------------------------------------------------- #
REQUIRE_ENV_RE = re.compile(r"\bJAMMI_REQUIRE_[A-Z0-9_]*\b")
PANIC_CALL_RE = re.compile(r"\bpanic!\s*\(")
RETURN_SKIP_RE = re.compile(r"\breturn\b\s*[;}]")


def find_require_gate_helpers(all_fns: list[FnRecord]) -> set[str]:
    """A fn is a REGISTERED require-gate helper iff its body reads a
    `JAMMI_REQUIRE_*` env var AND calls `panic!` — discovered by shape,
    never a hardcoded name list (see module doc).
    """
    helpers: set[str] = set()
    for fn in all_fns:
        if REQUIRE_ENV_RE.search(fn.body) and PANIC_CALL_RE.search(fn.body):
            helpers.add(fn.name)
    return helpers


@dataclass(frozen=True)
class UngatedSkip:
    file: str
    fn_name: str
    line_no: int


def _line_no_within(body: str, pos: int, body_start_line: int) -> int:
    return body_start_line + body.count("\n", 0, pos)


def check_ko7(all_fns: list[FnRecord], helper_names: set[str], source_texts: dict[str, str]) -> list[UngatedSkip]:
    """Every `return;`/`return}` inside every `#[test]` fn body must be
    textually dominated (an earlier position in the SAME fn body) by a call
    to a name in `helper_names`. `source_texts` is `{file_label: full_text}`
    — used only to translate a body-relative offset into a 1-indexed file
    line number for reporting.
    """
    helper_call_res = {name: re.compile(rf"\b{re.escape(name)}\s*\(") for name in helper_names}
    findings: list[UngatedSkip] = []
    for fn in all_fns:
        if not fn.is_test:
            continue
        skip_positions = [m.start() for m in RETURN_SKIP_RE.finditer(fn.body)]
        if not skip_positions:
            continue
        helper_positions: list[int] = []
        for name, cre in helper_call_res.items():
            helper_positions.extend(m.start() for m in cre.finditer(fn.body))
        for pos in skip_positions:
            if not any(hp < pos for hp in helper_positions):
                # translate `pos` (offset within fn.body) back to a file line
                full = source_texts[fn.file]
                abs_pos = fn.body_start_idx + pos
                line_no = full.count("\n", 0, abs_pos) + 1
                findings.append(UngatedSkip(file=fn.file, fn_name=fn.name, line_no=line_no))
    return findings


# --------------------------------------------------------------------------- #
# KO-2 — bound coverage parity (marker-scoped)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ko2Finding:
    marker: OracleCellMarker
    missing_bounds: tuple[str, ...] = ()
    control_not_found: bool = False


def check_ko2(markers: list[OracleCellMarker], all_fns: list[FnRecord]) -> list[Ko2Finding]:
    by_name: dict[str, list[FnRecord]] = {}
    for fn in all_fns:
        by_name.setdefault(fn.name, []).append(fn)

    findings: list[Ko2Finding] = []
    for marker in markers:
        if marker.is_control_none:
            continue
        candidates = by_name.get(marker.control)
        if not candidates:
            findings.append(Ko2Finding(marker=marker, control_not_found=True))
            continue
        combined_body = "\n".join(c.body for c in candidates)
        missing = tuple(
            b for b in marker.bounds if not re.search(rf"\b{re.escape(b)}\b", combined_body)
        )
        if missing:
            findings.append(Ko2Finding(marker=marker, missing_bounds=missing))
    return findings


# --------------------------------------------------------------------------- #
# KO-5 — off-sample bounds (marker-scoped)
# --------------------------------------------------------------------------- #
def check_ko5(markers: list[OracleCellMarker]) -> list[OracleCellMarker]:
    """Markers whose `derived-on` and `asserted-on` seed sets are BOTH real
    lists (neither the literal `none`) and intersect — circular calibration.
    """
    findings: list[OracleCellMarker] = []
    for marker in markers:
        if marker.derived_on is None or marker.asserted_on is None:
            continue
        if set(marker.derived_on) & set(marker.asserted_on):
            findings.append(marker)
    return findings


# --------------------------------------------------------------------------- #
# reconciliation — op domain × marker coverage
# --------------------------------------------------------------------------- #
# Reviewed, in-script. Empty in v1 — no op has been reviewed off the op
# domain yet; kept as a dict (not omitted) so a future round has somewhere
# to add a reviewed exclusion without inventing a second structure.
STRUCTURALLY_EXCLUDED_OPS: dict[str, str] = {}


def reconcile_ops(shipped: set[str], covered: dict[str, list[OracleCellMarker]], excluded: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Returns (pending_with_reason, failures). PENDING is COMPUTED (the
    complement of COVERED ∪ EXCLUDED within SHIPPED), never hand-curated —
    v1 has no markers, so every un-excluded op is honestly PENDING; a
    reviewer only ever ADDS to `STRUCTURALLY_EXCLUDED_OPS` or lands a real
    marker, never edits a PENDING dict directly.
    """
    failures: list[str] = []

    for op in covered:
        if op not in shipped:
            failures.append(f"COVERED marker references unknown op `{op}` — not in the SHIPPED admission-key set")
    for op in excluded:
        if op not in shipped:
            failures.append(f"STRUCTURALLY_EXCLUDED_OPS entry `{op}` references unknown op — not in the SHIPPED admission-key set")

    overlap = set(covered) & set(excluded)
    for op in sorted(overlap):
        failures.append(f"op `{op}` is claimed both COVERED and STRUCTURALLY_EXCLUDED — remove one")

    pending_reason = (
        "shipped admission key with no `oracle-cell` marker yet — v1 scope excludes every "
        "numerics-owned file (crates/jammi-kernels, crates/jammi-encoders); tracked debt, "
        "see docs/maintainer/cuda-kernel-guide.md §3's kernel-acceptance-oracle standard."
    )
    pending = {op: pending_reason for op in sorted(shipped - set(covered) - set(excluded))}
    return pending, failures


def print_reconciliation(shipped: set[str], covered: dict[str, list[OracleCellMarker]], excluded: dict[str, str], pending: dict[str, str]) -> None:
    print("kernel-oracle op reconciliation:")
    for op in sorted(shipped):
        if op in covered:
            legs = ", ".join(f"{m.file}:{m.line_no}" for m in covered[op])
            print(f"    COVERED             {op}  <- {legs}")
        elif op in excluded:
            print(f"    STRUCTURALLY_EXCL   {op}  — {excluded[op]}")
        elif op in pending:
            print(f"    PENDING             {op}")
        else:
            print(f"    !!!! UNACCOUNTED !!!! {op}")
    print(
        f"\nSummary: {len(covered)} COVERED, {len(excluded)} STRUCTURALLY_EXCLUDED, "
        f"{len(pending)} PENDING out of {len(shipped)} SHIPPED admission keys."
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


def run_gate(source_texts: dict[str, str], shipped_ops: set[str]) -> tuple[list[UngatedSkip], list[Ko2Finding], list[OracleCellMarker], dict[str, list[OracleCellMarker]], dict[str, str], list[str]]:
    """Pure orchestration over `{file: text}` + the SHIPPED op set — the
    self-test seam. Returns (ko7_findings, ko2_findings, ko5_findings,
    covered, pending, reconciliation_failures).
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

    covered: dict[str, list[OracleCellMarker]] = {}
    for marker in all_markers:
        covered.setdefault(marker.op, []).append(marker)

    pending, recon_failures = reconcile_ops(shipped_ops, covered, STRUCTURALLY_EXCLUDED_OPS)

    return ko7, ko2, ko5, covered, pending, recon_failures


def main() -> int:
    try:
        source_texts = scan_files()
        shipped_ops = load_shipped_ops()
        ko7, ko2, ko5, covered, pending, recon_failures = run_gate(source_texts, shipped_ops)
    except OracleError as exc:
        print(f"kernel-oracles: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    print_reconciliation(shipped_ops, covered, STRUCTURALLY_EXCLUDED_OPS, pending)

    ko7_by_file: dict[str, list[UngatedSkip]] = {}
    for f in ko7:
        ko7_by_file.setdefault(f.file, []).append(f)
    print("\nKO-7 (unrun-is-RED) per scanned file:")
    for file_label, text in source_texts.items():
        n_skips = sum(1 for fn in find_fns(text, file_label) if fn.is_test for _ in RETURN_SKIP_RE.finditer(fn.body))
        n_ungated = len(ko7_by_file.get(file_label, []))
        print(f"    {file_label}: {n_skips} runtime skip(s), {n_ungated} ungated")

    failures: list[str] = list(recon_failures)
    for f in ko7:
        failures.append(f"KO-7: {f.file}:{f.line_no} — ungated runtime skip in `#[test] fn {f.fn_name}` (no registered require-gate helper called before this return)")
    for f in ko2:
        if f.control_not_found:
            failures.append(f"KO-2: {f.marker.file}:{f.marker.line_no} — oracle-cell control fn `{f.marker.control}` not found in any scanned file")
        else:
            failures.append(f"KO-2: {f.marker.file}:{f.marker.line_no} — control fn `{f.marker.control}` does not assert against bound(s) {', '.join(f.missing_bounds)}")
    for f in ko5:
        overlap = sorted(set(f.derived_on or ()) & set(f.asserted_on or ()))
        failures.append(f"KO-5: {f.file}:{f.line_no} — derived-on/asserted-on share seed(s) {', '.join(overlap)} (circular calibration)")

    if failures:
        print("\nkernel-oracles: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\nkernel-oracles: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        "\nkernel-oracles: PASS — every scanned #[test] runtime skip is gated (KO-7); "
        "every marked oracle-cell's bounds are covered by its control (KO-2) and "
        "derived/asserted on disjoint seeds (KO-5); reconciliation is fully accounted "
        "(v1: 0 markers, every SHIPPED op honestly PENDING)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
