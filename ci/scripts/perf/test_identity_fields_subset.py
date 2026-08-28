#!/usr/bin/env python3
"""Unification contract C4.2 — Python COMPARISON tuple ⊆ Rust K7-completeness
const, for both jammi-vs-torch comparators this repo carries.

WHAT THIS CHECKS: `ci/scripts/perf/ab_merge.py`'s (re-exported from
`ci/scripts/perf/identity_fields.py` as of PR #381) `FINETUNE_IDENTITY_FIELDS`
and `ci/scripts/perf/compare_grad_oracle.py`'s `RUN_IDENTITY_FIELDS` are the
two COMPARISON tuples a jammi-vs-torch leg-premise check actually gates on
(contract C4.1: UNCHANGED BY THIS UNIT — growing either from WITHIN this
unit's own phase-1 work would invalidate every existing merge, since torch
never emits the Rust-only K7-completeness additions; it does NOT forbid
independent upstream work with its own reason to grow a tuple — PR #381
added `max_grad_norm`/`attention_arm`/`warmup` to `FINETUNE_IDENTITY_FIELDS`
for its own device-clip feature, landed on `main` before this unit merged,
and contract C3.4 explicitly names the rebase: "whichever of #381 and this
phase lands second rebases and adds the entry" — this merge is that rebase).
`crates/jammi-bench/src/report.rs`'s `FinetuneStepTier::IDENTITY_FIELDS` and
`crates/jammi-bench/src/grad_oracle.rs`'s `GradOracleReport::IDENTITY_FIELDS`
are the corresponding Rust consts (contract C3.1/C3.2) — a STRICT SUPERSET
of the Python tuple, adding provenance/dispatch facts the comparison
deliberately omits. This suite asserts, mechanically:

  1. Every entry of `FINETUNE_IDENTITY_FIELDS` is named in
     `FinetuneStepTier::IDENTITY_FIELDS`.
  2. Every entry of `RUN_IDENTITY_FIELDS` is named in
     `GradOracleReport::IDENTITY_FIELDS`.
  3. `FINETUNE_IDENTITY_FIELDS` has EXACTLY 18 entries (14 from this unit's
     own phase 1 + `max_grad_norm`/`attention_arm`/`warmup` from PR #381,
     merged onto this branch, + `row_lengths` from the M1b B3-padded-batch
     transport — `crates/jammi-bench/src/report.rs`'s own
     `FinetuneStepTier::IDENTITY_FIELDS` doc names this file's growth to 18
     as its expected companion docs-ci change) and `RUN_IDENTITY_FIELDS`
     has EXACTLY 11 — the "unchanged BY THIS UNIT" half of C4.1 is a
     NUMBER this suite pins, not a promise left to prose (a silent addition
     from WITHIN this unit that happened to still satisfy the subset check
     would pass (1)/(2) above but fail this count; the count itself is
     bumped only in lockstep with a real, externally-landed identity
     field, as this merge does).

HOW: the two Python tuples are IMPORTED directly (this is exactly what a
`main()` caller of either module reads, never a re-parsed copy of the
literal). The two Rust consts are extracted with a REGEX over the tracked
`.rs` source — never `rustc`/`cargo` (this checker's job is verifying the
COMMITTED SOURCE a compiled binary would be built from, not compiling
anything itself, so it stays usable in the plain-shallow-checkout `guard`
matrix leg alongside `check_citations.py` — no network, no build). A
missing const, or a `.rs` file that no longer exists, is a FAIL-CLOSED
`SystemExit`, never a silent skip (RED at base: `IDENTITY_FIELDS` does not
exist there at all).

Stdlib-only (`unittest`), same footing every other `ci/scripts/perf/test_*.py`
in this directory takes.

Run: `python3 ci/scripts/perf/test_identity_fields_subset.py`
"""

from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_merge  # noqa: E402
import compare_grad_oracle  # noqa: E402
import identity_fields  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT_RS = os.path.join(REPO_ROOT, "crates", "jammi-bench", "src", "report.rs")
GRAD_ORACLE_RS = os.path.join(REPO_ROOT, "crates", "jammi-bench", "src", "grad_oracle.rs")

# Matches `pub const IDENTITY_FIELDS: &'static [(&'static str, <..>Nullable)]
# = &[` — `<..>` covers both `report.rs`'s bare `Nullable` and
# `grad_oracle.rs`'s `crate::report::Nullable` spelling — up to the closing
# `];` at the SAME 4-space indent the array body is written at (both files'
# `IDENTITY_FIELDS` const bodies contain no nested `[`, so the first such
# closer IS the array's own close).
_IDENTITY_FIELDS_BLOCK_RE = re.compile(
    r"pub const IDENTITY_FIELDS:\s*&'static \[\(&'static str,\s*"
    r"[\w:]*Nullable\)\]\s*=\s*&\[(.*?)\n    \];",
    re.DOTALL,
)
# `("field_name",` — deliberately NOT anchored to a specific `Nullable::..`
# spelling on the right (this checker only needs the field NAME half of
# each tuple; the nullability half is `check_cuda_run_artifacts.py` rule
# (g)'s job, not this Python-⊆-Rust suite's).
_FIELD_NAME_RE = re.compile(r'\(\s*"([A-Za-z0-9_]+)"\s*,')


def _extract_rust_identity_fields(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise SystemExit(f"FAIL-CLOSED: {path} does not exist")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    match = _IDENTITY_FIELDS_BLOCK_RE.search(text)
    if match is None:
        raise SystemExit(
            f"FAIL-CLOSED: no `pub const IDENTITY_FIELDS: &'static [(&'static str, "
            f"..Nullable)] = &[ .. ];` block found in {path} — either the const was "
            f"never added (RED at base) or this regex needs updating for a genuine "
            f"reformat"
        )
    fields = _FIELD_NAME_RE.findall(match.group(1))
    if not fields:
        raise SystemExit(f"FAIL-CLOSED: IDENTITY_FIELDS block in {path} matched but named zero fields")
    return fields


# `EncodeStepTier` and `FinetuneStepTier` BOTH declare a const literally
# named `IDENTITY_FIELDS` in the SAME file (`report.rs`) — `_IDENTITY_FIELDS_
# BLOCK_RE.search(text)` alone would always find `FinetuneStepTier`'s block
# first (it sits earlier in the file) and never `EncodeStepTier`'s own.
# `_scoped_to_impl` narrows the search text to everything from a given
# struct's OWN `impl <Struct> {` marker onward, so the SAME "first matching
# block" regex then unambiguously finds THAT struct's const — never the
# unscoped file-wide first match. Mirrors `check_cuda_run_artifacts.py`'s own
# restructure of `build_identity_tuples()` (unit-62 E6 gate-restructure
# commit) — the SAME struct-scoping idea, applied independently here since
# this checker never imports that gate module.
_PROVENANCE_FIELDS_BLOCK_RE = re.compile(
    r"pub const PROVENANCE_FIELDS:\s*&'static \[\(&'static str,\s*"
    r"[\w:]*Nullable\)\]\s*=\s*&\[(.*?)\n    \];",
    re.DOTALL,
)


def _scoped_to_impl(text: str, struct: str) -> str:
    anchor = f"impl {struct} {{"
    idx = text.find(anchor)
    if idx == -1:
        raise SystemExit(f"FAIL-CLOSED: no `{anchor}` block found — cannot scope extraction to struct {struct!r}")
    return text[idx:]


def _extract_rust_fields_block(path: str, block_re: re.Pattern, struct: str) -> list[str]:
    if not os.path.isfile(path):
        raise SystemExit(f"FAIL-CLOSED: {path} does not exist")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    scoped = _scoped_to_impl(text, struct)
    match = block_re.search(scoped)
    if match is None:
        raise SystemExit(
            f"FAIL-CLOSED: no matching const block found in {path}, scoped to `impl {struct} {{` — "
            f"either the const was never added (RED at base) or this regex needs updating for a "
            f"genuine reformat"
        )
    fields = _FIELD_NAME_RE.findall(match.group(1))
    if not fields:
        raise SystemExit(f"FAIL-CLOSED: matched const block in {path} (struct {struct!r}) named zero fields")
    return fields


class FinetuneStepIdentityFieldsSubsetTests(unittest.TestCase):
    def setUp(self):
        self.rust_fields = set(_extract_rust_identity_fields(REPORT_RS))

    def test_finetune_identity_fields_has_exactly_18_entries(self):
        self.assertEqual(
            len(ab_merge.FINETUNE_IDENTITY_FIELDS),
            18,
            "ab_merge.py::FINETUNE_IDENTITY_FIELDS must have EXACTLY 18 entries — 14 "
            "from unit 61 phase 1 + max_grad_norm/attention_arm/warmup from PR #381 "
            "+ row_lengths from the M1b B3-padded-batch transport (contract C3.4's "
            "named rebase, then this row_lengths fold-in). A count OTHER than 18 "
            "means either this unit grew the tuple itself (forbidden, contract C4.1) "
            "or an upstream identity set changed shape after this fold-in was "
            "written — either way, re-derive this number from source, never bump it "
            "to make the test pass.",
        )
        # The tuple itself must not carry a duplicate — a dup would let the
        # subset check below pass trivially without covering 18 distinct
        # comparison fields.
        self.assertEqual(
            len(set(ab_merge.FINETUNE_IDENTITY_FIELDS)),
            18,
            "FINETUNE_IDENTITY_FIELDS contains a duplicate entry",
        )

    def test_finetune_identity_fields_is_a_subset_of_the_rust_const(self):
        missing = set(ab_merge.FINETUNE_IDENTITY_FIELDS) - self.rust_fields
        self.assertFalse(
            missing,
            f"ab_merge.py::FINETUNE_IDENTITY_FIELDS names field(s) NOT present in "
            f"FinetuneStepTier::IDENTITY_FIELDS ({REPORT_RS}): {sorted(missing)} — "
            f"every Python comparison field must be a K7-completeness identity field "
            f"on the Rust side too",
        )

    def test_the_rust_const_is_a_strict_superset(self):
        # Not a hard requirement of C4.1, but documents the shape this
        # suite expects: the Rust const carries the 5 K7-completeness
        # additions beyond the 14-field comparison tuple.
        extra = self.rust_fields - set(ab_merge.FINETUNE_IDENTITY_FIELDS)
        self.assertGreaterEqual(
            len(extra),
            1,
            "FinetuneStepTier::IDENTITY_FIELDS should carry at least one K7-"
            "completeness field beyond ab_merge.py's own comparison tuple "
            "(device_name / kernels_disabled_requested / kernels_disabled_fired / "
            "flash_compiled / build_features) — if this ever regresses to exactly "
            "the Python tuple, the two consts collapsed into one and the 'strict "
            "superset' framing in report.rs's own doc is no longer true",
        )


class GradOracleIdentityFieldsSubsetTests(unittest.TestCase):
    def setUp(self):
        self.rust_fields = set(_extract_rust_identity_fields(GRAD_ORACLE_RS))

    def test_run_identity_fields_has_exactly_11_entries(self):
        self.assertEqual(
            len(compare_grad_oracle.RUN_IDENTITY_FIELDS),
            11,
            "compare_grad_oracle.py::RUN_IDENTITY_FIELDS must stay UNCHANGED at 11 "
            "entries (contract C4.1) — a silent growth here invalidates every "
            "existing comparison",
        )
        self.assertEqual(
            len(set(compare_grad_oracle.RUN_IDENTITY_FIELDS)),
            11,
            "RUN_IDENTITY_FIELDS contains a duplicate entry",
        )

    def test_run_identity_fields_is_a_subset_of_the_rust_const(self):
        missing = set(compare_grad_oracle.RUN_IDENTITY_FIELDS) - self.rust_fields
        self.assertFalse(
            missing,
            f"compare_grad_oracle.py::RUN_IDENTITY_FIELDS names field(s) NOT present "
            f"in GradOracleReport::IDENTITY_FIELDS ({GRAD_ORACLE_RS}): "
            f"{sorted(missing)}",
        )

    def test_the_rust_const_is_a_strict_superset(self):
        extra = self.rust_fields - set(compare_grad_oracle.RUN_IDENTITY_FIELDS)
        self.assertGreaterEqual(
            len(extra),
            1,
            "GradOracleReport::IDENTITY_FIELDS should carry at least one K7-"
            "completeness field beyond compare_grad_oracle.py's own comparison "
            "tuple (lora_init / device_name)",
        )


class EncodeStepIdentityFieldsSubsetTests(unittest.TestCase):
    """Unit-62 E6 mirror: `identity_fields.ENCODE_IDENTITY_FIELDS` (Python)
    against `EncodeStepTier::IDENTITY_FIELDS`/`::PROVENANCE_FIELDS` (Rust,
    `report.rs`). UNLIKE `FinetuneStepIdentityFieldsSubsetTests` above, this
    is an EQUALITY check, not a subset check — `EncodeStepTier` keeps its
    provenance fields in an entirely DISJOINT const (never folded into
    `IDENTITY_FIELDS`, unlike `FinetuneStepTier`'s own superset-folding
    shape), so there is no larger Rust identity const for the Python tuple
    to be a strict subset of — the two sets must be the SAME set. A drift on
    EITHER side (a name added/removed from either the Python tuple or the
    Rust const, or a field crossing from one const to the other) REDs here.
    """

    def setUp(self):
        self.rust_identity_fields = set(
            _extract_rust_fields_block(REPORT_RS, _IDENTITY_FIELDS_BLOCK_RE, "EncodeStepTier")
        )
        self.rust_provenance_fields = set(
            _extract_rust_fields_block(REPORT_RS, _PROVENANCE_FIELDS_BLOCK_RE, "EncodeStepTier")
        )

    def test_encode_identity_fields_has_exactly_12_entries(self):
        self.assertEqual(
            len(identity_fields.ENCODE_IDENTITY_FIELDS),
            12,
            "identity_fields.py::ENCODE_IDENTITY_FIELDS must have EXACTLY 12 entries "
            "(unit-62 CONTRACT.md §E3/E6's pinned list) — a count other than 12 means "
            "either this const drifted from EncodeStepTier::IDENTITY_FIELDS or the "
            "Rust side itself grew/shrank; re-derive from source, never bump to make "
            "this test pass.",
        )
        self.assertEqual(
            len(set(identity_fields.ENCODE_IDENTITY_FIELDS)),
            12,
            "ENCODE_IDENTITY_FIELDS contains a duplicate entry",
        )

    def test_rust_provenance_fields_has_exactly_7_entries(self):
        self.assertEqual(
            len(self.rust_provenance_fields),
            7,
            f"EncodeStepTier::PROVENANCE_FIELDS ({REPORT_RS}) must have EXACTLY 7 "
            "entries (unit-62 CONTRACT.md §E3's pinned provenance list) — a count "
            f"other than 7 means the Rust const drifted: {sorted(self.rust_provenance_fields)}",
        )

    def test_encode_identity_fields_equals_the_rust_const(self):
        python_fields = set(identity_fields.ENCODE_IDENTITY_FIELDS)
        self.assertEqual(
            python_fields,
            self.rust_identity_fields,
            f"identity_fields.py::ENCODE_IDENTITY_FIELDS ({sorted(python_fields)}) must equal "
            f"EncodeStepTier::IDENTITY_FIELDS ({sorted(self.rust_identity_fields)}) EXACTLY — "
            "this tier's identity/provenance split is disjoint, not superset-folded, so the "
            "Python mirror is the WHOLE identity set, never merely a subset of it",
        )

    def test_identity_and_provenance_are_disjoint(self):
        overlap = self.rust_identity_fields & self.rust_provenance_fields
        self.assertFalse(
            overlap,
            f"EncodeStepTier::IDENTITY_FIELDS and ::PROVENANCE_FIELDS share field(s) {sorted(overlap)} — "
            "unit-62's E3 design keeps these two sets DISJOINT (never a field in both)",
        )
        overlap_py = set(identity_fields.ENCODE_IDENTITY_FIELDS) & self.rust_provenance_fields
        self.assertFalse(
            overlap_py,
            f"identity_fields.py::ENCODE_IDENTITY_FIELDS names provenance-only field(s) "
            f"{sorted(overlap_py)} — provenance must never be admitted to the Python comparison "
            "tuple either",
        )

    def test_attention_arm_is_not_an_identity_field(self):
        # Negative control (v2 reshape 3 of the unit-62 plan): a dispatched
        # arm is post-hoc, never knowable before compute, so it can never be
        # a memoization key — mirrors EncodeStepTier's own Rust-side
        # negative-control test in encode_step.rs.
        self.assertNotIn("attention_arm", identity_fields.ENCODE_IDENTITY_FIELDS)
        self.assertNotIn("attention_arm", self.rust_identity_fields)
        self.assertIn("attention_arm", self.rust_provenance_fields)


if __name__ == "__main__":
    unittest.main()
