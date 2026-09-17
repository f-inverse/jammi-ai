#!/usr/bin/env python3
"""F4 (adversarial audit fold-in — "the ninth key"), B2 (round-2 adversarial
audit — "the scanner silently drops unresolvable sites"), wave-5 identity
(#546's typed-op migration retired the text scan entirely): the MECHANICAL
sweep `finetune_ab.sh`'s own `JAMMI_EAGER_DISABLE_OP_KEYS` constant names as
its own enumeration method — run this after touching
`crates/jammi-kernels/src/admission.rs`'s `PROBED_OPS` table to catch an
ELEVENTH addition (or a stale entry) mechanically, never by re-reading the
whole call graph by eye.

WHY THIS EXISTS: an eight-key version of `JAMMI_EAGER_DISABLE_OP_KEYS`
shipped with a real gap (`mem_efficient_attention`) that went undetected
because every `finetune_ab.sh` sweep config has `seq <= 512`, and that op's
own domain predicate declines unconditionally for `seq <= ATTENTION_BLOCK_MAX_SEQ`
(4096) — a coincidence of the SWEEP's own shape, not proof the key was
unneeded. A hand-re-read of the call graph missed it once; this script
performs the identical sweep mechanically, every run.

METHOD, REBUILT (wave-5 identity, #546): every `admit`/`admit_cascade` call
site across the workspace now passes a typed `&'static
jammi_kernels::admission::ProbedOp` const (`crates/jammi-kernels/src/
admission.rs`'s own `PROBED_OPS` table), never a bare string literal — the
PRIOR version of this script read the op key off a text scan of each call
site's own quoted-string argument (a balanced-paren extraction over
`crates/jammi-encoders/src`, `crates/jammi-lora/src`,
`crates/jammi-ai/src/fine_tune`), which the typed migration makes
structurally impossible: there is no literal left at an `admit`/
`admit_cascade` call site for a regex to read at all. Fixing this by
teaching the old regex the const names would be exactly the "second,
string-keyed enumeration" wave-5 identity's own I4 fix eliminated at the
Rust layer — re-introducing the identical shape here, one layer down, is
not a fix.

The live standalone op-key set is `PROBED_OPS` itself. This script reads it
from `ci/tools/probed-ops-index` (`cargo run --release -p probed-ops-index`),
a tiny CI-only Rust binary that imports `jammi_kernels::admission::PROBED_OPS`
directly and prints its own `registry_keys`/`report_keys` as JSON — the
SAME "run the real, compiled tool, never a regex over source" posture
`ci/tools/symbol-index` already established for `check_plan_citations.py`/
`check_no_consumer_names.py` (this repo's own recorded lesson: "regex
readers over YAML/Rust lost five audits"). `JAMMI_EAGER_DISABLE_OP_KEYS`'s
own live set is `PROBED_OPS`'s registry keys MINUS
`KNOWN_NON_STANDALONE_REGISTRY_KEYS` (the dtype-branching cast-boundary
rows, reached only through `jammi-kernels`'s own internal
`admit_cast_boundary` wrapper, never a standalone call site any sweep
config could disable directly) — a set-equality assertion, never a subset
check either direction, in `test_declared_set_equals_the_live_standalone_registry_key_set`.

B2's own concern — "an unresolved call site must never read identically to
'no call exists here'" — is answered differently now than it was by the
retired text scan, but not abandoned. Precisely what the compiler proves
and what a source oracle proves, stated separately: `admit`/`admit_cascade`'s
own `pub fn` signature (`crates/jammi-kernels/src/admission.rs`) requires
a `&'static jammi_kernels::admission::ProbedOp` argument, and that type is
now sealed by TWO separate mechanisms, proved separately, neither alone
sufficient: `#[non_exhaustive]` refuses struct-literal CONSTRUCTION from
outside the crate (a forged value built with a struct literal in another
crate is `error[E0639]: cannot create non-exhaustive struct using struct
expression`); every field being `pub(crate)`, not `pub`, refuses field
ASSIGNMENT on a value already held from outside the crate, including one
obtained by copying a real `PROBED_OPS` row (`Copy`, no struct expression
at all, so `#[non_exhaustive]` alone never engages there — a probe against
an earlier revision that sealed only `#[non_exhaustive]` copied a real
row, assigned a field on the copy directly, and `admit` honoured the
forged value; the same probe against field-private `ProbedOp` is
`error[E0616]: field \`report_key\` of struct \`ProbedOp\` is private`).
Together, on every `cargo build`, the Rust compiler proves that a call
site OUTSIDE `jammi-kernels` cannot pass anything but one of `PROBED_OPS`'s
own named consts, by either route. Neither mechanism has any effect
INSIDE the crate that defines the type, though — a same-crate forgery
inside `jammi-kernels` itself is a residual the compiler alone does not
close, closed instead by a real `syn` source oracle,
`crates/jammi-kernels/tests/probed_op_construction_sites.rs`, which proves
every `ProbedOp::new(...)`-equivalent call (matched by its own last two
path segments under any qualifying prefix, qualified-self syntax,
`Self::new` inside `impl ProbedOp`, or a same-crate type alias — never a
fixed, exact segment count — and name-keyed against the real, linked-in
`PROBED_OPS` constant's own `report_key`s, never a bare count), every
`ProbedOp { ... }`-equivalent struct literal, every fn whose own return
type names `ProbedOp`-equivalent, every macro INVOCATION whose own token
stream names `ProbedOp`/a resolved alias at all (`vec![ProbedOp::new(...)]`
and `vec![ProbedOp { ... }]` are both opaque to the other directions'
typed traversal), and every `transmute` whose target type is named
explicitly (a turbofish, or a `let`-binding's own annotation), anywhere
under that crate's `src/`/`tests/` trees, is either the one reviewed
constructor (`ProbedOp::new`'s own body) or one of two named, reviewed
`#[cfg(test)]` fixture macros. The one HONESTLY NAMED residual neither the
compiler's two mechanisms nor the oracle closes: a `transmute` (or
raw-pointer cast) whose target `ProbedOp` type is established some OTHER
way a syntax-only, type-checker-free scan cannot resolve — `jammi-kernels`
does not carry `#![forbid(unsafe_code)]`, so this is not claimed closed.
Together, the compiler's two proofs and the oracle's seven directions
cover every crate; no one of them alone does. Separately, this
script still checks NON-VACUITY: that real
`admit`/`admit_cascade` call sites genuinely exist under the scan roots at
all (`admit_call_sites`, via
`ci/tools/symbol-index`'s real `syn` parse — never a regex — filtered to
non-test call sites by callee name) — a scan root silently returning zero
hits (a typo'd path, a directory that stopped existing) is the failure mode
this residual check still catches; WHICH op each site passes needs no
further verification, because the type system already did it.

Not stdlib-only anymore (an intentional, documented departure from an
earlier revision's own "no build" boast): `probed_ops_registry_keys` and
`admit_call_sites` both shell out to a real `cargo run --release`, exactly
as `ci/tools/symbol-index`'s own established callers already do.

Run: `python3 ci/scripts/perf/test_finetune_ab_disable_op_keys.py`
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FINETUNE_AB_SH = os.path.join(REPO_ROOT, "ci", "scripts", "perf", "finetune_ab.sh")

PROBED_OPS_INDEX_CRATE = "probed-ops-index"
SYMBOL_INDEX_CRATE = "symbol-index"

# The three crates a live standalone `admit`/`admit_cascade` call site for
# `JAMMI_EAGER_DISABLE_OP_KEYS`'s own purpose can appear in — deliberately
# NOT `crates/jammi-kernels/src` (that crate is `admit`/`admit_cascade`'s
# own DEFINITION site, and the dtype-branching cast-boundary wrapper that
# makes `cast_scale_bf16_f32`/`cast_scale_f16_f32`/`cast_add_bf16`/
# `cast_add_f16` non-standalone lives there too — see
# `KNOWN_NON_STANDALONE_REGISTRY_KEYS`'s own doc). Paths are REPO-ROOT-
# RELATIVE (not absolute): `symbol-index`'s own CLI takes root directories
# resolved against its own `cwd`, and `admit_call_sites` runs it with
# `cwd=REPO_ROOT` — matching `check_no_consumer_names.py::build_symbol_index`'s
# own established calling convention exactly.
_SCAN_ROOTS = (
    "crates/jammi-encoders/src",
    "crates/jammi-lora/src",
    "crates/jammi-ai/src/fine_tune",
)

# Registered `PROBED_OPS` registry keys that are real rows (checked
# mechanically below, never assumed) but never reached by a STANDALONE
# `admit`/`admit_cascade` call site any `finetune_ab.sh` sweep config could
# disable directly: the dtype-branching `cast_scale`/`cast_add` rows are
# only ever reached through `crates/jammi-kernels/src/ops/
# low_rank_residual_linear.rs`'s own `admit_cast_boundary` wrapper (inside
# `crates/jammi-kernels/src`, outside `_SCAN_ROOTS` by design), whose OWN
# internal `admit(mode, op, ...)` call passes a *variable* `op`, never a
# literal — verified against `crates/jammi-kernels/src/admission.rs`'s own
# module doc (the authoritative reachability classification for every
# `PROBED_OPS` row) each time this constant changes, the manual half of
# this sweep its own module doc calls its complement, never a substitute.
KNOWN_NON_STANDALONE_REGISTRY_KEYS = frozenset(
    {"cast_scale_bf16_f32", "cast_scale_f16_f32", "cast_add_bf16", "cast_add_f16"}
)

# Registered-but-DEAD registry keys from an earlier `lora_linear.rs`
# registry generation that were never promoted to a `PROBED_OPS` row at
# all — no `admit()`/`admit_cascade()` call site anywhere in the workspace
# ever passed either, in ANY crate, past or present. Named here purely so
# `JAMMI_EAGER_DISABLE_OP_KEYS` never accidentally re-adds either; checked
# mechanically (`test_known_dead_registry_keys_are_not_probed_ops_entries`)
# in the OPPOSITE direction from `KNOWN_NON_STANDALONE_REGISTRY_KEYS` above
# (these must be ABSENT from `PROBED_OPS`'s own registry set, not merely
# present-but-unreachable) — a real row appearing for either would be
# exactly as much a drift as a stale entry in the non-standalone set.
KNOWN_DEAD_REGISTRY_KEYS = frozenset({"lora_epilogue", "lora_dropout"})


def _run_cargo_tool(crate, extra_args=()):
    """Runs `cargo run --release -p <crate> [-- extra_args]` from
    `REPO_ROOT` and returns its parsed stdout JSON. Shared by
    `probed_ops_registry_keys` and `admit_call_sites` below — one
    "run the real compiled tool, never a regex" invocation, not two
    independently-drifting copies.
    """
    cmd = ["cargo", "run", "--release", "-p", crate]
    if extra_args:
        cmd += ["--", *extra_args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{crate} failed (rc={proc.returncode}) with args {list(extra_args)}:\n"
            f"{proc.stderr.strip()[-4000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{crate} did not emit valid JSON on stdout ({exc}); stderr tail:\n"
            f"{proc.stderr.strip()[-2000:]}"
        ) from exc


def probed_ops_registry_keys():
    """The real, live `PROBED_OPS` registry-key set — `ci/tools/
    probed-ops-index`'s own dump of `jammi_kernels::admission::PROBED_OPS`,
    never a text scan. See this module's own doc for why a text scan can
    no longer read this set at all (#546's typed-op migration).
    """
    data = _run_cargo_tool(PROBED_OPS_INDEX_CRATE)
    return set(data["registry_keys"])


def admit_call_sites(roots):
    """Every non-test `admit`/`admit_cascade` call site `ci/tools/
    symbol-index`'s real `syn` parse finds under `roots` — `(path, line)`
    pairs, sorted. A NON-VACUITY check only (see this module's own doc):
    WHICH `PROBED_OPS` const each site passes needs no verification here,
    because `admit`/`admit_cascade`'s own `pub fn` signature already
    forces it, compiler-side, on every `cargo build`.
    """
    index = _run_cargo_tool(SYMBOL_INDEX_CRATE, roots)
    return sorted(
        (call["path"], call["line"])
        for call in index["calls"]
        if call["callee"] in ("admit", "admit_cascade") and not call.get("in_test", False)
    )


def parse_jammi_eager_disable_op_keys(finetune_ab_sh_path):
    """Extracts the CURRENT `JAMMI_EAGER_DISABLE_OP_KEYS="..."` literal
    from `finetune_ab.sh` — the real source, never a hand-copied literal
    this test could itself drift from. Unaffected by #546's typed-op
    migration (this is a bash constant, not a Rust call site) — kept
    exactly as it was.
    """
    with open(finetune_ab_sh_path, encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r'JAMMI_EAGER_DISABLE_OP_KEYS="([^"]+)"', text)
    if match is None:
        raise SystemExit(
            f"FAIL-CLOSED: no JAMMI_EAGER_DISABLE_OP_KEYS=\"...\" assignment found in "
            f"{finetune_ab_sh_path} — either the constant was renamed/removed (RED at base) "
            f"or this regex needs updating for a genuine reformat"
        )
    return [key for key in match.group(1).split(",") if key]


class ParseJammiEagerDisableOpKeysTests(unittest.TestCase):
    """Unit coverage of the one text-extraction this script still performs
    (a bash literal, not a Rust call site — #546's typed-op migration does
    not touch this).
    """

    def test_extracts_the_literal_from_a_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "finetune_ab.sh")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('JAMMI_EAGER_DISABLE_OP_KEYS="a_key,b_key,c_key"\n')
            self.assertEqual(
                parse_jammi_eager_disable_op_keys(path), ["a_key", "b_key", "c_key"]
            )

    def test_a_missing_assignment_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "finetune_ab.sh")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# no assignment here at all\n")
            with self.assertRaises(SystemExit):
                parse_jammi_eager_disable_op_keys(path)


class AdmitCallSitesTests(unittest.TestCase):
    """Proves `admit_call_sites` genuinely finds a real, non-test call site
    and genuinely excludes a `#[cfg(test)]`-scoped one — against the REAL,
    compiled `symbol-index` tool over a synthetic fixture tree (the same
    "RED->GREEN shape against the real compiled tool" posture
    `check_plan_citations.py`'s own symbol-index self-tests take), never a
    mock of its output.
    """

    def test_a_real_admit_call_is_found_and_a_test_only_one_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "fn real_call_site(mode: AdmissionMode, op: &'static ProbedOp) {\n"
                    "    let _ = admit(mode, op, \"a_predicate\", true, counters);\n"
                    "}\n"
                    "\n"
                    "#[cfg(test)]\n"
                    "mod tests {\n"
                    "    #[test]\n"
                    "    fn some_test() {\n"
                    "        let _ = admit_cascade(mode, op, \"pred\", outcome, true, counters);\n"
                    "    }\n"
                    "}\n"
                )
            sites = admit_call_sites([tmp])
            self.assertEqual(len(sites), 1, sites)
            self.assertTrue(sites[0][0].endswith("fake.rs"), sites[0])
            self.assertEqual(sites[0][1], 2)

    def test_an_empty_root_finds_nothing_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(admit_call_sites([tmp]), [])


class RealSourceParityTests(unittest.TestCase):
    """Drives the REAL sweep against the REAL, compiled tools — the
    mechanical half of F4's "sweep method" the constant's own comment
    names. This is the test that catches an ELEVENTH addition (or a stale
    entry) in CI.
    """

    def setUp(self):
        self.registry_keys = probed_ops_registry_keys()
        self.declared = parse_jammi_eager_disable_op_keys(FINETUNE_AB_SH)
        self.admit_call_sites = admit_call_sites(_SCAN_ROOTS)

    def test_every_admit_admit_cascade_call_site_is_type_resolved(self):
        # Non-vacuity: real, syn-derived (never regex-derived) admit()/
        # admit_cascade() call sites genuinely exist under the scan roots.
        # WHICH PROBED_OPS const each one passes needs no further check
        # here — see this module's own doc for why that half is already a
        # compiler-enforced fact, not a Python-testable one.
        self.assertGreater(
            len(self.admit_call_sites),
            0,
            "found zero admit()/admit_cascade() call sites under the scan roots via "
            "symbol-index -- either every fused-op call site was genuinely removed "
            "(update _SCAN_ROOTS' own doc) or the scan-root paths/symbol-index "
            "integration broke; investigate before trusting the rest of this suite",
        )

    def test_jammi_eager_disable_op_keys_has_exactly_ten_entries(self):
        self.assertEqual(
            len(self.declared),
            10,
            f"JAMMI_EAGER_DISABLE_OP_KEYS ({FINETUNE_AB_SH}) must have EXACTLY 10 entries "
            f"(F4 fold-in: the original 8 plus mem_efficient_attention; issue #463 fold-in: "
            f"plus gelu_erf_fused) — a count other than 10 means the constant drifted; "
            f"re-derive from PROBED_OPS's own registry, never bump this number to make the "
            f"test pass: {sorted(self.declared)}",
        )
        self.assertEqual(len(set(self.declared)), 10, "JAMMI_EAGER_DISABLE_OP_KEYS contains a duplicate entry")

    def test_declared_set_equals_the_live_standalone_registry_key_set(self):
        # SET EQUALITY, never a subset check either direction: a registry
        # key present in PROBED_OPS but missing from the constant is
        # exactly the F4 bug this test exists to catch; a key present in
        # the constant but no longer a PROBED_OPS registry entry at all is
        # equally a drift.
        live_standalone = self.registry_keys - KNOWN_NON_STANDALONE_REGISTRY_KEYS
        self.assertEqual(
            set(self.declared),
            live_standalone,
            "JAMMI_EAGER_DISABLE_OP_KEYS "
            f"({sorted(self.declared)}) must equal PROBED_OPS's own registry-key set minus "
            f"KNOWN_NON_STANDALONE_REGISTRY_KEYS ({sorted(live_standalone)}) EXACTLY -- see "
            "this module's own doc for the sweep method",
        )

    def test_known_non_standalone_registry_keys_are_real_probed_ops_entries(self):
        # Non-vacuity anchor: proves KNOWN_NON_STANDALONE_REGISTRY_KEYS
        # names REAL PROBED_OPS rows this set-equality check is correctly
        # excluding, not four typo'd strings that happen to never collide
        # with anything for an unrelated reason.
        missing = KNOWN_NON_STANDALONE_REGISTRY_KEYS - self.registry_keys
        self.assertFalse(
            missing,
            f"KNOWN_NON_STANDALONE_REGISTRY_KEYS names key(s) {sorted(missing)} that are NOT "
            "real PROBED_OPS registry entries -- either a row was renamed/removed (update "
            "this set) or one of these was never real to begin with",
        )

    def test_known_dead_registry_keys_are_not_probed_ops_entries(self):
        # The opposite-direction non-vacuity anchor: these two must stay
        # ABSENT from PROBED_OPS's own registry set. A real row appearing
        # for either would itself be a drift this test catches.
        present = KNOWN_DEAD_REGISTRY_KEYS & self.registry_keys
        self.assertFalse(
            present,
            f"KNOWN_DEAD_REGISTRY_KEYS names key(s) {sorted(present)} that ARE now real "
            "PROBED_OPS registry entries -- one of these gained a real row; update "
            "JAMMI_EAGER_DISABLE_OP_KEYS and move it out of this set",
        )

    def test_declared_never_names_a_non_standalone_or_dead_key(self):
        overlap = set(self.declared) & (
            KNOWN_NON_STANDALONE_REGISTRY_KEYS | KNOWN_DEAD_REGISTRY_KEYS
        )
        self.assertFalse(
            overlap,
            f"JAMMI_EAGER_DISABLE_OP_KEYS names registered-but-non-standalone-or-dead "
            f"key(s) {sorted(overlap)} directly -- naming any of these aborts a real run "
            "(see JAMMI_EAGER_DISABLE_OP_KEYS's own 'NOT lora_epilogue/...' bullet)",
        )


if __name__ == "__main__":
    unittest.main()
