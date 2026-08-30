#!/usr/bin/env python3
"""F4 (adversarial audit fold-in — "the ninth key"): the MECHANICAL sweep
`finetune_ab.sh`'s own `JAMMI_EAGER_DISABLE_OP_KEYS` constant names as its
own enumeration method — run this after touching any admission call graph
in `crates/jammi-encoders/src/`, `crates/jammi-lora/src/`, or
`crates/jammi-ai/src/fine_tune/` to catch a TENTH addition (or a stale
entry) mechanically, never by re-reading the whole call graph by eye.

WHY THIS EXISTS: an eight-key version of `JAMMI_EAGER_DISABLE_OP_KEYS`
shipped with a real gap (`mem_efficient_attention`, a live per-layer
`admit_cascade` site AND a live once-per-forward `op_disabled` gate,
`crates/jammi-encoders/src/modernbert.rs:1233`/`:2872`) that went
undetected because every `finetune_ab.sh` sweep config has `seq <= 512`,
and that op's own domain predicate DomainMisses unconditionally for
`seq <= ATTENTION_BLOCK_MAX_SEQ` (4096) — a coincidence of the SWEEP's own
shape, not proof the key was unneeded. A hand-re-read of the call graph
missed it once; this script performs the identical sweep mechanically,
every run.

METHOD: a balanced-paren scan (never a naive "grep the next 400 chars",
which mis-attributes a wrapper call's own literal — e.g.
`admit_cast_boundary`'s callers pass a LITERAL `op` argument, but the
wrapper's OWN internal `admit(mode, op, ...)` call passes a *variable*,
which this scan correctly does NOT resolve to any literal at all) over
every `admit(`/`admit_cascade(`/`op_disabled(` call in
`crates/jammi-encoders/src/`, `crates/jammi-lora/src/`, and
`crates/jammi-ai/src/fine_tune/`, reading off each call's first
double-quoted string literal argument.

SCOPE, DELIBERATELY NOT `crates/jammi-kernels/src/`: the SUBSUMED
`cast_scale_bf16_f32`/`cast_add_bf16` keys live there, behind
`ops/low_rank_residual_linear.rs`'s `admit_cast_boundary` wrapper, called
with a *variable* `op` argument at its own `admit()` call site — this
mechanical scan cannot (and must not pretend to) resolve that, and
`JAMMI_EAGER_DISABLE_OP_KEYS` must never name either of those two directly
regardless (see the constant's own "NOT lora_epilogue/lora_dropout/
cast_scale_bf16_f32/cast_add_bf16" bullet). That exclusion is verified
MANUALLY against `crates/jammi-kernels/src/admission.rs`'s own module doc
(the authoritative reachability classification) each time this constant
changes — the documented manual-sweep protocol this mechanical test's own
module doc names as its complement, not a substitute for it.

Stdlib-only, no network, no build — reads tracked source text only, same
footing `test_identity_fields_subset.py`'s own Rust-const extraction takes.

Run: `python3 ci/scripts/perf/test_finetune_ab_disable_op_keys.py`
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FINETUNE_AB_SH = os.path.join(REPO_ROOT, "ci", "scripts", "perf", "finetune_ab.sh")

# The three crates this scan covers -- see this module's own "SCOPE"
# section for why `jammi-kernels` is deliberately excluded.
_SCAN_ROOTS = (
    os.path.join(REPO_ROOT, "crates", "jammi-encoders", "src"),
    os.path.join(REPO_ROOT, "crates", "jammi-lora", "src"),
    os.path.join(REPO_ROOT, "crates", "jammi-ai", "src", "fine_tune"),
)

_CALL_RE = re.compile(r"\b(admit|admit_cascade|op_disabled)\s*\(")
_STR_RE = re.compile(r'"([a-z0-9_]+)"')

# Registered-but-non-standalone op keys: NEVER expected to appear in this
# scan's own discovered set (they live behind a wrapper in
# `crates/jammi-kernels/src/`, out of scope by design — see module doc),
# and NEVER legitimate members of `JAMMI_EAGER_DISABLE_OP_KEYS` either
# (naming any of them aborts a real run — see that constant's own doc).
# Verified against `crates/jammi-kernels/src/admission.rs`'s own module
# doc, not derived mechanically — the manual half of this sweep.
KNOWN_NON_STANDALONE_KEYS = frozenset(
    {"lora_epilogue", "lora_dropout", "cast_scale_bf16_f32", "cast_add_bf16"}
)


def _find_call_body(text, open_paren_idx):
    """Text strictly between a call's own opening `(` (at `open_paren_idx`)
    and its MATCHING closing `)` -- a balanced-paren scan, not a fixed
    lookahead window, so a call whose own body has no string literal never
    accidentally absorbs an unrelated literal from LATER, unrelated code
    (the exact false positive an earlier "next 400 chars" heuristic hit
    against `admit_cast_boundary`'s own wrapper body).
    """
    depth = 1
    i = open_paren_idx + 1
    n = len(text)
    while i < n and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return text[open_paren_idx + 1 : i - 1]


def discover_live_standalone_op_keys(roots):
    """The mechanical sweep itself: every literal op-key string passed
    directly to `admit(`/`admit_cascade(`/`op_disabled(` anywhere under
    `roots` (recursively, every `.rs` file). Returns a `set[str]` — never
    a list, since call-site MULTIPLICITY (the same op key named at several
    sites, e.g. a production call site plus its own `#[cfg(test)]`
    coverage in the SAME file) is not itself meaningful to this check.
    """
    discovered = set()
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".rs"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                for match in _CALL_RE.finditer(text):
                    open_idx = text.index("(", match.start())
                    body = _find_call_body(text, open_idx)
                    str_match = _STR_RE.search(body)
                    if str_match:
                        discovered.add(str_match.group(1))
    return discovered


def parse_jammi_eager_disable_op_keys(finetune_ab_sh_path):
    """Extracts the CURRENT `JAMMI_EAGER_DISABLE_OP_KEYS="..."` literal
    from `finetune_ab.sh` — the real source, never a hand-copied literal
    this test could itself drift from.
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


class DiscoverLiveStandaloneOpKeysTests(unittest.TestCase):
    """Unit coverage of the scanner itself, against SYNTHETIC throwaway
    fixtures — the same "prove the checker actually bites" posture
    `check_producer_provenance_gates.py --self-test` already takes for its
    own gates, applied here to prove a TENTH addition (or a wrapper-passed
    variable) is handled correctly BEFORE trusting this scanner against
    the real crates.
    """

    def test_finds_a_direct_literal_admit_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let outcome = admit(admission_mode(), "brand_new_fused_op", predicate, holds, counters)?;\n'
                )
            self.assertEqual(discover_live_standalone_op_keys([tmp]), {"brand_new_fused_op"})

    def test_finds_an_admit_cascade_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let d = admit_cascade(admission_mode(), "a_tenth_cascade_op", reason, outcome, true, counters)?;\n'
                )
            self.assertEqual(discover_live_standalone_op_keys([tmp]), {"a_tenth_cascade_op"})

    def test_finds_an_op_disabled_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('let will_fire = !op_disabled("an_eleventh_gate");\n')
            self.assertEqual(discover_live_standalone_op_keys([tmp]), {"an_eleventh_gate"})

    def test_a_wrapper_call_passing_a_variable_argument_resolves_to_nothing(self):
        """The `admit_cast_boundary`-shaped case: the wrapper's OWN
        `admit(mode, op, ...)` call passes a *variable* `op`, not a
        literal — this scan must NOT hallucinate a key from unrelated
        code further down the file (the exact false positive a naive
        "grep the next N chars" heuristic produced during this fix's own
        development).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "fn admit_cast_boundary(op: &'static str, predicate_name: &'static str) -> Result<DispatchOutcome> {\n"
                    "    admit(admission_mode(), op, predicate_name, true, counters_for(op))\n"
                    "}\n"
                    "\n"
                    'const UNRELATED_LATER_STRING: &str = "cast_scale_bf16_f32";\n'
                )
            # The wrapper's own `admit(...)` call has NO literal argument
            # within its own balanced parens -- the later, unrelated
            # `"cast_scale_bf16_f32"` string must never be absorbed into it.
            self.assertEqual(discover_live_standalone_op_keys([tmp]), set())

    def test_a_call_site_named_twice_in_the_same_file_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let a = admit(mode, "same_op", pred, holds, c1)?;\n'
                    "#[cfg(test)]\n"
                    'mod tests { fn t() { let b = admit(mode, "same_op", pred2, holds2, c2); } }\n'
                )
            self.assertEqual(discover_live_standalone_op_keys([tmp]), {"same_op"})


class RealSourceParityTests(unittest.TestCase):
    """Drives the REAL sweep against the REAL tracked source — the
    mechanical half of F4's "sweep method" the constant's own comment
    names. This is the test that catches a TENTH addition (or a stale
    entry) in CI.
    """

    def setUp(self):
        self.discovered = discover_live_standalone_op_keys(_SCAN_ROOTS)
        self.declared = parse_jammi_eager_disable_op_keys(FINETUNE_AB_SH)

    def test_jammi_eager_disable_op_keys_has_exactly_nine_entries(self):
        self.assertEqual(
            len(self.declared),
            9,
            f"JAMMI_EAGER_DISABLE_OP_KEYS ({FINETUNE_AB_SH}) must have EXACTLY 9 entries "
            f"(F4 fold-in: the original 8 plus mem_efficient_attention) — a count other than "
            f"9 means the constant drifted; re-derive from the real call graph, never bump "
            f"this number to make the test pass: {sorted(self.declared)}",
        )
        self.assertEqual(len(set(self.declared)), 9, "JAMMI_EAGER_DISABLE_OP_KEYS contains a duplicate entry")

    def test_declared_set_equals_the_discovered_live_standalone_set(self):
        # SET EQUALITY, never a subset check either direction: a key
        # present in the source but missing from the constant is exactly
        # the F4 bug this test exists to catch; a key present in the
        # constant but no longer reachable anywhere in the source is
        # equally a drift (a call site removed/renamed without updating
        # this constant).
        self.assertEqual(
            set(self.declared),
            self.discovered,
            "JAMMI_EAGER_DISABLE_OP_KEYS "
            f"({sorted(self.declared)}) must equal the mechanically-discovered live standalone "
            f"admit()/admit_cascade()/op_disabled() call-site set ({sorted(self.discovered)}) "
            "EXACTLY -- see this module's own doc for the sweep method and its "
            "jammi-kernels/src exclusion",
        )

    def test_known_non_standalone_keys_never_appear_in_the_discovered_set(self):
        # Non-vacuity anchor: proves the scanner's own `jammi-kernels/src`
        # exclusion is doing real work, not merely "these four keys happen
        # to never be grepped for any reason".
        overlap = self.discovered & KNOWN_NON_STANDALONE_KEYS
        self.assertFalse(
            overlap,
            f"the discovered set unexpectedly contains registered-but-non-standalone "
            f"key(s) {sorted(overlap)} -- either the scan scope grew to include "
            "jammi-kernels/src (update this test's own scope reasoning) or one of these "
            "keys gained a real standalone call site (update admission.rs's own "
            "classification doc AND this KNOWN_NON_STANDALONE_KEYS set)",
        )

    def test_known_non_standalone_keys_never_appear_in_the_declared_constant(self):
        overlap = set(self.declared) & KNOWN_NON_STANDALONE_KEYS
        self.assertFalse(
            overlap,
            f"JAMMI_EAGER_DISABLE_OP_KEYS names registered-but-non-standalone key(s) "
            f"{sorted(overlap)} directly -- naming any of these aborts a real run (see the "
            "constant's own 'NOT lora_epilogue/...' bullet)",
        )


if __name__ == "__main__":
    unittest.main()
