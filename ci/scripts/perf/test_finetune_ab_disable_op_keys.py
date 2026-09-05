#!/usr/bin/env python3
"""F4 (adversarial audit fold-in — "the ninth key"), B2 (round-2
adversarial audit — "the scanner silently drops unresolvable sites"): the
MECHANICAL sweep `finetune_ab.sh`'s own `JAMMI_EAGER_DISABLE_OP_KEYS`
constant names as its own enumeration method — run this after touching any
admission call graph in `crates/jammi-encoders/src/`, `crates/jammi-lora/
src/`, or `crates/jammi-ai/src/fine_tune/` to catch a TENTH addition (or a
stale entry) mechanically, never by re-reading the whole call graph by
eye.

WHY THIS EXISTS: an eight-key version of `JAMMI_EAGER_DISABLE_OP_KEYS`
shipped with a real gap (`mem_efficient_attention`, a live per-layer
`admit_cascade` (`crates/jammi-encoders/src/modernbert.rs:1309`) site AND
a live once-per-forward `op_disabled`
(`crates/jammi-encoders/src/modernbert.rs:2976`) gate) that went
undetected because every `finetune_ab.sh` sweep config has `seq <= 512`,
and that op's own domain predicate DomainMisses unconditionally for
`seq <= ATTENTION_BLOCK_MAX_SEQ` (4096) — a coincidence of the SWEEP's own
shape, not proof the key was unneeded. A hand-re-read of the call graph
missed it once; this script performs the identical sweep mechanically,
every run.

B2 (round-2 audit): a FIRST version of this scanner's own `_STR_RE.search`
step silently returned NOTHING (an empty discovered set, no error, no
report) for any call it could not resolve a literal op key from — which
is exactly the same "absent counters is not evidence of zero" class of
bug `admission.rs`'s own `unmatched_disables()` doc warns against, one
level up: a call this scanner cannot understand read IDENTICALLY to "no
call exists here at all". `discover_live_standalone_op_keys` now returns
`(keys, unresolved_sites)` — a NON-EMPTY `unresolved_sites` (each entry a
`path:line` description) is a LOUD, mechanically-checked finding
(`RealSourceParityTests.test_no_unresolved_call_sites`), never a silent
drop. See METHOD below for exactly what makes a site "unresolved" versus
correctly excluded from the scan entirely.

METHOD: over every `.rs` file under the SCAN ROOTS (below), with comment
LINES stripped first (any line whose trimmed content starts with `//` —
covers plain `//`, `///`, and `//!` uniformly; a doc comment mentioning
`op_disabled("...")` as PROSE, e.g. `crates/jammi-encoders/src/
modernbert.rs:1460`, or a bare `// ... admit() ...` mention, e.g. that
same file's lines 1221-1222, must never be mistaken for a real call —
this is a LINE-level heuristic, never a `/* ... */` block-comment strip;
no block comment currently sits over a real call site in the scanned
crates, and this scope limitation is deliberate, not an oversight) and
`#[cfg(test)]`-attributed items excluded (a cheap brace-balanced span:
find the `{` immediately after each `#[cfg(test)]` attribute — already
comment-stripped, so a DOC COMMENT merely mentioning the literal text
"#[cfg(test)]" as prose, e.g. `crates/jammi-encoders/src/
modernbert.rs:2571/2582/2748`, can never be mistaken for a real attribute
either — and balance braces from there to the item's own close; the audit
named two REAL, in-scope test call sites this excludes,
`strict_mode_errors_instead_of_falling_back_on_a_failed_predicate`
(`crates/jammi-encoders/src/layer_norm.rs:2418`) and
`attention_block_strict_mode_errors_instead_of_falling_back_on_a_failed_predicate`
(`crates/jammi-encoders/src/modernbert.rs:11170`), both `admit(AdmissionMode::Strict, "<a key
already found at its own production site>", ...)` calls that exist purely
to unit-test THAT op's Strict-mode error path, not a second live call
site) — for every remaining `admit(`/`admit_cascade(`/`op_disabled(` call,
a BALANCED-PAREN scan (never a fixed lookahead window, which mis-attributes
a wrapper call's own literal from LATER, unrelated code) extracts the call's
own argument list, SPLITS it into top-level comma-separated arguments
(respecting nested parens/brackets/braces and string literals, so a nested
call or a string containing a comma never mis-splits), and reads OFF THE
OP-KEY ARGUMENT'S OWN POSITION specifically (`op_disabled`'s sole arg, at
index 0; `admit`/`admit_cascade`'s SECOND arg, at index 1 — both fixed by
their own `pub fn` signature in `admission.rs`, never inferred) — the
FIRST quoted string ANYWHERE in the call is deliberately NOT good enough:
`admit(mode, op, "some_predicate_name", holds, counters)` with a
*variable* `op` has ITS OWN first quoted string sitting at the PREDICATE
position, not the op-key position (the exact shape `admission.rs:1881`'s
own test cell has), and reading that off as if it were the op key would
silently manufacture a bogus discovered key from unrelated text. If the
op-position argument, after trimming, is not of the exact shape
`"[a-z0-9_]+"` (a bare double-quoted literal — a variable name, a `const`
reference, a method call, anything else) the site is UNRESOLVED, reported
with its own `path:line`, never silently dropped and never mis-attributed
to some OTHER literal found elsewhere in the same call.

SCAN ROOTS: `crates/jammi-encoders/src/`, `crates/jammi-lora/src/`,
`crates/jammi-ai/src/fine_tune/` — every `crates/jammi-bench/src/
finetune_step.rs`-adjacent crate a live standalone `admit`/`admit_cascade`/
`op_disabled` call for THIS constant's own purpose can appear in.

SCOPE, DELIBERATELY NOT `crates/jammi-kernels/src/` (corrected rationale,
round-2 audit fix (e) — the audit verified the OPERATIVE reason
empirically, and it is NOT primarily the cast-wrapper shape below, which
this scanner now handles correctly regardless): `admission.rs` itself (in
that crate) is the DEFINITION site of `admit`/`admit_cascade`/
`op_disabled`, and its own substantial `#[cfg(test)] mod tests` block unit
-tests the ADMISSION MACHINERY directly with dozens of literal, PURELY
SYNTHETIC op-key strings that name nothing real at all (e.g.
`"lattice_cell_03_real_admit_warn_op"`) — a scan of that crate is
therefore dominated by test-fixture noise this constant has no interest
in, regardless of how precisely the `#[cfg(test)]` heuristic resolves it.
Secondarily (now correctly handled, never an excuse to skip the crate
outright, just a genuinely-out-of-reach case even with a perfect scanner):
the SUBSUMED `cast_scale_bf16_f32`/`cast_add_bf16` keys live there too,
behind `ops/low_rank_residual_linear.rs`'s `admit_cast_boundary` wrapper,
whose OWN internal `admit(mode, op, ...)` call passes a *variable* `op` —
this scan correctly reports that as UNRESOLVED (never mis-attributes it to
some unrelated literal), which is exactly why it is not, and never can be,
a reason this scan needs to include that crate: nothing resolvable lives
there for THIS constant's purpose. `JAMMI_EAGER_DISABLE_OP_KEYS` must
never name either of those two directly regardless (see the constant's own
"NOT lora_epilogue/lora_dropout/cast_scale_bf16_f32/cast_add_bf16"
bullet) — verified MANUALLY against `crates/jammi-kernels/src/
admission.rs`'s own module doc (the authoritative reachability
classification) each time this constant changes, the documented
manual-sweep protocol this mechanical test's own doc names as its
complement, not a substitute for it.

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
_CFG_TEST_RE = re.compile(r"#\[cfg\(test\)\]")
_COMMENT_LINE_RE = re.compile(r"^[ \t]*//")
_LITERAL_ARG_RE = re.compile(r'^"([a-z0-9_]+)"$')

# The op-key argument's own FIXED position in each call's argument list --
# read directly off `admission.rs`'s own `pub fn` signatures
# (`op_disabled(op: &'static str)`, `admit(mode, op, predicate_name,
# predicate_holds, counters)`, `admit_cascade(mode, op, predicate_name,
# outcome, next_arm_can_run, counters)`), never inferred from "the first
# quoted string in the call" -- see this module's own doc for why that
# weaker rule mis-reads a literal PREDICATE as an op key the moment the op
# argument itself is a variable.
_OP_ARG_INDEX = {"admit": 1, "admit_cascade": 1, "op_disabled": 0}

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


def _strip_comment_lines(text):
    """Blanks (never deletes -- LINE NUMBERS must stay stable for
    `path:line` reporting) every line whose trimmed content starts with
    `//` — plain line comments, `///` doc comments, and `//!` inner doc
    comments uniformly (Rust's `//` is a strict prefix of both doc-comment
    spellings, so one check covers all three). Applied BEFORE both the
    call-site scan and the `#[cfg(test)]` span scan below, so a comment
    that merely MENTIONS either "admit(...)" or "#[cfg(test)]" as prose
    can never be mistaken for the real thing by either pass — see this
    module's own doc for concrete examples of both failure modes this
    fixes.
    """
    return "\n".join("" if _COMMENT_LINE_RE.match(line) else line for line in text.split("\n"))


def _balanced_span(text, open_idx, open_char, close_char):
    """The end index (EXCLUSIVE) of the balanced-bracket span opened by
    `text[open_idx] == open_char` — shared brace/paren-balancing core for
    both the call-body extractor and the `#[cfg(test)]` span finder below,
    so the two never independently drift on the SAME "count opens, count
    closes" logic.
    """
    depth = 1
    i = open_idx + 1
    n = len(text)
    while i < n and depth > 0:
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
        i += 1
    return i


def _find_call_body(text, open_paren_idx):
    """Text strictly between a call's own opening `(` (at `open_paren_idx`)
    and its MATCHING closing `)` -- a balanced-paren scan, not a fixed
    lookahead window, so a call whose own body has no string literal never
    accidentally absorbs an unrelated literal from LATER, unrelated code
    (the exact false positive an earlier "next 400 chars" heuristic hit
    against `admit_cast_boundary`'s own wrapper body).
    """
    end = _balanced_span(text, open_paren_idx, "(", ")")
    return text[open_paren_idx + 1 : end - 1]


def _split_top_level_args(body):
    """Splits a call's own argument-list text on TOP-LEVEL commas only --
    a comma inside a nested call's own parens (`counters_for(op)`), a
    nested `[...]`/`{...}`, or inside a string literal, must never split
    an argument in two. Returns a list of TRIMMED argument strings, in
    order -- `_OP_ARG_INDEX` then indexes directly into this list.
    """
    args = []
    depth = 0
    current = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '"':
            current.append(c)
            i += 1
            while i < n and body[i] != '"':
                if body[i] == "\\" and i + 1 < n:
                    current.append(body[i])
                    i += 1
                current.append(body[i])
                i += 1
            if i < n:
                current.append(body[i])
                i += 1
            continue
        if c in "([{":
            depth += 1
            current.append(c)
            i += 1
            continue
        if c in ")]}":
            depth -= 1
            current.append(c)
            i += 1
            continue
        if c == "," and depth == 0:
            args.append("".join(current))
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    tail = "".join(current)
    # A trailing comma before the closing paren (this codebase's own
    # rustfmt multi-line-call convention, e.g. `admit_cascade(...,
    # cascade_counters_for(...),\n)`) leaves `tail` empty/whitespace-only
    # here — dropped, never appended as a spurious final "argument".
    if tail.strip():
        args.append(tail)
    return [a.strip() for a in args]


def _cfg_test_module_spans(text):
    """`(start, end)` spans (character offsets into `text`, `end`
    EXCLUSIVE) of every `#[cfg(test)]`-attributed item — `text` MUST
    already be comment-stripped (see `_strip_comment_lines`) or a doc
    comment merely naming "#[cfg(test)]" as prose will be mistaken for a
    real attribute, whose "next `{`" then belongs to some ARBITRARY later
    item, silently swallowing real production code into a bogus "test"
    span (the exact failure this ordering — comment-strip, THEN span-scan
    — exists to prevent; reproduced during this fix's own development
    against `crates/jammi-encoders/src/modernbert.rs`'s own doc-comment
    mentions of `#[cfg(test)]` at lines 2571/2582/2748, which — un-fixed —
    swallowed the REAL production `op_disabled("mem_efficient_attention")`
    call at that file's line 2872 into a phantom "test" span).

    A CHEAP brace/attr heuristic, documented rather than hardened further
    (round-2 audit fix (a)): finds the FIRST `{` after each `#[cfg(test)]`
    attribute (the attributed item's own opening brace — correct for this
    codebase's own `#[cfg(test)] mod tests { ... }` / `#[cfg(test)] fn ...
    { ... }` conventions, both of which have exactly one such brace
    immediately reachable) and balances FROM THERE to that item's own
    close via `_balanced_span`.
    """
    spans = []
    for match in _CFG_TEST_RE.finditer(text):
        brace_idx = text.find("{", match.end())
        if brace_idx == -1:
            continue
        end = _balanced_span(text, brace_idx, "{", "}")
        spans.append((match.start(), end))
    return spans


def _in_any_span(pos, spans):
    return any(start <= pos < end for start, end in spans)


def discover_live_standalone_op_keys(roots):
    """The mechanical sweep itself. Returns `(keys, unresolved_sites)`:

      * `keys` — a `set[str]`, every op-key LITERAL successfully resolved
        from the OP-KEY ARGUMENT'S OWN POSITION (see this module's own
        METHOD doc) of a real `admit(`/`admit_cascade(`/`op_disabled(`
        call, outside any `#[cfg(test)]` span, anywhere under `roots`.
        Multiplicity is not itself meaningful (the SAME op key named at
        several sites collapses to one set member), so this is never a
        list.
      * `unresolved_sites` — a `list[str]`, one `"path:line: <call
        text>"` entry per call whose OWN op-key-position argument is NOT
        a bare double-quoted literal (a variable, a `const` reference, an
        expression) — B2's own fix: NEVER silently dropped, NEVER
        mis-attributed to some OTHER literal elsewhere in the call. Sorted
        for deterministic output. A NON-EMPTY list here means this scan
        found something it could not classify — `RealSourceParityTests`
        REDs on that, loudly, rather than silently under-counting
        `keys`.
    """
    keys = set()
    unresolved = []
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".rs"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as fh:
                    raw_text = fh.read()
                text = _strip_comment_lines(raw_text)
                test_spans = _cfg_test_module_spans(text)
                for match in _CALL_RE.finditer(text):
                    if _in_any_span(match.start(), test_spans):
                        continue
                    call_kind = match.group(1)
                    open_idx = text.index("(", match.start())
                    body = _find_call_body(text, open_idx)
                    args = _split_top_level_args(body)
                    op_index = _OP_ARG_INDEX[call_kind]
                    literal_match = None
                    if op_index < len(args):
                        literal_match = _LITERAL_ARG_RE.match(args[op_index])
                    if literal_match:
                        keys.add(literal_match.group(1))
                    else:
                        lineno = text[: match.start()].count("\n") + 1
                        call_text = text[match.start() : open_idx + 1] + body[:60]
                        unresolved.append(f"{os.path.relpath(path, REPO_ROOT)}:{lineno}: {call_text}")
    return keys, sorted(unresolved)


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
    own gates, applied here to prove a TENTH addition, a wrapper-passed
    variable, a mis-positioned literal, a doc-comment mention, and a
    `#[cfg(test)]` site are ALL handled correctly BEFORE trusting this
    scanner against the real crates.
    """

    def test_finds_a_direct_literal_admit_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let outcome = admit(admission_mode(), "brand_new_fused_op", predicate, holds, counters)?;\n'
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"brand_new_fused_op"})
            self.assertEqual(unresolved, [])

    def test_finds_an_admit_cascade_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let d = admit_cascade(admission_mode(), "a_tenth_cascade_op", reason, outcome, true, counters)?;\n'
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"a_tenth_cascade_op"})
            self.assertEqual(unresolved, [])

    def test_finds_an_op_disabled_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('let will_fire = !op_disabled("an_eleventh_gate");\n')
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"an_eleventh_gate"})
            self.assertEqual(unresolved, [])

    def test_a_wrapper_call_passing_a_variable_op_argument_is_unresolved_not_silently_dropped(self):
        """B2's own reproduction: the `admit_cast_boundary`-shaped case —
        the wrapper's OWN `admit(mode, op, ...)` call passes a *variable*
        `op` at the op-key POSITION. An earlier version of this scanner
        silently resolved this to NOTHING (an empty set, no report at
        all) — indistinguishable from "no call exists here". This must
        now be a REPORTED unresolved site, never silently blessed as
        "nothing found here".
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
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            # The later, unrelated `"cast_scale_bf16_f32"` string must
            # never be absorbed into this call, in EITHER direction.
            self.assertEqual(keys, set())
            self.assertEqual(len(unresolved), 1)
            self.assertIn("fake.rs:2", unresolved[0])

    def test_a_literal_predicate_with_a_variable_op_argument_is_unresolved_not_misread(self):
        """B2 fix (b): the `admission.rs:1881` shape — `admit(mode, op,
        "a_literal_predicate_name", holds, counters)` with a *variable*
        `op`. The call's OWN first quoted string sits at the PREDICATE
        position, not the op-key position — reading it off as if it were
        the op key would manufacture a bogus discovered key from
        unrelated text. This must resolve to UNRESOLVED, and
        "a_literal_predicate_name" must NEVER appear in `keys`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "let op = \"lattice_cell_03_real_admit_warn_op\";\n"
                    "let outcome = admit(\n"
                    "    AdmissionMode::Fallback,\n"
                    "    op,\n"
                    '    "a_literal_predicate_name",\n'
                    "    false,\n"
                    "    &counters,\n"
                    ").expect(\"Fallback mode never errors\");\n"
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertNotIn("a_literal_predicate_name", keys)
            self.assertNotIn("lattice_cell_03_real_admit_warn_op", keys)
            self.assertEqual(keys, set())
            self.assertEqual(len(unresolved), 1)

    def test_a_named_const_at_the_op_position_is_unresolved(self):
        """The audit's own const-OP fixture, verbatim: a named constant
        reference (not a bare literal) at the op-key position is exactly
        as unresolvable to this scanner as a plain local variable — a
        `const` name is an IDENTIFIER at the call site, not a literal
        this scan can read without constant-folding, which is
        deliberately out of scope for a cheap mechanical sweep.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'const OP: &str = "a_synthetic_fused_op";\n'
                    "let outcome = admit(admission_mode(), OP, \"some_predicate\", holds, counters)?;\n"
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, set())
            self.assertEqual(len(unresolved), 1)
            self.assertIn("fake.rs:2", unresolved[0])

    def test_a_call_site_named_twice_in_the_same_file_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let a = admit(mode, "same_op", pred, holds, c1)?;\n'
                    'let b = admit(mode, "same_op", pred2, holds2, c2);\n'
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"same_op"})
            self.assertEqual(unresolved, [])

    def test_cfg_test_module_is_excluded_entirely(self):
        """B2 fix (a): a REAL, syntactically-valid call inside a
        `#[cfg(test)] mod tests { ... }` block is excluded from the scan
        entirely — neither counted toward `keys` (even a NEW key named
        only in a test cell must not silently satisfy this constant) nor
        reported as `unresolved` (it is not a site this constant's own
        sweep cares about at all, not a site this scan failed to
        classify).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    'let a = admit(mode, "real_prod_op", pred, holds, c1)?;\n'
                    "\n"
                    "#[cfg(test)]\n"
                    "mod tests {\n"
                    "    #[test]\n"
                    "    fn some_test() {\n"
                    '        let b = admit(mode, "test_only_op_never_a_real_key", pred2, holds2, c2);\n'
                    "        let op_var = compute_op();\n"
                    '        let c = admit(mode, op_var, "another_predicate", holds3, c3);\n'
                    "    }\n"
                    "}\n"
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"real_prod_op"})
            self.assertNotIn("test_only_op_never_a_real_key", keys)
            self.assertEqual(unresolved, [])

    def test_a_doc_comment_mentioning_admit_as_prose_is_never_a_call_site(self):
        """B2 fix (c): the `modernbert.rs:1460` shape — a `///` doc
        comment mentioning `op_disabled("some_op")` as PROSE describing
        what a DIFFERENT real call site does, never a call itself.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "/// This function consults `op_disabled(\"phantom_op_from_prose\")`\n"
                    "/// before doing anything real.\n"
                    "//! Another mention: admit(\"also_phantom\") never runs here either.\n"
                    "// A plain comment mentioning admit() too -- never a real call.\n"
                    "fn real_function() -> bool { true }\n"
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, set())
            self.assertEqual(unresolved, [])

    def test_a_doc_comment_mentioning_cfg_test_as_prose_never_swallows_real_code(self):
        """The companion to the previous test, for the `#[cfg(test)]` span
        finder specifically — a doc comment merely NAMING "#[cfg(test)]"
        (describing where some OTHER real test lives) must never be
        mistaken for a real attribute, which would otherwise swallow
        whatever REAL code follows into a bogus "test" span (reproduced
        during this fix's own development against
        `crates/jammi-encoders/src/modernbert.rs`'s own doc comments at
        lines 2571/2582/2748, which — before comment-stripping ran BEFORE
        span-scanning — silently excluded the real production
        `op_disabled("mem_efficient_attention")` call at that file's line
        2872).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rs")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "/// See this module's own `#[cfg(test)]` harness for the full proof.\n"
                    "fn unrelated_helper() {}\n"
                    "\n"
                    "fn real_production_site() {\n"
                    '    let outcome = op_disabled("real_prod_op_after_the_doc_mention");\n'
                    "}\n"
                )
            keys, unresolved = discover_live_standalone_op_keys([tmp])
            self.assertEqual(keys, {"real_prod_op_after_the_doc_mention"})
            self.assertEqual(unresolved, [])


class RealSourceParityTests(unittest.TestCase):
    """Drives the REAL sweep against the REAL tracked source — the
    mechanical half of F4's "sweep method" the constant's own comment
    names. This is the test that catches a TENTH addition (or a stale
    entry) in CI, and (B2) the test that REDs loudly on any call site this
    scan could not classify at all.
    """

    def setUp(self):
        self.discovered, self.unresolved = discover_live_standalone_op_keys(_SCAN_ROOTS)
        self.declared = parse_jammi_eager_disable_op_keys(FINETUNE_AB_SH)

    def test_no_unresolved_call_sites(self):
        # B2's own top-level assertion: every real admit/admit_cascade/
        # op_disabled call this scan finds (outside a #[cfg(test)] span)
        # in the three scanned crates must resolve its own op-key
        # argument to a literal. A non-empty list here is a genuine
        # finding this test surfaces LOUDLY (the file:line of every
        # unresolved site), never a silent gap in `self.discovered`.
        self.assertEqual(
            self.unresolved,
            [],
            "unresolved admit()/admit_cascade()/op_disabled() call site(s) -- this scan could "
            "not read a literal op key from the op-argument position at:\n  "
            + "\n  ".join(self.unresolved),
        )

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
