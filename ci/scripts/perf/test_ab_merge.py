#!/usr/bin/env python3
"""Fixture-directory tests for `ab_merge.py` — the merge/table stage
`finetune_ab.sh` invokes as `python3 "$DIR/ab_merge.py" ...`.

B3: the merge/table stage previously had ZERO automated coverage (it was an
inline heredoc; `AB_DRY_RUN=1` only ever exercised the DRY_RUN arm, never a
real report shape). Every test here builds a fixture directory shaped
EXACTLY like `run_leg`'s own `.exit`/`.json`/`.stderr` triples, then drives
`ab_merge.main(argv)` — the REAL entry point `finetune_ab.sh` calls — never
`fused_proof()` or `dispatch_pairs()` in isolation with literal tuples
standing in for a report.

Stdlib-only (`unittest`), no external dependency — same footing
`torch_finetune_step.py`'s own "never a Cargo dependency, never a pinned
requirements file" stance (crates/jammi-bench/reference/README.md's B2
section): this is a CI-adjacent script, not a package, and nothing here
should ever tempt CI into enforcing a Python requirements file against a
crate that has no Python toolchain.

Run directly: `python3 ci/scripts/perf/test_ab_merge.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_merge  # noqa: E402


LEGS = ab_merge.LEGS

# Every fused/eager pair `FinetuneStepTier` actually serializes today (see
# `crates/jammi-bench/src/report.rs`'s `FinetuneStepTier` and this repo's
# real `finetune-step` output — captured directly, not guessed at, while
# building this fixture set).
ALL_BASES = ("ln", "rope", "softmax", "geglu", "lora_epilogue", "lora_linear", "attention_block")


def jammi_fs(dispatches, **overrides):
    """Build a `finetune_step.rs`-shaped `finetune_step` block. `dispatches`
    maps a pair base to `(fused, eager)`; any base in `ALL_BASES` not given
    defaults to `(0, 0)`. `overrides` can drop a `..._dispatches` key
    entirely (pass `None` as its value) to simulate a solo/vanished
    counter, or override any other field (e.g. `loss_first`).
    """
    fs = {
        "device": "cpu",
        "device_name": "cpu",
        # `seed` is new this round (`crates/jammi-bench/src/report.rs`'s
        # `FinetuneStepTier::seed` field): the leg-premise check below reads
        # it directly off THIS sub-block for a jammi leg (jammi's own
        # `finetune_step.rs` carries it inline, unlike torch's — see
        # `torch_fs`'s own doc for why torch's lives one level up).
        "seed": 42,
        "backbone_dtype": "bf16",
        # round-4 audit fold-in: checkpoint content identity, lora_alpha,
        # and margin — all new `FinetuneStepTier` fields this round. Same
        # literal defaults as `torch_fs` below so a matching-premise pair
        # (the overwhelming default use of both fixtures) stays matching
        # without every existing call site having to override them.
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_weights_sha256": "b" * 64,
        "checkpoint_weights_size_bytes": 1024,
        "batch": 8,
        "seq": 128,
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "lora_dropout": 0.0,
        "margin": 0.3,
        "target_modules": ["Wqkv", "Wo", "Wi"],
        "batched_forward": True,
        "trainable_tensors": 4,
        "steps_measured": 20,
        "losses": [0.3046, 0.3012],
        "loss_first": 0.3046,
        "loss_last": 0.3012,
        "s_per_step_p50": {"value": 0.01, "unit": "s"},
        "s_per_step_mean": {"value": 0.0105, "unit": "s"},
        "steps_per_s": {"value": 100.0, "unit": "steps/s"},
        "triplets_per_s": {"value": 800.0, "unit": "triplets/s"},
        "peak_rss_bytes": {"value": 123456.0, "unit": "bytes"},
        "peak_vram_bytes": {"value": 999.0, "unit": "bytes"},
    }
    for base in ALL_BASES:
        fused, eager = dispatches.get(base, (0, 0))
        fs[f"{base}_fused_dispatches"] = fused
        fs[f"{base}_eager_dispatches"] = eager
    fs.update(overrides)
    for key, value in list(fs.items()):
        if value is None:
            del fs[key]
    return {"tiers": {"finetune_step": fs}}


def drop_dispatch_keys(*bases):
    """An `overrides` dict for `jammi_fs` that DELETES both
    `_fused_dispatches` and `_eager_dispatches` for each given base
    entirely (`jammi_fs`'s own `None`-deletes-the-key convention) --
    simulates the base being ABSENT from the schema (a field renamed,
    deleted, or feature-gated off), never merely reading `(0, 0)`. F5's
    fix requires every `ALL_BASES` member to be PRESENT; this is how the
    fixtures below construct the "classified base vanished from the
    schema entirely" regression it now catches.
    """
    overrides = {}
    for base in bases:
        overrides[f"{base}_fused_dispatches"] = None
        overrides[f"{base}_eager_dispatches"] = None
    return overrides


def torch_fs(seed=42, attn_requested="sdpa", lora_alpha=32.0, margin=0.3, **overrides):
    """Builds the FULL top-level `torch_finetune_step.py` report shape, not
    just the `finetune_step` sub-block: `seed`/`attn_requested`/`lora_alpha`/
    `margin` live under `report["args"]` on the REAL torch producer
    (`torch_finetune_step.py`'s own report literal), never inside
    `report["finetune_step"]` — the leg-premise check reads jammi's copies
    off `finetune_step.rs`'s own `FinetuneStepTier` (one level down, see
    `jammi_fs`'s doc) and torch's off `args.*` (one level UP) precisely
    because the two real producers do not put them in the same place; this
    fixture mirrors that asymmetry rather than flattening it away, matching
    the REAL producer's own shape. `**overrides` still lands in the
    `finetune_step` sub-block (e.g. `attn_implementation="eager"`),
    unchanged from before this round.
    """
    fs = {
        "device": "cpu",
        "backbone_dtype": "bf16",
        # Same literal defaults as `jammi_fs` above -- see that fixture's
        # own comment for why.
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_weights_sha256": "b" * 64,
        "checkpoint_weights_size_bytes": 1024,
        "attn_implementation": "sdpa",
        "batch": 8,
        "seq": 128,
        "lora_rank": 16,
        "lora_dropout": 0.0,
        "lora_init": "peft",
        "target_modules": ["Wqkv", "Wo", "Wi"],
        "batched_forward": True,
        "trainable_tensors": 4,
        "steps_measured": 20,
        "losses": [0.31, 0.10],
        "loss_first": 0.31,
        "loss_last": 0.10,
        "s_per_step_p50": {"value": 0.011, "unit": "s"},
        "s_per_step_mean": {"value": 0.0115, "unit": "s"},
        "steps_per_s": {"value": 90.9, "unit": "steps/s"},
        "triplets_per_s": {"value": 727.0, "unit": "triplets/s"},
        "peak_rss_bytes": {"value": 654321.0, "unit": "bytes"},
        "peak_vram_baseline_bytes": {"value": 100.0, "unit": "bytes"},
        "peak_vram_absolute_bytes": {"value": 1100.0, "unit": "bytes"},
        "peak_vram_delta_bytes": {"value": 1000.0, "unit": "bytes"},
    }
    fs.update(overrides)
    return {
        "tool": "torch_finetune_step",
        "args": {"seed": seed, "attn_requested": attn_requested, "lora_alpha": lora_alpha, "margin": margin},
        "finetune_step": fs,
    }


def write_leg(raw_dir, slug, leg, exit_code=0, report=None, stderr=""):
    base = os.path.join(raw_dir, f"{slug}__{leg}")
    with open(base + ".exit", "w") as fh:
        fh.write(str(exit_code))
    with open(base + ".stderr", "w") as fh:
        fh.write(stderr)
    if report is not None:
        with open(base + ".json", "w") as fh:
            json.dump(report, fh)


def write_ok_config(raw_dir, slug, dispatches, jammi_overrides=None, torch_overrides=None):
    """Write all 4 legs for one config, jammi-eager/torch-eager as
    plausible-but-uninteresting OK rows (this proof only reads jammi-fused
    + torch-sdpa), jammi-fused carrying `dispatches`.
    """
    write_leg(raw_dir, slug, "jammi-eager", report=jammi_fs({}))
    write_leg(raw_dir, slug, "jammi-fused", report=jammi_fs(dispatches, **(jammi_overrides or {})))
    write_leg(raw_dir, slug, "torch-eager", report=torch_fs(attn_implementation="eager"))
    write_leg(raw_dir, slug, "torch-sdpa", report=torch_fs(**(torch_overrides or {})))


class FusedProofFixtureTests(unittest.TestCase):
    """Drives `ab_merge.main` (the real `finetune_ab.sh` entry point)
    against a fixture RAW_DIR, then reads back `jammi_fused_dispatch_proof`
    from the merged JSON — never calling `fused_proof`/`dispatch_pairs`
    directly with a literal tuple in place of a report.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        with open(os.path.join(out_dir, "finetune_ab_table.txt")) as fh:
            table = fh.read()
        return rc, merged, table

    def test_exclusive_pair_yes(self):
        """ln/geglu required+independent pairs fused; attention_block fused
        (so rope/softmax legitimately (0, 0), absorbed); lora_epilogue
        (0, 0) but lora_linear fused (the group's sum > 0). Every eager
        count 0. This is what a genuinely fully-fused run looks like.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                {
                    "ln": (9, 0),
                    "rope": (0, 0),
                    "softmax": (0, 0),
                    "geglu": (3, 0),
                    "lora_epilogue": (0, 0),
                    "lora_linear": (3, 0),
                    "attention_block": (3, 0),
                },
            )
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], True)

    def test_eager_leak_no(self):
        """Same as the YES case, but lora_linear ALSO shows a real eager
        fallback (1) alongside its fused count -- an admitted call site
        that actually fell back must hard-fail regardless of how many
        OTHER pairs look clean.

        Advisory (iv), round-2 audit fix on PR #372: a `False` proof now
        turns the verdict `INVALID` (never merely a `[WARN]` suffix on
        whatever ratio-based verdict would have applied) and the SWEEP's
        own exit code goes non-zero -- driven at the real `main()` entry
        point, not `build_report`'s internals in isolation.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                {
                    "ln": (9, 0),
                    "geglu": (3, 0),
                    "lora_linear": (3, 1),
                    "attention_block": (3, 0),
                },
            )
            rc, merged, table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertIn("INVALID", table)
        self.assertEqual(rc, 1, "a failed fused_proof must turn the sweep's own exit code non-zero")

    def test_all_zero_no(self):
        """Every single pair reads (0, 0) -- a schema regression that
        dropped every counter, or a config that dispatched nothing at all.
        NOT vacuously True.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_mixed_fused_and_eager_no(self):
        """Some pairs show real fused activity (ln, geglu), but ANOTHER
        pair shows a real eager fallback (softmax, here NOT absorbed since
        attention_block itself is all-eager) -- a mixed report is still a
        hard fail, never averaged into a YES.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                {
                    "ln": (9, 0),
                    "geglu": (3, 0),
                    "softmax": (0, 3),
                    "attention_block": (0, 3),
                    "lora_linear": (3, 0),
                },
            )
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_solo_counter_is_a_loud_per_config_failure_not_a_crash(self):
        """B2 + B6: a `_fused_dispatches` key with no `_eager_dispatches`
        sibling (a schema bug -- a struct field added without its pair)
        must be a LOUD, visible failure for THIS config's `jammi-fused`
        row (an `"ERROR: ..."` string, not a silent False/None that looks
        like an ordinary negative) -- and must NOT abort the merge for a
        SECOND, otherwise-healthy config in the same sweep.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            broken = jammi_fs({"ln": (9, 0), "geglu": (3, 0), "lora_linear": (3, 0), "attention_block": (3, 0)})
            del broken["tiers"]["finetune_step"]["softmax_eager_dispatches"]  # solo counter
            write_leg(raw_dir, "b8-s128-solo", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-solo", "jammi-fused", report=broken)
            write_leg(raw_dir, "b8-s128-solo", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-solo", "torch-sdpa", report=torch_fs())

            write_ok_config(
                raw_dir,
                "b8-s128-healthy",
                {"ln": (9, 0), "geglu": (3, 0), "lora_linear": (3, 0), "attention_block": (3, 0)},
            )

            rc, merged, table = self.run_merge(raw_dir)

        broken_proof = merged["configs"]["b8-s128-solo"]["jammi_fused_dispatch_proof"]
        self.assertIsInstance(broken_proof, str)
        self.assertIn("ERROR", broken_proof)
        self.assertIn("softmax_eager_dispatches", broken_proof)
        # B6: one bad leg must not abort the merge for the OTHER config --
        # both configs are still present and correctly classified in the
        # merged JSON, one bad leg does not silently swallow the other.
        self.assertTrue(merged["configs"]["b8-s128-solo"]["verdict"].startswith("INVALID"))
        self.assertIs(merged["configs"]["b8-s128-healthy"]["jammi_fused_dispatch_proof"], True)
        self.assertFalse(merged["configs"]["b8-s128-healthy"]["verdict"].startswith("INVALID"))
        # The error is visible in the printed table too, not just the JSON.
        self.assertIn("ERROR", table)
        # Advisory (iv): the errored config's own INVALID verdict is what
        # now gates the SWEEP's exit code non-zero -- the healthy config
        # passing does not paper over it.
        self.assertEqual(rc, 1)

    def test_vanished_site_case_all_zero_except_attention_block(self):
        """B2's own worst-case example: ln/rope/softmax/geglu/lora_epilogue/
        lora_linear ALL read (0, 0) and only attention_block reads (10, 0).
        The pre-fix generalized code printed YES here (a net loss of
        detection vs. the pre-generalization ln/rope/softmax-required
        check); this must be NO, because `ln` (REQUIRED, absorbed by
        nothing) reads (0, 0).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"attention_block": (10, 0)})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_vanished_site_case_only_lora_epilogue_positive(self):
        """B2's other named example: only `lora_epilogue` reads (1, 0),
        everything else including `ln` reads (0, 0). Must be NO for the
        same reason as the attention_block-only case above.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"lora_epilogue": (1, 0)})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_no_dispatch_pairs_at_all_reads_false_not_none(self):
        """`jammi_fused_dispatch_proof` is `None` when the jammi-fused leg
        itself did not run at all (MISSING/FAIL), and `False` (never
        treated the same as "did not run") when it DID run OK but its
        schema carries literally ZERO dispatch-pair keys (every base's
        `_fused_dispatches`/`_eager_dispatches` pair entirely absent) --
        both driven through `ab_merge.main`, the REAL entry point, never
        `fused_proof`/`dispatch_pairs` called directly with a literal dict
        standing in for a report (F5's own fix to this exact test: an
        earlier draft called `fused_proof({"dispatch_pairs": []})`
        directly, contradicting this file's and `ab_merge.py`'s own "never
        a hand-rolled call with literal tuples" claim).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-missing", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-missing", "jammi-fused", exit_code=1, stderr="boom")
            write_leg(raw_dir, "b8-s128-missing", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-missing", "torch-sdpa", report=torch_fs())

            write_leg(raw_dir, "b8-s128-empty", "jammi-eager", report=jammi_fs({}))
            empty_report = jammi_fs({}, **drop_dispatch_keys(*ALL_BASES))
            write_leg(raw_dir, "b8-s128-empty", "jammi-fused", report=empty_report)
            write_leg(raw_dir, "b8-s128-empty", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-empty", "torch-sdpa", report=torch_fs())

            rc, merged, _table = self.run_merge(raw_dir)

        self.assertIsNone(merged["configs"]["b8-s128-missing"]["jammi_fused_dispatch_proof"])
        # `None` (leg did not run at all) is NOT the same failure class as
        # `False` (leg ran, proof checked, and failed) -- only the latter
        # becomes an INVALID verdict; a leg that never ran gets its own
        # ordinary (non-INVALID) FAIL/N-A verdict from the outcome-based
        # rules, unaffected by advisory (iv)'s fix.
        self.assertFalse(merged["configs"]["b8-s128-missing"]["verdict"].startswith("INVALID"))
        self.assertIs(merged["configs"]["b8-s128-empty"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-empty"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1, "the b8-s128-empty config's INVALID verdict must gate the sweep exit code")

    def test_geglu_zero_zero_now_fails_the_f5_reproduction(self):
        """F5 REPRODUCTION: before this fix, `fused_proof` over pairs
        including `geglu = (0, 0)` (present, reading zero -- e.g. a
        deleted/feature-gated-off fused MLP) returned `True` as long as
        `ln`/`attention_block`/`lora_linear` each independently cleared
        their own bar, because `geglu` was in NO classification set at
        all. `geglu` is now in `REQUIRED_PAIRS` (matching
        `finetune_ab.sh`'s header, which always claimed this) -- must now
        be NO.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                {
                    "ln": (9, 0),
                    "geglu": (0, 0),
                    "attention_block": (3, 0),
                    "lora_linear": (3, 0),
                },
            )
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_rope_softmax_entirely_absent_from_schema_now_fails_the_f5_reproduction(self):
        """F5 REPRODUCTION: `fused_proof([('ln', 9, 0), ('lora_linear', 3,
        0)])` -- rope/softmax/geglu/attention_block ALL entirely ABSENT
        from the report's schema, not merely reading `(0, 0)` -- used to
        return `True`: the old code's `continue`d past an
        `ABSORBABLE_BY_ATTENTION_BLOCK` member that was simply not present
        at all, granting it a free pass no `REQUIRED_PAIRS` member ever
        got. Must now be NO: an ABSENT classified base is a hard fail for
        EVERY class, the same treatment `ln`'s absence already got before
        this fix.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            report = jammi_fs(
                {"ln": (9, 0), "lora_linear": (3, 0)},
                **drop_dispatch_keys("rope", "softmax", "geglu", "attention_block", "lora_epilogue"),
            )
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_only_ln_present_everything_else_absent_now_fails_the_f5_reproduction(self):
        """F5's most extreme reproduction, quoted directly from the audit:
        `fused_proof([('ln', 9, 0)])` -- EVERY OTHER classified base
        entirely missing from the schema -- used to return `True`. Must
        now be NO.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            report = jammi_fs(
                {"ln": (9, 0)},
                **drop_dispatch_keys("rope", "softmax", "geglu", "attention_block", "lora_epilogue", "lora_linear"),
            )
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_unclassified_base_is_a_loud_per_config_error_not_a_silent_pass(self):
        """A NEW fused kernel's dispatch pair landing in `finetune_step.rs`
        without `ab_merge.py`'s classification tables being updated in
        lockstep is a schema-drift bug (`dispatch_pairs` raises), never a
        silently-ignored/exempted base -- caught per-leg (B6), never
        crashing the whole merge for every OTHER config.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            report = jammi_fs({"ln": (9, 0), "geglu": (3, 0), "attention_block": (3, 0), "lora_linear": (3, 0)})
            report["tiers"]["finetune_step"]["mystery_kernel_fused_dispatches"] = 5
            report["tiers"]["finetune_step"]["mystery_kernel_eager_dispatches"] = 0
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged, table = self.run_merge(raw_dir)
        proof = merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"]
        self.assertIsInstance(proof, str)
        self.assertIn("ERROR", proof)
        self.assertIn("mystery_kernel", proof)
        self.assertIn("ERROR", table)
        # Advisory (iv): an errored proof is now INVALID, and INVALID gates
        # the sweep's exit code -- unlike B6's "one bad leg must not abort
        # the merge for another config" guarantee (which is about NOT
        # crashing/dropping the other config's row, still true here since
        # there is only one config in this fixture), the FINAL exit code is
        # allowed -- expected -- to reflect this config's own bad leg.
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)


_CLEAN_YES_DISPATCHES = {
    "ln": (9, 0),
    "rope": (0, 0),
    "softmax": (0, 0),
    "geglu": (3, 0),
    "lora_epilogue": (0, 0),
    "lora_linear": (3, 0),
    "attention_block": (3, 0),
}


class LegPremiseCheckTests(unittest.TestCase):
    """Fold-in this round (the lead's own adjacent probe on this PR): before
    this fix, `ab_merge.py` merged jammi-vs-torch legs with NO premise-
    identity check at all -- identity was "by construction" of
    `finetune_ab.sh`'s own matched CLI flags, an ASSUMPTION never a checked
    RECORD in the merged artifact. Every test here uses `_CLEAN_YES_DISPATCHES`
    (the same dispatch shape `test_exclusive_pair_yes` uses) so `fused_proof`
    itself stays `True` -- isolating the leg-premise check as the ONLY
    possible source of an `INVALID` verdict in these fixtures.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        return rc, merged

    def test_matching_premise_across_all_four_legs_is_not_invalid(self):
        """Positive control: the check above must not false-fail a
        genuinely matching sweep -- otherwise `test_exclusive_pair_yes` and
        every other existing fixture test would have been a false negative
        waiting to happen.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertFalse(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_seed_mismatch_between_jammi_and_torch_legs_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES, seed=42))
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(seed=999))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("seed" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertIn("leg premise mismatch", cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_batch_mismatch_between_jammi_and_torch_legs_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES, batch=8))
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(batch=64))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("batch" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_seed_missing_from_jammi_leg_is_invalid(self):
        """A jammi binary built BEFORE this round's `FinetuneStepTier::seed`
        field lands here -- the field is simply absent, not present-and-
        wrong. Must refuse just as loudly as a value mismatch, never
        silently skip the check because one side has nothing to compare.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            report = jammi_fs(_CLEAN_YES_DISPATCHES, seed=None)  # jammi_fs's None-deletes-the-key convention
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("seed" in v and "missing" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_backbone_dtype_legacy_spelling_is_not_a_violation_shared_canonicalizer(self):
        """Proves the SHARED `identity_fields.canonicalize_identity_field`
        is actually wired in here, not a second copy: torch's legacy
        CLI-flag spelling `fp32` (see `identity_fields.py`'s own doc) must
        canonicalize against jammi's `f32` here exactly as it does in
        `compare_grad_oracle.py`.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, backbone_dtype="f32"),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(backbone_dtype="fp32"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(
            [v for v in cfg["leg_premise_violations"] if "backbone_dtype" in v], [],
            f"leg_premise_violations={cfg['leg_premise_violations']!r}",
        )

    def test_backbone_dtype_genuinely_different_is_still_a_violation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, backbone_dtype="bf16"),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(backbone_dtype="f32"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("backbone_dtype" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))

    def test_provenance_recorded_for_both_legs(self):
        """torch's `attn_requested`/`attn_implementation` pair and jammi's
        14 dispatch counters, recorded (never compared) in the merged row.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["provenance"]["torch"]["torch_attn_requested"], "sdpa")
        self.assertEqual(cfg["provenance"]["torch"]["torch_attn_implementation"], "sdpa")
        self.assertIsNone(cfg["provenance"]["jammi"]["torch_attn_requested"])
        self.assertIn("ln_fused_dispatches", cfg["provenance"]["jammi"]["jammi_dispatch_counters"])
        self.assertEqual(cfg["provenance"]["jammi"]["jammi_dispatch_counters"]["ln_fused_dispatches"], 9)
        self.assertIsNone(cfg["provenance"]["torch"]["jammi_dispatch_counters"])

    def test_no_ok_leg_on_either_side_skips_the_check_without_crashing(self):
        """When neither side has an OK leg to compare, the check reports
        `None` (checked=nothing), never an empty-list false claim of "no
        violations found" -- and must not crash the merge for this config.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-fail", "jammi-eager", exit_code=1, stderr="boom")
            write_leg(raw_dir, "b8-s128-fail", "jammi-fused", exit_code=1, stderr="boom")
            write_leg(raw_dir, "b8-s128-fail", "torch-eager", exit_code=1, stderr="boom")
            write_leg(raw_dir, "b8-s128-fail", "torch-sdpa", exit_code=1, stderr="boom")
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-fail"]
        self.assertIsNone(cfg["leg_premise_violations"])
        self.assertIsNone(cfg["leg_premise_checked_legs"])

    def test_steps_measured_mismatch_between_legs_is_invalid(self):
        """ROUND-4 AUDIT REPRODUCTION: two legs measured at a DIFFERENT step
        count (e.g. `--steps 20` vs `--steps 5`, a mismatched per-leg
        override) used to still merge to a "clean" ratio and PASS verdict --
        `steps_measured` is recorded on BOTH sides already but was never
        compared.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, steps_measured=20),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(steps_measured=5))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("steps_measured" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_lora_alpha_mismatch_between_legs_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, lora_alpha=32.0),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(lora_alpha=16.0))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("lora_alpha" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))

    def test_margin_mismatch_between_legs_is_invalid(self):
        """jammi hardcodes `margin=0.3` (no CLI flag) -- an operator running
        the torch leg with `--margin` overridden away from the matching
        default is exactly the case this field exists to catch: the two
        legs would then be minimizing a DIFFERENT loss.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES))
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(margin=0.5))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("margin" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))

    def test_checkpoint_weights_sha256_mismatch_between_legs_is_invalid(self):
        """Two legs pointed at DIFFERENT `--model-dir` checkpoints -- the
        same base-checkpoint content-identity check `grad_oracle.rs`'s
        determinant table already covers, now on this tier too.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, checkpoint_weights_sha256="b" * 64),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(checkpoint_weights_sha256="c" * 64))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("checkpoint_weights_sha256" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))

    def test_identity_field_present_but_null_on_both_legs_is_invalid(self):
        """ROUND-4 AUDIT REPRODUCTION: present-but-`null` (`None` in the
        fixture dict, matching a JSON `null` — e.g. `serde_json` serializing
        a NaN `lora_alpha`) on BOTH legs used to compare `None == None` and
        silently PASS, the same class `compare_grad_oracle.py`'s own fix
        this round closes on the grad-oracle side. `jammi_fs`'s existing
        `None`-deletes-the-key convention cannot express "present but
        null" (it removes the key entirely, testing ABSENCE, not nullness)
        -- this test sets the finetune_step sub-block key directly instead.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            jammi_report = jammi_fs(_CLEAN_YES_DISPATCHES)
            jammi_report["tiers"]["finetune_step"]["lora_alpha"] = None
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            torch_report = torch_fs()
            torch_report["args"]["lora_alpha"] = None
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("lora_alpha" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))

    def test_max_grad_norm_absent_on_both_legs_is_not_a_violation(self):
        """CONTRAST with `test_identity_field_present_but_null_on_both_legs_
        is_invalid` above: `max_grad_norm` is NOT a `FINETUNE_IDENTITY_
        FIELDS` member precisely because `None`/absent on BOTH legs is the
        EVERYDAY, legitimate premise (neither leg clips) — unlike
        `lora_alpha`, this must NOT be flagged. Neither fixture sets
        `max_grad_norm` here, matching `finetune_ab.sh`'s own sweep today
        (which passes `--max-grad-norm` to neither leg).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(
            [v for v in cfg["leg_premise_violations"] if "clip setting" in v], [],
            f"leg_premise_violations={cfg['leg_premise_violations']!r}",
        )
        self.assertFalse(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_max_grad_norm_matching_on_both_legs_is_not_a_violation(self):
        """Both legs clip at the SAME `max_grad_norm` (the shipped trainer's
        own default) — the premise a real `--max-grad-norm 1.0` sweep leg
        pair is supposed to establish.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, max_grad_norm=1.0),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(max_grad_norm=1.0))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(
            [v for v in cfg["leg_premise_violations"] if "clip setting" in v], [],
            f"leg_premise_violations={cfg['leg_premise_violations']!r}",
        )
        self.assertFalse(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_jammi_clipped_torch_unclipped_is_invalid(self):
        """Class-census finding (ledger row 215, addition #3): before
        `torch_finetune_step.py` had a `--max-grad-norm` flag, this exact
        premise mismatch — jammi clipping, torch's reference NOT clipping —
        was UNCATCHABLE (the field did not exist on the torch side at all,
        so there was nothing to compare against). This is that regression,
        reproduced and asserted INVALID now that the field exists on both.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, max_grad_norm=1.0),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())  # max_grad_norm absent -> None
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("clip setting" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_max_grad_norm_different_values_on_both_legs_is_invalid(self):
        """Both legs clip, but at DIFFERENT norms — a subtler mismatch than
        one-clips-one-doesn't, still a different computation on each side.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}))
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, max_grad_norm=1.0),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(max_grad_norm=0.5))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("clip setting" in v for v in cfg["leg_premise_violations"]))
        self.assertTrue(cfg["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)


class LoraInitProvenanceTests(unittest.TestCase):
    """B4: `--lora-init` is overridable, and the merged report records
    which init each side actually used.
    """

    def test_default_lora_init_provenance_is_peft(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"ln": (9, 0)})
            out_dir = tempfile.mkdtemp()
            ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
            with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(merged["lora_init"]["torch"], "peft")
        self.assertIn("ZerosB", merged["lora_init"]["jammi"])

    def test_overridden_lora_init_provenance_is_recorded(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"ln": (9, 0)})
            out_dir = tempfile.mkdtemp()
            ab_merge.main([raw_dir, out_dir, "20", "5", "0.9", "jammi"])
            with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(merged["lora_init"]["torch"], "jammi")


class LossPrecisionTests(unittest.TestCase):
    """B5: losses are bf16-sourced; the table must not print more decimal
    digits than the dtype carries.
    """

    def test_fmt_loss_uses_three_decimals_not_four(self):
        self.assertEqual(ab_merge.fmt_loss(0.304601), "0.305")
        self.assertNotEqual(ab_merge.fmt(0.304601), ab_merge.fmt_loss(0.304601))

    def test_bf16_ulp_constant_matches_the_documented_figure(self):
        # 2**-9 == 0.001953125, the figure both finetune_step.rs's `losses`
        # field doc and torch_finetune_step.py's `loss_note` state.
        self.assertAlmostEqual(ab_merge.BF16_LOSS_ULP_NEAR_0P3, 0.001953125, places=9)

    def test_table_caveat_mentions_ulp(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"ln": (9, 0)})
            out_dir = tempfile.mkdtemp()
            ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
            with open(os.path.join(out_dir, "finetune_ab_table.txt")) as fh:
                table = fh.read()
        self.assertIn("0.00195", table)


class EmptyRawDirTests(unittest.TestCase):
    def test_no_leg_output_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
