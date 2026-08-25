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
        "backbone_dtype": "bf16",
        "batch": 8,
        "seq": 128,
        "lora_rank": 16,
        "lora_dropout": 0.0,
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


def torch_fs(**overrides):
    fs = {
        "device": "cpu",
        "backbone_dtype": "bf16",
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
    return {"tool": "torch_finetune_step", "finetune_step": fs}


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
