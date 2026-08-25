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
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)

    def test_all_zero_no(self):
        """Every single pair reads (0, 0) -- a schema regression that
        dropped every counter, or a config that dispatched nothing at all.
        NOT vacuously True.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)

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
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)

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

        self.assertEqual(rc, 0, "one bad leg must not abort the whole merge (B6)")
        broken_proof = merged["configs"]["b8-s128-solo"]["jammi_fused_dispatch_proof"]
        self.assertIsInstance(broken_proof, str)
        self.assertIn("ERROR", broken_proof)
        self.assertIn("softmax_eager_dispatches", broken_proof)
        self.assertIs(merged["configs"]["b8-s128-healthy"]["jammi_fused_dispatch_proof"], True)
        # The error is visible in the printed table too, not just the JSON.
        self.assertIn("ERROR", table)

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
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)

    def test_vanished_site_case_only_lora_epilogue_positive(self):
        """B2's other named example: only `lora_epilogue` reads (1, 0),
        everything else including `ln` reads (0, 0). Must be NO for the
        same reason as the attention_block-only case above.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"lora_epilogue": (1, 0)})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertEqual(rc, 0)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)

    def test_no_dispatch_pairs_at_all_reads_false_not_none(self):
        """`fused_proof(None)` (the leg itself did not run -- MISSING/FAIL)
        is `None`; a leg that DID run OK but somehow carries an empty pair
        list is `False`, never treated the same as "did not run".
        """
        self.assertIsNone(ab_merge.fused_proof(None))
        self.assertIs(ab_merge.fused_proof({"dispatch_pairs": []}), False)


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
