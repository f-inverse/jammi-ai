#!/usr/bin/env python3
"""Hermetic unit tests for `lora_bias_ab_merge.py` -- drives the REAL
functions (`dispatch_violations`, `apply_identity_checks`, `compute_buckets`,
`compute_model_verdict`, and `main` itself at least once) against synthetic
raw-leg fixtures built in a throwaway temp dir, never a hand-rolled
re-implementation of the merge math.

No GPU, no network, no jammi-bench binary. `python3 -m unittest
ci/scripts/perf/test_lora_bias_ab_merge.py` (or run directly).
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_bias_ab_merge as m  # noqa: E402
from identity_fields import FINETUNE_RUN_IDENTITY_FIELDS  # noqa: E402

BERT_MODULES = ["query", "key", "value", "dense"]


def base_identity(batch, width, dtype):
    d = {
        "seed": 42,
        "batch": batch,
        "seq": width,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "margin": None,
        "target_modules": BERT_MODULES,
        "layers_to_transform": None,
        "backbone_dtype": dtype,
        "checkpoint_config_sha256": "c" * 64,
        "checkpoint_weights_sha256": "w" * 64,
        "checkpoint_weights_size_bytes": 1234,
        "max_grad_norm": None,
        "warmup": None,
        "row_lengths": None,
        "epochs": 1,
        "lr": 0.0002,
        "schedule": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "grad_accum": 1,
        "validation_fraction": 0.0,
        "train_pairs_file_sha256": "t" * 64,
        "heldout_ids_sha256": "h" * 64,
        "heldout_pairs_sha256": "p" * 64,
        "heldout_batch_partition_sha256": "b" * 64,
        "embedding_loss": "mnrl",
        "temperature": 20.0,
        "matryoshka_dims": [],
        "early_stopping_patience": 10000,
        "early_stopping_metric": "train_loss",
        "eval_cadence": 1,
    }
    assert set(d.keys()) == set(FINETUNE_RUN_IDENTITY_FIELDS), (
        "test fixture's identity dict has drifted from FINETUNE_RUN_IDENTITY_FIELDS "
        f"-- missing {set(FINETUNE_RUN_IDENTITY_FIELDS) - set(d.keys())}, "
        f"extra {set(d.keys()) - set(FINETUNE_RUN_IDENTITY_FIELDS)}"
    )
    return d


def base_provenance(steps, requested, fired):
    return {
        "arm": "fused",
        "device_name": "cpu",
        "kernels_disabled_requested": list(requested),
        "kernels_disabled_fired": list(fired),
        "flash_compiled": False,
        "build_features": [],
        "attention_arm": "fused",
        "split_rule": "positional_fraction_split",
        "batched_forward": True,
        "steps_measured": steps,
    }


def make_tier(*, batch, width, dtype, steps, requested, fired, wall, fused_disp, eager_disp,
              identity_overrides=None, ln_fused_disp=1, ln_eager_disp=0):
    tier = {}
    tier.update(base_identity(batch, width, dtype))
    if identity_overrides:
        tier.update(identity_overrides)
    tier.update(base_provenance(steps, requested, fired))
    tier["train_run_wall_s"] = wall
    tier["lora_linear_fused_dispatches"] = fused_disp
    tier["lora_linear_eager_dispatches"] = eager_disp
    # A real report always carries the `ln_*` counters too (#460) --
    # defaulting to fused=1/eager=0 (the ModernBERT-already-fused shape)
    # keeps every pre-#460 `lora_linear` fixture below unaffected, since
    # `dispatch_violations(row, tier)` (its 2-arg default) never reads
    # these fields at all.
    tier["ln_fused_dispatches"] = ln_fused_disp
    tier["ln_eager_dispatches"] = ln_eager_disp
    return tier


# Op tables mirroring `lora_bias_ab_merge.OPS`/`lora_bias_ab.sh`'s own AB_OP
# table (#460) -- kept as a small local literal (never imported from `m.OPS`)
# so this fixture module stays an independent check on the module under
# test, not a mirror that could drift in lockstep with a bug in it.
_OP_DISABLE_KEY = {"lora_linear": "lora_linear_fused", "ln": "layer_norm_fused"}
_OP_EAGER_ARM = {"lora_linear": "lora_eager", "ln": "ln_eager"}


def add_leg(
    root,
    rows,
    *,
    model,
    shape,
    arm,
    steps,
    repeat,
    wall,
    fused_disp,
    eager_disp,
    requested,
    fired,
    extra_disable=(),
    batch=8,
    width=512,
    dtype="f32",
    identity_overrides=None,
    status="ok",
    reason="",
    write_report=True,
    ab_op="lora_linear",
    ln_fused_disp=1,
    ln_eager_disp=0,
):
    leg_id = f"{model}-{shape}-{arm}-{steps}-r{repeat}"
    tier = make_tier(
        batch=batch,
        width=width,
        dtype=dtype,
        steps=steps,
        requested=requested,
        fired=fired,
        wall=wall,
        fused_disp=fused_disp,
        eager_disp=eager_disp,
        identity_overrides=identity_overrides,
        ln_fused_disp=ln_fused_disp,
        ln_eager_disp=ln_eager_disp,
    )
    report_path = None
    if write_report:
        rel = f"raw/{leg_id}.json"
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            json.dump({"tool": "test", "tiers": {"finetune_run": tier}}, f)
        report_path = rel
    row = {
        "leg_id": leg_id,
        "model": model,
        "shape": shape,
        "batch": batch,
        "width": width,
        "dtype": dtype,
        "arm": arm,
        "steps": steps,
        "repeat": repeat,
        "ab_op": ab_op,
        "env": {"JAMMI_KERNELS_STRICT": "1", "JAMMI_KERNELS_DISABLE": ",".join(requested)},
        "extra_disable": list(extra_disable),
        "argv": [],
        "rc": 0 if status == "ok" else 1,
        "wall_s": wall,
        "status": status,
        "reason": reason,
        "git_sha": "0" * 40,
        "box": "test-box",
        "dry_run": True,
        "report_path": report_path,
        "stderr_path": f"raw/{leg_id}.stderr",
    }
    rows.append(row)
    return tier


def add_cell(root, rows, model, shape, arm, *, repeats, walls_n, walls_m, batch, width, dtype,
             extra_disable=(), ab_op="lora_linear"):
    """Adds `repeats` complete (N, M) leg pairs for one (model, shape, arm).

    `ab_op="lora_linear"` (default) is byte-for-byte the pre-#460 fixture
    shape. `ab_op="ln"` forces `layer_norm_fused` eager instead, on the
    `ln_eager` arm, and PINS `lora_linear` fused>0/eager==0 on EVERY leg
    regardless of arm -- the same cross-op invariant `lora_bias_ab.sh`'s own
    fake-bench stub and the real admission lattice both hold for an
    `AB_OP=ln` sweep.
    """
    eager_arm_label = _OP_EAGER_ARM[ab_op]
    disable_key = _OP_DISABLE_KEY[ab_op]
    is_eager_arm = arm == eager_arm_label
    requested = [disable_key] if is_eager_arm else []
    requested = list(requested) + list(extra_disable)
    fired = [disable_key] if is_eager_arm else []
    if ab_op == "lora_linear":
        fused_disp = 0 if is_eager_arm else 1
        eager_disp = 1 if is_eager_arm else 0
        ln_fused_disp, ln_eager_disp = 1, 0
    else:
        # `ln`: lora_linear stays fused on EVERY leg of this sweep (the
        # cross-check invariant), regardless of this leg's own arm.
        fused_disp, eager_disp = 1, 0
        ln_fused_disp = 0 if is_eager_arm else 1
        ln_eager_disp = 1 if is_eager_arm else 0
    for r in range(1, repeats + 1):
        add_leg(
            root, rows, model=model, shape=shape, arm=arm, steps=m.STEPS_N, repeat=r,
            wall=walls_n[r - 1], fused_disp=fused_disp, eager_disp=eager_disp,
            requested=requested, fired=fired, extra_disable=extra_disable,
            batch=batch, width=width, dtype=dtype, ab_op=ab_op,
            ln_fused_disp=ln_fused_disp, ln_eager_disp=ln_eager_disp,
        )
        add_leg(
            root, rows, model=model, shape=shape, arm=arm, steps=m.STEPS_M, repeat=r,
            wall=walls_m[r - 1], fused_disp=fused_disp, eager_disp=eager_disp,
            requested=requested, fired=fired, extra_disable=extra_disable,
            batch=batch, width=width, dtype=dtype, ab_op=ab_op,
            ln_fused_disp=ln_fused_disp, ln_eager_disp=ln_eager_disp,
        )


def write_manifest(root, rows):
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(rows, f)


class DispatchProofTests(unittest.TestCase):
    def test_fused_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        self.assertEqual(m.dispatch_violations(row, tier), [])

    def test_control_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "control", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 3, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        self.assertEqual(m.dispatch_violations(row, tier), [])

    def test_eager_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "lora_eager", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 0, "lora_linear_eager_dispatches": 5,
                "kernels_disabled_requested": ["lora_linear_fused"],
                "kernels_disabled_fired": ["lora_linear_fused"]}
        self.assertEqual(m.dispatch_violations(row, tier), [])

    def test_fused_leg_with_eager_dispatches_is_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 1,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_fused_leg_naming_lora_linear_fused_in_requested_is_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": ["lora_linear_fused"], "kernels_disabled_fired": ["lora_linear_fused"]}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_eager_leg_with_fused_dispatches_is_a_violation(self):
        row = {"leg_id": "x", "arm": "lora_eager", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 5,
                "kernels_disabled_requested": ["lora_linear_fused"], "kernels_disabled_fired": ["lora_linear_fused"]}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_eager_leg_missing_from_requested_is_a_violation(self):
        row = {"leg_id": "x", "arm": "lora_eager", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 0, "lora_linear_eager_dispatches": 5,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_eager_leg_requested_but_never_fired_is_a_violation(self):
        # Requested-but-dropped-environment class: JAMMI_KERNELS_DISABLE named
        # the op but this process's own admission lattice never actually saw
        # a live call disabled by it.
        row = {"leg_id": "x", "arm": "lora_eager", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 0, "lora_linear_eager_dispatches": 5,
                "kernels_disabled_requested": ["lora_linear_fused"], "kernels_disabled_fired": []}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_unrecognized_arm_is_a_violation(self):
        row = {"leg_id": "x", "arm": "bogus", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("unrecognized arm" in v for v in vs), vs)

    def test_asymmetric_extra_disable_is_a_violation(self):
        # The leg's own env named an extra op the manifest never recorded as
        # this leg's declared EXTRA_DISABLE -- an asymmetric
        # JAMMI_KERNELS_DISABLE that silently changed which op this arm's
        # numbers describe.
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": ["some_other_op"], "kernels_disabled_fired": ["some_other_op"]}
        vs = m.dispatch_violations(row, tier)
        self.assertTrue(any("asymmetric" in v for v in vs), vs)

    def test_symmetric_extra_disable_is_not_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": ["some_other_op"]}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": ["some_other_op"], "kernels_disabled_fired": ["some_other_op"]}
        self.assertEqual(m.dispatch_violations(row, tier), [])


class WallViolationTests(unittest.TestCase):
    def test_finite_positive_is_fine(self):
        self.assertEqual(m.wall_violations("x", {"train_run_wall_s": 1.5}), [])

    def test_nan_is_invalid(self):
        vs = m.wall_violations("x", {"train_run_wall_s": float("nan")})
        self.assertTrue(vs)

    def test_inf_is_invalid(self):
        vs = m.wall_violations("x", {"train_run_wall_s": float("inf")})
        self.assertTrue(vs)

    def test_zero_is_invalid(self):
        vs = m.wall_violations("x", {"train_run_wall_s": 0.0})
        self.assertTrue(vs)

    def test_negative_is_invalid(self):
        vs = m.wall_violations("x", {"train_run_wall_s": -1.0})
        self.assertTrue(vs)

    def test_missing_is_invalid(self):
        vs = m.wall_violations("x", {})
        self.assertTrue(vs)


class IdentityCheckTests(unittest.TestCase):
    def test_matching_identity_across_arms_is_clean(self):
        tier_a = make_tier(batch=8, width=512, dtype="f32", steps=100, requested=[], fired=[],
                            wall=1.0, fused_disp=1, eager_disp=0)
        tier_b = make_tier(batch=8, width=512, dtype="f32", steps=100,
                            requested=["lora_linear_fused"], fired=["lora_linear_fused"],
                            wall=1.2, fused_disp=0, eager_disp=1)
        legs = [
            {"row": {"leg_id": "a", "model": "bert", "shape": "s", "steps": 100}, "tier": tier_a, "violations": []},
            {"row": {"leg_id": "b", "model": "bert", "shape": "s", "steps": 100}, "tier": tier_b, "violations": []},
        ]
        m.apply_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])
        self.assertEqual(legs[1]["violations"], [])

    def test_lora_rank_mismatch_voids_both_legs(self):
        tier_a = make_tier(batch=8, width=512, dtype="f32", steps=100, requested=[], fired=[],
                            wall=1.0, fused_disp=1, eager_disp=0)
        tier_b = make_tier(batch=8, width=512, dtype="f32", steps=100,
                            requested=["lora_linear_fused"], fired=["lora_linear_fused"],
                            wall=1.2, fused_disp=0, eager_disp=1,
                            identity_overrides={"lora_rank": 16})
        legs = [
            {"row": {"leg_id": "a", "model": "bert", "shape": "s", "steps": 100}, "tier": tier_a, "violations": []},
            {"row": {"leg_id": "b", "model": "bert", "shape": "s", "steps": 100}, "tier": tier_b, "violations": []},
        ]
        m.apply_identity_checks(legs)
        self.assertTrue(any("lora_rank" in v for v in legs[0]["violations"]), legs[0]["violations"])
        self.assertTrue(any("lora_rank" in v for v in legs[1]["violations"]), legs[1]["violations"])

    def test_different_groups_never_compared(self):
        tier_a = make_tier(batch=8, width=512, dtype="f32", steps=100, requested=[], fired=[],
                            wall=1.0, fused_disp=1, eager_disp=0)
        tier_b = make_tier(batch=32, width=64, dtype="bf16", steps=100, requested=[], fired=[],
                            wall=1.0, fused_disp=1, eager_disp=0)
        legs = [
            {"row": {"leg_id": "a", "model": "bert", "shape": "wire", "steps": 100}, "tier": tier_a, "violations": []},
            {"row": {"leg_id": "b", "model": "bert", "shape": "chapter", "steps": 100}, "tier": tier_b, "violations": []},
        ]
        m.apply_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])
        self.assertEqual(legs[1]["violations"], [])


class DifferencingArithmeticTests(unittest.TestCase):
    def test_known_values_produce_the_expected_s_per_step(self):
        tier_n = make_tier(batch=8, width=512, dtype="f32", steps=100, requested=[], fired=[],
                            wall=1.0, fused_disp=1, eager_disp=0)
        tier_m = make_tier(batch=8, width=512, dtype="f32", steps=600, requested=[], fired=[],
                            wall=51.0, fused_disp=1, eager_disp=0)
        legs = [
            {"row": {"leg_id": "n", "model": "bert", "shape": "wire", "arm": "fused", "steps": 100, "repeat": 1},
             "tier": tier_n, "violations": []},
            {"row": {"leg_id": "m", "model": "bert", "shape": "wire", "arm": "fused", "steps": 600, "repeat": 1},
             "tier": tier_m, "violations": []},
        ]
        buckets = m.compute_buckets(legs)
        cell = buckets[("measurement", "bert", "wire")]
        self.assertTrue(cell["valid"], cell["reasons"])
        stats = cell["per_arm"]["fused"]
        self.assertEqual(stats["n"], 1)
        self.assertAlmostEqual(stats["median"], (51.0 - 1.0) / 500, places=9)


class EndToEndVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, rows):
        write_manifest(self.root, rows)
        legs = m.build_legs(self.root, rows)
        m.apply_identity_checks(legs)
        buckets = m.compute_buckets(legs)
        return legs, buckets

    def test_activate(self):
        rows = []
        # control at wire: median ~0.102
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[52.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32")
        # wire fused: median ~0.100
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 50.0, 51.0],
                 batch=8, width=512, dtype="f32")
        # wire eager: median ~0.130 -> gain = 1 - 0.100/0.130 ~ 0.2308
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[66.0, 65.0, 65.0],
                 batch=8, width=512, dtype="f32")
        # chapter shape: also present, doesn't matter which way
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.0, 25.0, 25.0],
                 batch=32, width=64, dtype="bf16")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[36.0, 35.0, 35.0],
                 batch=32, width=64, dtype="bf16")
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertEqual(verdict["verdict"], "ACTIVATE", verdict)

    def test_neutral(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32")
        # eager only marginally slower than fused -> gain below the 0.05 bar
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[52.0, 52.0, 52.0],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.0, 26.0, 26.0],
                 batch=32, width=64, dtype="bf16")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.5, 26.5, 26.5],
                 batch=32, width=64, dtype="bf16")
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertEqual(verdict["verdict"], "NEUTRAL", verdict)

    def test_regression(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32")
        # wire fused SLOWER than eager -- a genuine regression, well past
        # any plausible floor.
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[201.0, 201.0, 201.0],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[101.0, 101.0, 101.0],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.0, 26.0, 26.0],
                 batch=32, width=64, dtype="bf16")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.5, 26.5, 26.5],
                 batch=32, width=64, dtype="bf16")
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertEqual(verdict["verdict"], "REGRESSION", verdict)

    def test_fused_slower_by_less_than_floor_is_neutral_not_regression(self):
        rows = []
        # control noise floor: control median 0.104 vs wire fused median
        # 0.100 -> floor = |0.100-0.104|/0.100 = 0.04.
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[53.0, 53.0, 53.0],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32")
        # wire eager SLIGHTLY faster than fused (gain slightly negative) --
        # |gain| well under the 0.04 floor, so this must read NEUTRAL, not
        # REGRESSION: gain = 1 - 51/50.5 ~ -0.0099, floor ~0.04.
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[50.5, 50.5, 50.5],
                 batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.0, 26.0, 26.0],
                 batch=32, width=64, dtype="bf16")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "lora_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.1, 26.1, 26.1],
                 batch=32, width=64, dtype="bf16")
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertGreater(verdict["floor"], 0.03)
        self.assertLess(verdict["gain_by_shape"][m.WIRE_SHAPE], 0.0)
        self.assertEqual(verdict["verdict"], "NEUTRAL", verdict)

    def test_missing_m_leg_voids_the_bucket(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        # Only the N-step run for wire/fused -- no M-step run at all.
        add_leg(self.root, rows, model="bert", shape=m.WIRE_SHAPE, arm="fused", steps=m.STEPS_N,
                repeat=1, wall=1.0, fused_disp=1, eager_disp=0, requested=[], fired=[])
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertEqual(verdict["verdict"], "INVALID", verdict)
        cell = buckets[("measurement", "bert", m.WIRE_SHAPE)]
        self.assertFalse(cell["valid"])
        self.assertTrue(any("missing its N=100 or M=600 leg" in r for r in cell["reasons"]), cell["reasons"])

    def test_rc_nonzero_leg_voids_its_bucket(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_leg(self.root, rows, model="bert", shape=m.WIRE_SHAPE, arm="fused", steps=m.STEPS_N,
                repeat=1, wall=1.0, fused_disp=1, eager_disp=0, requested=[], fired=[],
                status="invalid", reason="finetune-run exited 1", write_report=False)
        add_leg(self.root, rows, model="bert", shape=m.WIRE_SHAPE, arm="fused", steps=m.STEPS_M,
                repeat=1, wall=51.0, fused_disp=1, eager_disp=0, requested=[], fired=[])
        legs, buckets = self._run(rows)
        cell = buckets[("measurement", "bert", m.WIRE_SHAPE)]
        self.assertFalse(cell["valid"])
        self.assertTrue(any("finetune-run exited 1" in r for r in cell["reasons"]), cell["reasons"])

    def test_nan_wall_voids_its_bucket(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_leg(self.root, rows, model="bert", shape=m.WIRE_SHAPE, arm="fused", steps=m.STEPS_N,
                repeat=1, wall=float("nan"), fused_disp=1, eager_disp=0, requested=[], fired=[])
        add_leg(self.root, rows, model="bert", shape=m.WIRE_SHAPE, arm="fused", steps=m.STEPS_M,
                repeat=1, wall=51.0, fused_disp=1, eager_disp=0, requested=[], fired=[])
        legs, buckets = self._run(rows)
        cell = buckets[("measurement", "bert", m.WIRE_SHAPE)]
        self.assertFalse(cell["valid"])

    def test_asymmetric_extra_disable_across_arms_voids_the_bucket(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32",
                 extra_disable=["some_op"])
        # Eager arm never got the same extra disable -- asymmetric.
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=1, walls_n=[1.0], walls_m=[66.0], batch=8, width=512, dtype="f32",
                 extra_disable=())
        legs, buckets = self._run(rows)
        cell = buckets[("measurement", "bert", m.WIRE_SHAPE)]
        self.assertFalse(cell["valid"])
        self.assertTrue(any("inconsistent extra_disable" in r for r in cell["reasons"]), cell["reasons"])

    def test_control_series_invalid_makes_the_model_invalid(self):
        rows = []
        # No control legs at all.
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=1, walls_n=[1.0], walls_m=[66.0], batch=8, width=512, dtype="f32")
        legs, buckets = self._run(rows)
        verdict = m.compute_model_verdict("bert", buckets)
        self.assertEqual(verdict["verdict"], "INVALID", verdict)
        self.assertIn("control", verdict["reason"])

    def test_end_to_end_main_writes_a_schema_shaped_artifact(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=1, walls_n=[1.0], walls_m=[66.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[26.0], batch=32, width=64, dtype="bf16")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "lora_eager",
                 repeats=1, walls_n=[1.0], walls_m=[36.0], batch=32, width=64, dtype="bf16")
        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        rc = m.main([self.root, out_path, "--git-sha", "a" * 40, "--box", "unit-test-box"])
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["git_sha"], "a" * 40)
        self.assertEqual(artifact["box"], "unit-test-box")
        self.assertEqual(artifact["producer"]["path"], "ci/scripts/perf/lora_bias_ab.sh")
        self.assertEqual(artifact["producer"]["kind"], "script")
        self.assertIn("status", artifact)
        self.assertIn("bert", artifact["notes"]["verdicts"])
        self.assertIsInstance(artifact["legs"], list)
        self.assertTrue(len(artifact["legs"]) > 0)
        # distilbert never appeared in this fixture -- it must still be
        # reported, INVALID, never silently omitted from the notes.
        self.assertIn("distilbert", artifact["notes"]["verdicts"])
        self.assertEqual(artifact["notes"]["verdicts"]["distilbert"]["verdict"], "INVALID")


class AbOpLnTests(unittest.TestCase):
    """#460: `AB_OP=ln` -- the per-op generalization's second table entry.
    Dispatch-proof shape, the `ln_eager` arm label, the cross-op invariant
    pinning `lora_linear` fused on every leg of an `ln` sweep, and the
    artifact's own `landing_rule` note. Every `lora_linear`-op test above
    this class is untouched and still passes with `dispatch_violations`'s
    2-arg default / `compute_model_verdict`'s 2-arg default / `main`'s
    manifest-inferred `ab_op` default."""

    def test_ln_fused_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 1, "ln_eager_dispatches": 0,
            "lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["ln"]), [])

    def test_ln_eager_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "ln_eager", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 0, "ln_eager_dispatches": 4,
            "lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
            "kernels_disabled_requested": ["layer_norm_fused"],
            "kernels_disabled_fired": ["layer_norm_fused"],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["ln"]), [])

    def test_ln_eager_leg_naming_lora_linear_fused_is_still_clean(self):
        # An AB_OP=ln leg's own requested set only ever names
        # layer_norm_fused -- lora_linear_fused never appears in it (the
        # cross-check below is about the COUNTER, not about requested).
        row = {"leg_id": "x", "arm": "ln_eager", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 0, "ln_eager_dispatches": 4,
            "lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
            "kernels_disabled_requested": ["layer_norm_fused"],
            "kernels_disabled_fired": ["layer_norm_fused"],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["ln"]), [])

    def test_cross_check_lora_linear_not_fused_on_ln_fused_arm_is_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 1, "ln_eager_dispatches": 0,
            "lora_linear_fused_dispatches": 0, "lora_linear_eager_dispatches": 3,
            "kernels_disabled_requested": ["lora_linear_fused"],
            "kernels_disabled_fired": ["lora_linear_fused"],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["ln"])
        self.assertTrue(any("cross-op invariant failed" in v for v in vs), vs)

    def test_cross_check_lora_linear_not_fused_on_ln_eager_arm_is_a_violation(self):
        row = {"leg_id": "x", "arm": "ln_eager", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 0, "ln_eager_dispatches": 2,
            "lora_linear_fused_dispatches": 0, "lora_linear_eager_dispatches": 1,
            "kernels_disabled_requested": ["layer_norm_fused", "lora_linear_fused"],
            "kernels_disabled_fired": ["layer_norm_fused", "lora_linear_fused"],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["ln"])
        self.assertTrue(any("cross-op invariant failed" in v for v in vs), vs)

    def test_cross_check_missing_lora_linear_fields_is_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "ln_fused_dispatches": 1, "ln_eager_dispatches": 0,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["ln"])
        self.assertTrue(any("cross-op invariant fields missing" in v for v in vs), vs)

    def test_ab_op_lora_linear_default_never_runs_the_cross_check(self):
        # OPS["lora_linear"]'s own cross_check is None -- a leg with no ln_*
        # fields at all (every pre-#460 fixture) must still be clean.
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {"lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
                "kernels_disabled_requested": [], "kernels_disabled_fired": []}
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["lora_linear"]), [])
        self.assertEqual(m.dispatch_violations(row, tier), [])

    def test_op_cfg_for_row_defaults_missing_ab_op_to_lora_linear(self):
        cfg, name = m.op_cfg_for_row({"leg_id": "x"})
        self.assertEqual(name, "lora_linear")
        self.assertIs(cfg, m.OPS["lora_linear"])

    def test_op_cfg_for_row_unrecognized_ab_op_returns_none(self):
        cfg, name = m.op_cfg_for_row({"leg_id": "x", "ab_op": "bogus_op"})
        self.assertIsNone(cfg)
        self.assertEqual(name, "bogus_op")


class AbOpAttnGeluTests(unittest.TestCase):
    """#462/#463: the multi-key `disable_keys`/`expected_shapes` shape and
    the list-form `cross_check` (strong + pair_identity)."""

    def test_attn_fused_leg_clean_with_softmax_absorbed_00(self):
        # The fused-arm (0, 0) softmax shape: softmax is ABSORBED into the
        # block op while the block admits -- never independently
        # dispatched at all, not "eager".
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "attention_block_fused_dispatches": 1, "attention_block_eager_dispatches": 0,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 0,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["attn"]), [])

    def test_attn_eager_leg_clean(self):
        row = {"leg_id": "x", "arm": "attn_eager", "extra_disable": []}
        tier = {
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 3,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 3,
            "kernels_disabled_requested": ["attention_block_fused", "softmax_last_dim_fused"],
            "kernels_disabled_fired": ["attention_block_fused", "softmax_last_dim_fused"],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["attn"]), [])

    def test_attn_eager_leg_missing_softmax_key_from_requested_is_a_violation(self):
        # Only attention_block_fused named -- the block-fused arm subsumes
        # softmax, so naming softmax_last_dim_fused too is REQUIRED on this
        # ops own eager arm (disabling attention_block_fused alone would be
        # a silent no-op on softmax's own dispatch, per admission.rs's
        # "Standalone vs subsumed op keys" section).
        row = {"leg_id": "x", "arm": "attn_eager", "extra_disable": []}
        tier = {
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 3,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 3,
            "kernels_disabled_requested": ["attention_block_fused"],
            "kernels_disabled_fired": ["attention_block_fused"],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["attn"])
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_attn_eager_leg_with_softmax_fused_gt_0_is_a_red_control(self):
        # RED CONTROL: the eager arm still shows softmax_fused_dispatches
        # > 0 -- softmax did NOT actually run eager despite the disable
        # claim. Must fail, never silently pass.
        row = {"leg_id": "x", "arm": "attn_eager", "extra_disable": []}
        tier = {
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 3,
            "softmax_fused_dispatches": 1, "softmax_eager_dispatches": 2,
            "kernels_disabled_requested": ["attention_block_fused", "softmax_last_dim_fused"],
            "kernels_disabled_fired": ["attention_block_fused", "softmax_last_dim_fused"],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["attn"])
        self.assertTrue(
            any("dispatch proof failed for key 'softmax'" in v for v in vs), vs
        )

    def test_attn_multi_key_symmetry_check_clean(self):
        # Both of attn's own disable_keys plus one EXTRA_DISABLE entry,
        # symmetrically recorded -- the symmetry check subtracts attn's
        # WHOLE key set, not one key.
        row = {"leg_id": "x", "arm": "attn_eager", "extra_disable": ["some_other_op"]}
        tier = {
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 3,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 3,
            "kernels_disabled_requested": [
                "attention_block_fused", "softmax_last_dim_fused", "some_other_op",
            ],
            "kernels_disabled_fired": ["attention_block_fused", "softmax_last_dim_fused"],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["attn"]), [])

    def test_attn_multi_key_symmetry_check_asymmetric_is_a_violation(self):
        row = {"leg_id": "x", "arm": "attn_eager", "extra_disable": []}
        tier = {
            "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 3,
            "softmax_fused_dispatches": 0, "softmax_eager_dispatches": 3,
            "kernels_disabled_requested": [
                "attention_block_fused", "softmax_last_dim_fused", "some_other_op",
            ],
            "kernels_disabled_fired": ["attention_block_fused", "softmax_last_dim_fused"],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["attn"])
        self.assertTrue(any("asymmetric" in v for v in vs), vs)

    def test_gelu_fused_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "gelu_fused_dispatches": 1, "gelu_eager_dispatches": 0,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["gelu"]), [])

    def test_gelu_eager_leg_clean_is_no_violation(self):
        row = {"leg_id": "x", "arm": "gelu_eager", "extra_disable": []}
        tier = {
            "gelu_fused_dispatches": 0, "gelu_eager_dispatches": 4,
            "kernels_disabled_requested": ["gelu_erf_fused"],
            "kernels_disabled_fired": ["gelu_erf_fused"],
        }
        self.assertEqual(m.dispatch_violations(row, tier, m.OPS["gelu"]), [])

    def test_gelu_fused_leg_with_eager_dispatches_is_a_violation(self):
        row = {"leg_id": "x", "arm": "fused", "extra_disable": []}
        tier = {
            "gelu_fused_dispatches": 1, "gelu_eager_dispatches": 1,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        vs = m.dispatch_violations(row, tier, m.OPS["gelu"])
        self.assertTrue(any("dispatch proof failed" in v for v in vs), vs)

    def test_op_cfg_for_row_resolves_attn_and_gelu(self):
        cfg, name = m.op_cfg_for_row({"leg_id": "x", "ab_op": "attn"})
        self.assertEqual(name, "attn")
        self.assertIs(cfg, m.OPS["attn"])
        cfg, name = m.op_cfg_for_row({"leg_id": "x", "ab_op": "gelu"})
        self.assertEqual(name, "gelu")
        self.assertIs(cfg, m.OPS["gelu"])


class PairIdentityCheckTests(unittest.TestCase):
    """`apply_pair_identity_checks` -- the pair_identity cross-check form
    (#462): a counter family's value must read IDENTICAL between the
    fused/control arm and this op's own eager arm of the same
    (model, shape, steps, repeat) cell."""

    def _leg(self, leg_id, model, shape, steps, repeat, arm, tier):
        return {
            "row": {"leg_id": leg_id, "model": model, "shape": shape, "steps": steps,
                     "repeat": repeat, "arm": arm, "ab_op": "attn"},
            "tier": tier,
            "violations": [],
        }

    def test_matching_declined_counts_across_arms_is_clean(self):
        tier_fused = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 1200}
        tier_eager = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 1200}
        legs = [
            self._leg("f", "bert", "wire", 600, 1, "fused", tier_fused),
            self._leg("e", "bert", "wire", 600, 1, "attn_eager", tier_eager),
        ]
        m.apply_pair_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])
        self.assertEqual(legs[1]["violations"], [])

    def test_both_zero_is_a_legitimate_reading_not_a_violation(self):
        tier_fused = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 0}
        tier_eager = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 0}
        legs = [
            self._leg("f", "bert", "wire", 600, 1, "fused", tier_fused),
            self._leg("e", "bert", "wire", 600, 1, "attn_eager", tier_eager),
        ]
        m.apply_pair_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])
        self.assertEqual(legs[1]["violations"], [])

    def test_mismatched_declined_counts_voids_both_legs(self):
        tier_fused = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 1200}
        tier_eager = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 999}
        legs = [
            self._leg("f", "bert", "wire", 600, 1, "fused", tier_fused),
            self._leg("e", "bert", "wire", 600, 1, "attn_eager", tier_eager),
        ]
        m.apply_pair_identity_checks(legs)
        self.assertTrue(any("pair-identity check failed" in v for v in legs[0]["violations"]), legs[0])
        self.assertTrue(any("pair-identity check failed" in v for v in legs[1]["violations"]), legs[1])

    def test_different_cells_never_compared(self):
        tier_fused = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 1200}
        tier_eager = {"attention_block_flash_fused_dispatches": 0,
                      "attention_block_flash_declined_dispatches": 100}
        legs = [
            self._leg("f", "bert", "wire", 600, 1, "fused", tier_fused),
            self._leg("e", "bert", "chapter", 600, 1, "attn_eager", tier_eager),
        ]
        m.apply_pair_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])
        self.assertEqual(legs[1]["violations"], [])

    def test_ops_with_no_pair_identity_keys_are_untouched(self):
        legs = [
            self._leg("f", "bert", "wire", 600, 1, "fused", {}),
        ]
        legs[0]["row"]["ab_op"] = "lora_linear"
        m.apply_pair_identity_checks(legs)
        self.assertEqual(legs[0]["violations"], [])

    def test_list_form_invariant_with_one_live_and_one_absorbed_key(self):
        # A synthetic op config combining BOTH cross_check kinds in one
        # list -- the shape a real op could carry even though none of the
        # four in `OPS` today does: a "strong" entry (a LIVE key that must
        # stay fused>0/eager==0 throughout, like ln's own lora_linear
        # cross-check) alongside a "pair_identity" entry (an ABSORBED/
        # declined key that must read identically across arms, like attn's
        # own attention_block_flash). `dispatch_violations` (single-leg)
        # must apply ONLY the strong entry; `apply_pair_identity_checks`
        # (cross-leg) must apply ONLY the pair_identity entry -- each
        # ignores the other kind silently rather than mis-happing it.
        synthetic_op = {
            "disable_keys": ["synthetic_fused"],
            "counter_base": "synthetic",
            "eager_arm": "synthetic_eager",
            "expected_shapes": {
                "fused": {"synthetic": (">0", 0)},
                "synthetic_eager": {"synthetic": (0, ">0")},
            },
            "cross_check": [
                {"kind": "strong", "op": "lora_linear"},
                {
                    "kind": "pair_identity",
                    "field_base": "attention_block_flash",
                    "value_fields": ("fused_dispatches", "declined_dispatches"),
                },
            ],
        }
        # dispatch_violations: the strong entry alone applies.
        row = {"leg_id": "x", "arm": "fused", "extra_disable": [], "ab_op": "synthetic"}
        clean_tier = {
            "synthetic_fused_dispatches": 1, "synthetic_eager_dispatches": 0,
            "lora_linear_fused_dispatches": 1, "lora_linear_eager_dispatches": 0,
            "kernels_disabled_requested": [], "kernels_disabled_fired": [],
        }
        self.assertEqual(m.dispatch_violations(row, clean_tier, synthetic_op), [])
        broken_tier = dict(clean_tier)
        broken_tier["lora_linear_fused_dispatches"] = 0
        broken_tier["lora_linear_eager_dispatches"] = 1
        vs = m.dispatch_violations(row, broken_tier, synthetic_op)
        self.assertTrue(any("cross-op invariant failed" in v for v in vs), vs)

        # apply_pair_identity_checks: the pair_identity entry alone
        # applies, at the group level (dispatch_violations never touches
        # attention_block_flash at all for this synthetic op).
        legs = [
            {
                "row": {"leg_id": "sf", "model": "bert", "shape": "wire", "steps": 600,
                        "repeat": 1, "arm": "fused", "ab_op": "synthetic"},
                "tier": {"attention_block_flash_fused_dispatches": 0,
                          "attention_block_flash_declined_dispatches": 5},
                "violations": [],
            },
            {
                "row": {"leg_id": "se", "model": "bert", "shape": "wire", "steps": 600,
                        "repeat": 1, "arm": "synthetic_eager", "ab_op": "synthetic"},
                "tier": {"attention_block_flash_fused_dispatches": 0,
                          "attention_block_flash_declined_dispatches": 7},
                "violations": [],
            },
        ]
        orig_ops = dict(m.OPS)
        try:
            m.OPS["synthetic"] = synthetic_op
            m.apply_pair_identity_checks(legs)
        finally:
            m.OPS.clear()
            m.OPS.update(orig_ops)
        self.assertTrue(any("pair-identity check failed" in v for v in legs[0]["violations"]), legs[0])
        self.assertTrue(any("pair-identity check failed" in v for v in legs[1]["violations"]), legs[1])


class AbOpLnEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, rows):
        write_manifest(self.root, rows)
        legs = m.build_legs(self.root, rows)
        m.apply_identity_checks(legs)
        buckets = m.compute_buckets(legs)
        return legs, buckets

    def _ln_activate_rows(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[52.0, 51.0, 51.0],
                 batch=8, width=512, dtype="f32", ab_op="ln")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[51.0, 50.0, 51.0],
                 batch=8, width=512, dtype="f32", ab_op="ln")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "ln_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[66.0, 65.0, 65.0],
                 batch=8, width=512, dtype="f32", ab_op="ln")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "fused",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[26.0, 25.0, 25.0],
                 batch=32, width=64, dtype="bf16", ab_op="ln")
        add_cell(self.root, rows, "bert", m.CHAPTER_SHAPE, "ln_eager",
                 repeats=3, walls_n=[1.0, 1.0, 1.0], walls_m=[36.0, 35.0, 35.0],
                 batch=32, width=64, dtype="bf16", ab_op="ln")
        return rows

    def test_ln_activate_reads_the_same_as_lora_linear_would(self):
        # `add_cell`'s `ab_op="ln"` walls mirror `EndToEndVerdictTests.
        # test_activate`'s own `lora_linear` walls exactly -- the verdict
        # math is IDENTICAL across ops, only the counter base/arm label
        # differ.
        rows = self._ln_activate_rows()
        legs, buckets = self._run(rows)
        for row in rows:
            self.assertEqual(row["ab_op"], "ln", row)
        arms_seen = {leg["row"]["arm"] for leg in legs}
        self.assertEqual(arms_seen, {"fused", "ln_eager", "control"}, arms_seen)
        verdict = m.compute_model_verdict("bert", buckets, m.OPS["ln"])
        self.assertEqual(verdict["verdict"], "ACTIVATE", verdict)
        for leg in legs:
            self.assertEqual(leg["violations"], [], leg)

    def test_end_to_end_main_infers_ln_from_the_manifest_and_writes_landing_rule(self):
        rows = self._ln_activate_rows()
        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        rc = m.main([self.root, out_path, "--git-sha", "b" * 40, "--box", "unit-test-box"])
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["ab_op"], "ln")
        self.assertIn("landing_rule", artifact["notes"])
        self.assertIn("no activation bar", artifact["notes"]["landing_rule"])
        self.assertEqual(artifact["notes"]["verdicts"]["bert"]["verdict"], "ACTIVATE")
        for leg in artifact["legs"]:
            self.assertEqual(leg["ab_op"], "ln", leg)

    def test_end_to_end_main_explicit_op_flag_matches_inference(self):
        rows = self._ln_activate_rows()
        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        rc = m.main([self.root, out_path, "--git-sha", "c" * 40, "--op", "ln"])
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["ab_op"], "ln")

    def test_lora_linear_artifact_carries_no_landing_rule(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "lora_eager",
                 repeats=1, walls_n=[1.0], walls_m=[66.0], batch=8, width=512, dtype="f32")
        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        rc = m.main([self.root, out_path, "--git-sha", "d" * 40])
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["ab_op"], "lora_linear")
        self.assertNotIn("landing_rule", artifact["notes"])

    def test_end_to_end_attn_activate_with_softmax_absorbed_and_flash_declined(self):
        # A full `AB_OP=attn` pipeline: fused legs read
        # attention_block(>0,0)/softmax(0,0) (absorbed), eager legs read
        # attention_block(0,>0)/softmax(0,>0), and attention_block_flash is
        # declined identically on every leg regardless of arm (BERT/
        # DistilBERT never wire the flash transport) -- the pair-identity
        # cross-check this sweep depends on.
        rows = []

        def attn_tier(*, steps, requested, fired, wall, block_disabled, softmax_disabled):
            tier = {}
            tier.update(base_identity(8, 512, "f32"))
            tier.update(base_provenance(steps, requested, fired))
            tier["train_run_wall_s"] = wall
            tier["attention_block_fused_dispatches"] = 0 if block_disabled else 1
            tier["attention_block_eager_dispatches"] = 1 if block_disabled else 0
            if block_disabled:
                tier["softmax_fused_dispatches"] = 0 if softmax_disabled else 1
                tier["softmax_eager_dispatches"] = 1 if softmax_disabled else 0
            else:
                tier["softmax_fused_dispatches"] = 0
                tier["softmax_eager_dispatches"] = 0
            tier["attention_block_flash_fused_dispatches"] = 0
            tier["attention_block_flash_declined_dispatches"] = steps * 12
            return tier

        def add_attn_leg(shape, arm, steps, repeat, wall):
            eager = arm == "attn_eager"
            requested = ["attention_block_fused", "softmax_last_dim_fused"] if eager else []
            fired = list(requested)
            leg_id = f"bert-{shape}-{arm}-{steps}-r{repeat}"
            tier = attn_tier(
                steps=steps, requested=requested, fired=fired, wall=wall,
                block_disabled=eager, softmax_disabled=eager,
            )
            rel = f"raw/{leg_id}.json"
            full = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                json.dump({"tool": "test", "tiers": {"finetune_run": tier}}, f)
            rows.append({
                "leg_id": leg_id, "model": "bert", "shape": shape, "batch": 8, "width": 512,
                "dtype": "f32", "arm": arm, "steps": steps, "repeat": repeat, "ab_op": "attn",
                "env": {"JAMMI_KERNELS_STRICT": "1", "JAMMI_KERNELS_DISABLE": ",".join(requested)},
                "extra_disable": [], "argv": [], "rc": 0, "wall_s": wall, "status": "ok",
                "reason": "", "git_sha": "0" * 40, "box": "test-box", "dry_run": True,
                "report_path": rel, "stderr_path": f"raw/{leg_id}.stderr",
            })

        for arm, walls_m in (("fused", [51.0, 50.0, 51.0]), ("attn_eager", [66.0, 65.0, 65.0])):
            for r, wall_m in enumerate(walls_m, start=1):
                add_attn_leg(m.WIRE_SHAPE, arm, m.STEPS_N, r, 1.0)
                add_attn_leg(m.WIRE_SHAPE, arm, m.STEPS_M, r, wall_m)
        for arm, walls_m in (("fused", [26.0, 25.0, 25.0]), ("attn_eager", [36.0, 35.0, 35.0])):
            for r, wall_m in enumerate(walls_m, start=1):
                add_attn_leg(m.CHAPTER_SHAPE, arm, m.STEPS_N, r, 1.0)
                add_attn_leg(m.CHAPTER_SHAPE, arm, m.STEPS_M, r, wall_m)
        for r, wall_m in enumerate([52.0, 51.0, 51.0], start=1):
            add_attn_leg(m.CONTROL_SHAPE, "control", m.STEPS_N, r, 1.0)
            add_attn_leg(m.CONTROL_SHAPE, "control", m.STEPS_M, r, wall_m)

        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        rc = m.main([self.root, out_path, "--git-sha", "f" * 40, "--op", "attn"])
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["ab_op"], "attn")
        self.assertEqual(artifact["notes"]["verdicts"]["bert"]["verdict"], "ACTIVATE")
        for leg in artifact["legs"]:
            self.assertEqual(leg["violations"], [], leg)

    def test_manifest_disagreeing_on_ab_op_refuses(self):
        rows = []
        add_cell(self.root, rows, "bert", m.CONTROL_SHAPE, "control",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32",
                 ab_op="lora_linear")
        add_cell(self.root, rows, "bert", m.WIRE_SHAPE, "fused",
                 repeats=1, walls_n=[1.0], walls_m=[51.0], batch=8, width=512, dtype="f32",
                 ab_op="ln")
        write_manifest(self.root, rows)
        out_path = os.path.join(self.root, "out.json")
        with self.assertRaises(SystemExit):
            m.main([self.root, out_path, "--git-sha", "e" * 40])


if __name__ == "__main__":
    unittest.main()
