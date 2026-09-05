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
              identity_overrides=None):
    tier = {}
    tier.update(base_identity(batch, width, dtype))
    if identity_overrides:
        tier.update(identity_overrides)
    tier.update(base_provenance(steps, requested, fired))
    tier["train_run_wall_s"] = wall
    tier["lora_linear_fused_dispatches"] = fused_disp
    tier["lora_linear_eager_dispatches"] = eager_disp
    return tier


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
             extra_disable=()):
    """Adds `repeats` complete (N, M) leg pairs for one (model, shape, arm)."""
    requested = ["lora_linear_fused"] if arm == "lora_eager" else []
    requested = list(requested) + list(extra_disable)
    fired = ["lora_linear_fused"] if arm == "lora_eager" else []
    fused_disp = 0 if arm == "lora_eager" else 1
    eager_disp = 1 if arm == "lora_eager" else 0
    for r in range(1, repeats + 1):
        add_leg(
            root, rows, model=model, shape=shape, arm=arm, steps=m.STEPS_N, repeat=r,
            wall=walls_n[r - 1], fused_disp=fused_disp, eager_disp=eager_disp,
            requested=requested, fired=fired, extra_disable=extra_disable,
            batch=batch, width=width, dtype=dtype,
        )
        add_leg(
            root, rows, model=model, shape=shape, arm=arm, steps=m.STEPS_M, repeat=r,
            wall=walls_m[r - 1], fused_disp=fused_disp, eager_disp=eager_disp,
            requested=requested, fired=fired, extra_disable=extra_disable,
            batch=batch, width=width, dtype=dtype,
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


if __name__ == "__main__":
    unittest.main()
