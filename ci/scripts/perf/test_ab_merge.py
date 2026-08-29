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

`CascadePairFixtureTests` (P6 Stage B FA2 fold-in, a docs-ci co-sign of
`origin/perf/p6-fa2-dense` @ `5886c6b`) additionally drives two REAL,
committed raw-run reports from that branch
(`fixtures/p6_fa2_dense_raw_runs/*.json`, provenance in that directory's own
`PROVENANCE.md`) through this same real entry point — never a hand-rolled
dict standing in for what that branch's own `finetune-step` binary actually
emitted.

Run directly: `python3 ci/scripts/perf/test_ab_merge.py`
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_merge  # noqa: E402


LEGS = ab_merge.LEGS
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "p6_fa2_dense_raw_runs")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "finetune_run_golden")

# Every fused/eager pair `FinetuneStepTier` actually serializes today (see
# `crates/jammi-bench/src/report.rs`'s `FinetuneStepTier` and this repo's
# real `finetune-step` output — captured directly, not guessed at, while
# building this fixture set). `adamw` (unit-63 round-3 audit block 1): the
# multi-tensor AdamW commit's own pair, hand-typed here for the SAME reason
# `_FINETUNE_RUN_DISPATCH_COUNTERS` used to lack it -- see
# `GoldenProducerAnchoredFieldSetTests`, which pins this tuple against a
# REAL producer report rather than trusting this hand-kept list alone.
ALL_BASES = ("ln", "rope", "softmax", "geglu", "lora_epilogue", "lora_linear", "attention_block", "adamw")


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
        # PR #381 audit B1/B2: the clip's REQUEST (`null` = clip off, the
        # sweep's default), its COUNTED fact (`0` when off), and the
        # attention reference class — all three now on every real
        # `FinetuneStepTier`. `attention_arm` defaults to `"fused"` here (the
        # jammi-fused leg is the one the premise check reads); a test that
        # writes an eager or clip-on leg overrides them explicitly.
        "max_grad_norm": None,
        "clip_invocations": 0,
        "attention_arm": "fused",
        "warmup": 5,
        "row_lengths": [128] * 8,
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
        # `None`-deletes-the-key convention — EXCEPT for the fields where a
        # real producer's JSON `null` is a VALUE (`identity_fields.
        # FINETUNE_NULL_IS_A_VALUE_FIELDS`: `max_grad_norm = null` is "clip
        # off", the sweep's default). A test that wants that key ABSENT
        # (a pre-#381 binary) deletes it from the returned dict directly.
        if value is None and key not in ab_merge.FINETUNE_NULL_IS_A_VALUE_FIELDS:
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


def flash_overrides(fused=0, declined=0, compiled=True, disabled_requested=None, disabled_fired=None):
    """An `overrides` dict for `jammi_fs` that adds the P6 Stage B FA2
    cascade fields (`attention_block_flash_fused_dispatches`/
    `..._declined_dispatches`/`flash_compiled`/`kernels_disabled_requested`/
    `kernels_disabled_fired`) — NONE of these are in `jammi_fs`'s own base
    dict (they are entirely ABSENT on every report `main`'s own binary
    produces today; a report that predates this fold-in has no `flash`
    key at all), so this is additive, never a replacement of an existing
    key. `disabled_requested`/`disabled_fired` default to `[]` (not
    `None`) — `fs.get(...)` on a real report never reads `null` for these
    two list fields (K-aux's own field doc: "Always present, even on an
    ordinary run with nothing disabled").
    """
    return {
        "attention_block_flash_fused_dispatches": fused,
        "attention_block_flash_declined_dispatches": declined,
        "flash_compiled": compiled,
        "kernels_disabled_requested": list(disabled_requested or []),
        "kernels_disabled_fired": list(disabled_fired or []),
    }


def load_fixture_finetune_step(name):
    """Reads `fixtures/p6_fa2_dense_raw_runs/<name>.json` — a REAL,
    committed `jammi-bench finetune-step` raw-run report copied verbatim
    from `origin/perf/p6-fa2-dense` (see that directory's own
    `PROVENANCE.md`) — and returns its FULL top-level dict (the same shape
    `write_leg` writes straight to a `.json` fixture file), never just the
    `finetune_step` sub-block in isolation.
    """
    with open(os.path.join(FIXTURES_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def load_golden(name):
    """Reads `fixtures/finetune_run_golden/<name>.json` — a REAL, committed
    `jammi-bench finetune-run` report, run once by the actual compiled
    binary (see that directory's own `PROVENANCE.md` for the exact CLI
    invocation and git sha), never a hand-typed field list standing in for
    what the producer actually serializes. The unit-63 round-3 audit class
    fix: `_finetune_run_tier` below loads one of these as its STRUCTURAL
    base so a field this suite's own hand-written literal dict forgets (the
    way `adamw_{fused,eager}_dispatches` fell out of the old
    `_FINETUNE_RUN_DISPATCH_COUNTERS`, block 1's own reproduction) is still
    PRESENT with a real value, never silently absent.
    """
    with open(os.path.join(GOLDEN_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def torch_fs(seed=42, attn_requested="sdpa", lora_alpha=32.0, margin=0.3, warmup=5, **overrides):
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
        # PR #381 audit B1/B2 — same trio as `jammi_fs`, same defaults, so
        # a matching-premise pair stays matching. `attention_arm` is derived
        # from the (possibly overridden) `attn_implementation` below, the
        # way the real producer's `attention_arm_of` derives it, unless a
        # test overrides `attention_arm` itself.
        "max_grad_norm": None,
        "clip_invocations": 0,
        "row_lengths": [128] * 8,
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
    fs.setdefault("attention_arm", "eager" if fs.get("attn_implementation") == "eager" else "fused")
    return {
        "tool": "torch_finetune_step",
        "args": {
            "seed": seed,
            "attn_requested": attn_requested,
            "lora_alpha": lora_alpha,
            "margin": margin,
            "warmup": warmup,
        },
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
        """ln/geglu/adamw required+independent pairs fused; attention_block
        fused (so rope/softmax legitimately (0, 0), absorbed); lora_epilogue
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
                    "adamw": (6, 0),
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
                {"ln": (9, 0), "geglu": (3, 0), "lora_linear": (3, 0), "attention_block": (3, 0), "adamw": (6, 0)},
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
    "adamw": (6, 0),
}


class CascadePairFixtureTests(unittest.TestCase):
    """P6 Stage B FA2 fold-in (docs-ci co-sign of `origin/perf/p6-fa2-dense`
    @ `5886c6b`): `attention_block_flash_fused_dispatches` has no
    `_eager_dispatches` sibling — its fallback counter is
    `_declined_dispatches` instead (`CASCADE_BASES`). Running THIS FILE's
    OWN `test_ab_merge.py` suite (every test above this class) against an
    UNMODIFIED `ab_merge.py` reproduced the exact bug this class pins:
    `dispatch_pairs` raised `KeyError` on that branch's own committed
    fixtures, and `build_report`'s per-leg `try`/`except` turned every leg
    of every config `INVALID` -- verified directly against the REAL
    fixtures below before this fix landed, never merely asserted.

    `test_real_flash_on_fixture_no_longer_keyerrors_but_predates_adamw` /
    `test_real_flash_off_fixture_no_longer_keyerrors_but_predates_adamw`
    drive the two REAL, committed raw-run reports
    (`fixtures/p6_fa2_dense_raw_runs/`, provenance in that directory's own
    `PROVENANCE.md`) through `ab_merge.main` unmodified -- never a
    hand-rolled dict standing in for what that branch's own binary actually
    emitted (unit-63 round-3 audit block 1: both now correctly read
    INVALID, not the original `KeyError` crash -- see each test's own doc).
    Every other test here
    is a synthetic construction (there is no real recorded run of "nothing
    ran" or "flash_compiled=False but disabled" -- those are degenerate/
    contradictory shapes, not real outcomes), built by taking one of the
    real fixtures' `finetune_step` dict and overriding only the field(s)
    each case names (never inventing an unrelated shape), or via
    `jammi_fs`/`flash_overrides` for the isolated single-rule checks.

    Every test writes ONLY `jammi-eager`/`jammi-fused` legs (`torch-eager`/
    `torch-sdpa` stay MISSING) -- `jammi_fused_dispatch_proof` only ever
    reads the `jammi-fused` leg (see `build_report`'s own
    `proof = fused_proof(leg_metrics["jammi-fused"])`), and the `proof is
    False or isinstance(proof, str)` verdict override runs unconditionally
    regardless of whether a torch leg fit at all -- so this is not a
    fixture-completeness shortcut, it isolates exactly the mechanism this
    class exists to test.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "25", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        return rc, merged

    def write_jammi_fused_only(self, raw_dir, slug, report):
        write_leg(raw_dir, slug, "jammi-eager", report=jammi_fs({}))
        write_leg(raw_dir, slug, "jammi-fused", report=report)

    def test_real_flash_on_fixture_no_longer_keyerrors_but_predates_adamw(self):
        """THE ORIGINAL BUG REPRODUCTION: before the cascade-pair fix, this
        raised `KeyError` (via `dispatch_pairs`) on this exact fixture,
        caught per-leg by `build_report` and surfaced as an `"ERROR: ..."`
        string. That is fixed -- `dispatch_pairs` classifies
        `attention_block_flash` cleanly, no raise, no `"ERROR"` string.

        Unit-63 round-3 audit block 1 (docs-ci class fix): `s128_flash_on_1.json`
        is copied byte-for-byte from `origin/perf/p6-fa2-dense` @ `5886c6b`
        (see `fixtures/p6_fa2_dense_raw_runs/PROVENANCE.md`) -- a branch that
        PREDATES the multi-tensor AdamW commit, so this report carries no
        `adamw_{fused,eager}_dispatches` keys AT ALL. `adamw` is now a
        `REQUIRED_PAIRS` member (block 1), and `REQUIRED_PAIRS`'s own
        established doctrine (see `fused_proof`'s own doc, F5) is that an
        ABSENT required base is a hard fail for EVERY classified base,
        never a silently-granted exemption for an older schema -- the SAME
        treatment `ln`'s own absence already got before this fold-in, now
        correctly extended to `adamw` too. This is therefore no longer a
        crash (the original bug), but a CORRECT, INVALID verdict naming a
        real schema-staleness fact about this specific historical fixture
        -- `RealAdamwArtifactFixtureTests` below drives the actual GREEN
        (adamw-carrying) shape this proof exists to pass.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-on", load_fixture_finetune_step("s128_flash_on_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-on"]
        self.assertIsNot(cfg["jammi_fused_dispatch_proof"], None)
        self.assertNotIsInstance(
            cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"]
        )  # never the pre-fix "ERROR: ..." KeyError string
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_real_flash_off_fixture_no_longer_keyerrors_but_predates_adamw(self):
        """THE ORIGINAL BUG REPRODUCTION, the reference-leg side: before the
        cascade-pair fix, this ALSO raised `KeyError` on the exact same
        missing-sibling shape (`attention_block_flash_fused_dispatches`
        present, `..._eager_dispatches` absent -- the fallback key is
        `..._declined_dispatches` here too, just nonzero: `840`). That is
        fixed. `s128_flash_off_1.json` reads `attention_block_flash_fused_
        dispatches: 0`, `..._declined_dispatches: 840`,
        `attention_block_fused_dispatches: 840`,
        `kernels_disabled_requested == kernels_disabled_fired ==
        ["attention_block_flash"]` -- the JAMMI_KERNELS_DISABLE=
        attention_block_flash reference leg, and its `declined: 840` is
        correctly NOT treated as a silent fallback (rule 1's exemption) --
        never the original `KeyError`.

        Unit-63 round-3 audit block 1: this fixture is from the SAME
        pre-AdamW branch as the flash-on sibling above -- see that test's
        own doc for why `REQUIRED_PAIRS`'s absence rule now, correctly,
        also fails this leg (never silently exempted for an older schema).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-off", load_fixture_finetune_step("s128_flash_off_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-off"]
        self.assertNotIsInstance(
            cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"]
        )  # never the pre-fix "ERROR: ..." KeyError string
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_nothing_ran_in_attention_arm_is_invalid(self):
        """Truth-table case 3: `attention_block_flash` reads `(0, 0)` AND
        `attention_block` ALSO reads `fused == 0` -- the whole attention
        arm dispatched nothing at all. Built from the real flash-off
        fixture with only the attention-arm counters and the (no longer
        applicable) disable-request fields zeroed out.
        """
        report = load_fixture_finetune_step("s128_flash_off_1")
        report = copy.deepcopy(report)
        fs = report["tiers"]["finetune_step"]
        fs["attention_block_fused_dispatches"] = 0
        fs["attention_block_flash_fused_dispatches"] = 0
        fs["attention_block_flash_declined_dispatches"] = 0
        fs["kernels_disabled_requested"] = []
        fs["kernels_disabled_fired"] = []
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-nothing-ran", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-nothing-ran"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_flash_compiled_false_but_disable_requested_is_invalid(self):
        """Truth-table case 4: a build that never compiled flash in
        (`flash_compiled: false`) cannot possibly have exercised a disable
        request naming it -- the leg's own build configuration contradicts
        its own disable request, loud and INVALID regardless of what the
        dispatch counters themselves read. Built from the real flash-off
        fixture (which DOES carry a real `attention_block_flash` disable
        request) with only `flash_compiled` flipped.
        """
        report = copy.deepcopy(load_fixture_finetune_step("s128_flash_off_1"))
        report["tiers"]["finetune_step"]["flash_compiled"] = False
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-not-compiled", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-not-compiled"]
        proof = cfg["jammi_fused_dispatch_proof"]
        self.assertIsInstance(proof, str, proof)
        self.assertIn("flash_compiled", proof)
        self.assertIn("attention_block_flash", proof)
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_flash_compiled_false_capability_miss_is_still_a_hard_fail(self):
        """Unit-63 round-3 audit, coordinator correction: an earlier draft
        of this round exempted `flash_compiled is False` from rule 1 (a
        capability-miss carve-out) -- REVERTED. `fused_proof` is SHARED by
        `finetune-step`'s own campaigns; a whole-campaign premise fact
        (CONTRACT 63 Frame pre-registers the flash cascade as the
        finetune-run how-well A/B's own differential) belongs in THAT
        campaign's own premise check
        (`finetune_run_dispatch_proof_violations`'s `arm == "fused"`
        branch), never a silent, generic exemption inside the shared
        primitive. This is the regression pin: the exact capability-miss
        shape (a real disable request CLEARED, so this is genuinely a
        capability miss, not also a self-describing one, plus a synthetic
        `adamw` pair -- this fixture predates the multi-tensor AdamW commit)
        must still hard-fail, unconditionally, exactly like an ordinary
        silent eager fallback always has.
        """
        report = copy.deepcopy(load_fixture_finetune_step("s128_flash_off_1"))
        fs = report["tiers"]["finetune_step"]
        fs["flash_compiled"] = False
        fs["kernels_disabled_requested"] = []
        fs["kernels_disabled_fired"] = []
        fs["adamw_fused_dispatches"] = 6
        fs["adamw_eager_dispatches"] = 0
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-capability-miss", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-capability-miss"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_unrequested_decline_is_still_a_hard_fail_non_vacuous_control(self):
        """Negative control (non-vacuous): rule 1's exemption for a
        `CASCADE_BASES` decline is gated on `kernels_disabled_requested`
        AND `kernels_disabled_fired` BOTH naming the base -- a decline that
        happens WITHOUT either (a genuine domain/capability miss: real
        padding, wrong arch, `flash-attn` not compiled) must still hard-fail
        exactly like an ordinary silent eager fallback always has. Built
        from the real flash-on fixture (`kernels_disabled_requested: []`
        unmodified) with `attention_block_flash_declined_dispatches` alone
        flipped nonzero -- proves the exemption is NOT "any CASCADE_BASES
        decline is fine", only a SELF-DESCRIBING one.
        """
        report = copy.deepcopy(load_fixture_finetune_step("s128_flash_on_1"))
        report["tiers"]["finetune_step"]["attention_block_flash_declined_dispatches"] = 5
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-unrequested-decline", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-unrequested-decline"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_rope_softmax_absorbed_via_flash_arm_alone_isolated_rule(self):
        """Isolates `ABSORBABLE_BY_ATTENTION_BLOCK`'s extended OR condition:
        `rope`/`softmax` may read `(0, 0)` when `attention_block_flash`'s
        `fused > 0`, even though `attention_block` ITSELF also reads
        `(0, 0)` -- absorbed transitively through the flash arm, not merely
        because `attention_block` happened to be positive (that path is
        already covered by the real flash-off fixture test above; this
        isolates the flash-only leg of the OR).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(
                {
                    "ln": (9, 0),
                    "geglu": (3, 0),
                    "lora_linear": (3, 0),
                    "attention_block": (0, 0),
                    "adamw": (6, 0),
                },
                **flash_overrides(fused=5, declined=0),
            )
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-absorbs", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-absorbs"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertEqual(rc, 0)

    def test_flash_absent_from_schema_preserves_old_required_attention_block_behaviour(self):
        """Backward-compatibility pin: a report with NO `attention_block_
        flash` key at all (every report `main`'s own binary produces
        today) must treat `attention_block` EXACTLY as `REQUIRED_PAIRS`
        used to before this fold-in -- `attention_block` reading `(0, 0)`
        with no flash key present is still a hard fail, never silently
        exempted just because the classification table changed shape.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(
                {
                    "ln": (9, 0),
                    "geglu": (3, 0),
                    "lora_linear": (3, 0),
                    "attention_block": (0, 0),
                }
            )
            self.write_jammi_fused_only(raw_dir, "b8-s128-no-flash-key", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-no-flash-key"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_provenance_records_flash_compiled_and_declined_counter(self):
        """`leg_provenance`'s `jammi_flash_compiled` field, and
        `jammi_dispatch_counters` picking up a `_declined_dispatches`-
        suffixed key (not just `_fused_dispatches`/`_eager_dispatches`) --
        both recorded, never compared, same "provenance" row this fold-in
        adds to the module docstring's determinant table. `leg_provenance`
        records the RAW counters unconditionally (it is never itself gated
        by `fused_proof`), so this still holds even though this exact real
        fixture predates the multi-tensor AdamW commit (unit-63 round-3
        audit block 1: `rc`/`verdict` now read INVALID/1 for THIS config,
        since `adamw` is a `REQUIRED_PAIRS` member this schema-older report
        cannot supply -- see `RealAdamwArtifactFixtureTests` for the
        GREEN, adamw-carrying leg this proof exists to pass).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(
                raw_dir, "b8-s128-flash-on", load_fixture_finetune_step("s128_flash_on_1")
            )
            out_dir = tempfile.mkdtemp()
            rc = ab_merge.main([raw_dir, out_dir, "25", "5", "0.9"])
            self.assertEqual(rc, 1)
            with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
                merged = json.load(fh)
        cfg = merged["configs"]["b8-s128-flash-on"]
        self.assertIs(cfg["provenance"]["jammi"]["jammi_flash_compiled"], True)
        counters = cfg["provenance"]["jammi"]["jammi_dispatch_counters"]
        self.assertIn("attention_block_flash_declined_dispatches", counters)
        self.assertEqual(counters["attention_block_flash_declined_dispatches"], 0)
        self.assertEqual(counters["attention_block_flash_fused_dispatches"], 840)

    def test_cascade_shaped_unknown_base_without_declined_sibling_still_raises(self):
        """Schema-drift-is-loud, extended to the cascade shape: a NEW
        `_fused_dispatches` key for a base that is NOT in `CASCADE_BASES`
        (so `_fallback_key` looks for `_eager_dispatches`, not
        `_declined_dispatches`) and carries ONLY a `_declined_dispatches`
        sibling -- no `_eager_dispatches` at all -- must still raise a
        LOUD, per-leg error, never silently pass as if it were an
        ordinary `(0, 0)` pair.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs({"ln": (9, 0), "geglu": (3, 0), "lora_linear": (3, 0), "attention_block": (3, 0)})
            report["tiers"]["finetune_step"]["mystery_cascade_fused_dispatches"] = 7
            report["tiers"]["finetune_step"]["mystery_cascade_declined_dispatches"] = 0
            self.write_jammi_fused_only(raw_dir, "b8-s128-mystery-cascade", report)
            rc, merged = self.run_merge(raw_dir)
        proof = merged["configs"]["b8-s128-mystery-cascade"]["jammi_fused_dispatch_proof"]
        self.assertIsInstance(proof, str)
        self.assertIn("ERROR", proof)
        self.assertIn("mystery_cascade_eager_dispatches", proof)
        self.assertTrue(merged["configs"]["b8-s128-mystery-cascade"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)


# The lead's own reproduction path (unit-63 round-3 audit block 1): the
# committed real artifact, NOT copied into this crate's own `fixtures/`
# directory (unlike `p6_fa2_dense_raw_runs/`/`finetune_run_golden/`) --
# `crates/jammi-kernels/artifacts/cuda-runs/` is a DIFFERENT crate's own
# tracked-input artifact tree, read here in place, verbatim, at its real
# repo path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_REAL_ADAMW_ARTIFACT_DIR = os.path.join(
    _REPO_ROOT,
    "crates",
    "jammi-kernels",
    "artifacts",
    "cuda-runs",
    "2026-08-25-adamw-d959805-a100-sxm4-raw-runs",
    "a100b",
)


def load_real_adamw_artifact(name):
    """Reads `<name>.json.raw` from the committed real `a100b` adamw A/B
    raw-run directory (see that directory's own `PROVENANCE.md`) — a
    genuine `jammi-bench finetune-step` report, run by hand on real
    hardware, never a hand-rolled dict. Renamed `.json.raw` (not `.json`)
    by that directory's own convention (kept outside `check_cuda_run_
    artifacts.py`'s `*.json` schema glob — see its own `PROVENANCE.md`),
    but its CONTENTS are the exact JSON shape a real binary invocation
    emits.
    """
    with open(os.path.join(_REAL_ADAMW_ARTIFACT_DIR, f"{name}.json.raw")) as fh:
        return json.load(fh)


class RealAdamwArtifactFixtureTests(unittest.TestCase):
    """Unit-63 round-3 audit block 1's own reproduction: `dispatch_pairs`
    raised `KeyError('adamw')` on EVERY real leg of this committed artifact
    before `adamw` was added to `ALL_BASES`/`REQUIRED_PAIRS` — confirmed
    directly against `b8_s512_fused.r2.json.raw` (the exact leg the lead's
    audit named), never merely asserted. Every test here reads the REAL
    file at its own repo path (`load_real_adamw_artifact`), never a
    hand-rolled dict standing in for what this specific hardware run
    actually emitted.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "25", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        return rc, merged

    def write_jammi_fused_only(self, raw_dir, slug, report):
        write_leg(raw_dir, slug, "jammi-eager", report=jammi_fs({}))
        write_leg(raw_dir, slug, "jammi-fused", report=report)

    def test_real_fused_leg_no_longer_keyerrors_and_is_green(self):
        """THE BUG REPRODUCTION named directly by the audit: this leg's own
        `kernels_disabled_requested`/`kernels_disabled_fired` are BOTH empty
        (the fused arm, no disable request at all) and every `REQUIRED_PAIRS`/
        `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
        `LORA_SITE_EXCLUSIVE_GROUP` member this leg's own schema carries
        shows a real, positive fused count (`ln`, `geglu`, `adamw`,
        `attention_block`, `lora_linear` all `> 0`; `rope`/`softmax`
        legitimately absorbed at `(0, 0)` via `attention_block`'s own
        `fused > 0`) — this is the GENUINE green shape `fused_proof` exists
        to pass, once `dispatch_pairs` stops raising on `adamw`.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(
                raw_dir, "b8-s512-fused", load_real_adamw_artifact("b8_s512_fused.r2")
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s512-fused"]
        self.assertNotIsInstance(cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"])
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_real_fused_leg_s128_shape_is_also_green(self):
        """The sibling shape at the OTHER committed seq length — same
        finding, a second real leg."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(
                raw_dir, "b8-s128-fused", load_real_adamw_artifact("b8_s128_fused.r1")
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-fused"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertEqual(rc, 0)

    def test_real_disabled_leg_no_longer_keyerrors(self):
        """The sibling `JAMMI_KERNELS_DISABLE=adamw_step_fused` reference
        leg: `adamw_fused_dispatches: 0` / `adamw_eager_dispatches: 6720`,
        `kernels_disabled_requested == kernels_disabled_fired ==
        ["adamw_step_fused"]`. `adamw` is an ORDINARY `REQUIRED_PAIRS`
        member (never a `CASCADE_BASES` one — see `ALL_BASES`'s own doc),
        so rule 1's self-describing-disable-request exemption does NOT
        apply to it (that exemption is scoped to `CASCADE_BASES` only) --
        this leg correctly reads `fused_proof` `False` (a real, deliberate
        eager fallback on a REQUIRED pair), never the pre-fix `KeyError`.
        This leg is never itself passed through `fused_proof` by
        `build_report` (only the `jammi-fused` leg is), so this test drives
        `dispatch_pairs` directly — the exact function that raised.
        """
        report = load_real_adamw_artifact("b8_s512_disabled.r2")
        fs = report["tiers"]["finetune_step"]
        pairs = ab_merge.dispatch_pairs(fs)  # must not raise
        by_base = dict((base, (fused, fallback)) for base, fused, fallback in pairs)
        self.assertIn("adamw", by_base)
        self.assertEqual(by_base["adamw"], (0, 6720))

    def test_every_real_leg_in_the_artifact_directory_dispatch_pairs_cleanly(self):
        """Broad, non-vacuous sweep: EVERY `.json.raw` file in the committed
        real artifact directory (fused and disabled, both shapes, both
        repeats) must classify through `dispatch_pairs` without raising —
        the original bug raised on ALL EIGHT of these, unconditionally.
        """
        names = [
            "b8_s128_disabled.r1",
            "b8_s128_disabled.r2",
            "b8_s128_fused.r1",
            "b8_s128_fused.r2",
            "b8_s512_disabled.r1",
            "b8_s512_disabled.r2",
            "b8_s512_fused.r1",
            "b8_s512_fused.r2",
        ]
        for name in names:
            report = load_real_adamw_artifact(name)
            fs = report["tiers"]["finetune_step"]
            pairs = ab_merge.dispatch_pairs(fs)  # must not raise for any of the eight
            by_base = dict((base, (fused, fallback)) for base, fused, fallback in pairs)
            self.assertIn("adamw", by_base, f"{name}: adamw pair not discovered")


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


class GenericLegPremiseCheckTests(unittest.TestCase):
    """Unit-62 E6: `generic_leg_identity_fields`/`generic_leg_premise_violations`
    -- the shared core `leg_identity_fields`/`leg_premise_violations` above
    reduce to, factored out so `encode_ab.sh`'s own merge step (this unit's
    NEW producer, ENCODE_IDENTITY_FIELDS-driven, two `jammi-bench encode-step`
    replicate legs) can reuse the identical leg-premise-refusal logic instead
    of hand-rolling a second comparator. These tests exercise the two
    functions directly (no `finetune_ab.sh`/`main()` plumbing) against a
    small synthetic field tuple -- the same shape ENCODE_IDENTITY_FIELDS has,
    without depending on that tuple's exact membership so a future field
    added there cannot spuriously break this generic-machinery test.
    """

    FIELDS = ("seed", "batch", "seq")

    def test_matching_premise_across_two_legs_is_clean(self):
        block_a = {"seed": 42, "batch": 8, "seq": 128}
        block_b = {"seed": 42, "batch": 8, "seq": 128}
        fields_a = ab_merge.generic_leg_identity_fields(block_a, self.FIELDS)
        fields_b = ab_merge.generic_leg_identity_fields(block_b, self.FIELDS)
        self.assertEqual(
            ab_merge.generic_leg_premise_violations(self.FIELDS, fields_a, fields_b, "r1", "r2"),
            [],
        )

    def test_differing_field_is_a_violation(self):
        fields_a = ab_merge.generic_leg_identity_fields({"seed": 42, "batch": 8, "seq": 128}, self.FIELDS)
        fields_b = ab_merge.generic_leg_identity_fields({"seed": 42, "batch": 16, "seq": 128}, self.FIELDS)
        violations = ab_merge.generic_leg_premise_violations(self.FIELDS, fields_a, fields_b, "r1", "r2")
        self.assertTrue(any("batch" in v and "r1" in v and "r2" in v for v in violations), violations)

    def test_field_absent_from_one_side_is_a_violation_naming_that_side(self):
        fields_a = ab_merge.generic_leg_identity_fields({"seed": 42, "batch": 8, "seq": 128}, self.FIELDS)
        fields_b = ab_merge.generic_leg_identity_fields({"seed": 42, "seq": 128}, self.FIELDS)  # batch absent
        violations = ab_merge.generic_leg_premise_violations(self.FIELDS, fields_a, fields_b, "r1", "r2")
        self.assertTrue(any("batch" in v and "['r2']" in v for v in violations), violations)

    def test_present_but_null_is_folded_into_missing_by_default(self):
        """No `ENCODE_IDENTITY_FIELDS` entry is a `null_is_value_fields`
        member (every one is `Nullable::NonNull` on `EncodeStepTier`) -- a
        present-but-null value must be treated identically to an absent key
        with the default (empty) `null_is_value_fields`.
        """
        fields = ab_merge.generic_leg_identity_fields({"seed": None, "batch": 8, "seq": 128}, self.FIELDS)
        self.assertIs(fields["seed"], ab_merge._MISSING)

    def test_null_is_value_fields_lets_a_present_null_match(self):
        fields_a = ab_merge.generic_leg_identity_fields({"seed": None, "batch": 8, "seq": 128}, self.FIELDS, null_is_value_fields={"seed"})
        fields_b = ab_merge.generic_leg_identity_fields({"seed": None, "batch": 8, "seq": 128}, self.FIELDS, null_is_value_fields={"seed"})
        self.assertEqual(ab_merge.generic_leg_premise_violations(self.FIELDS, fields_a, fields_b), [])


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


class ClipAndAttentionIdentityTests(unittest.TestCase):
    """PR #381 audit B1 (+ the lead's class probe): `max_grad_norm` and the
    attention reference class are IDENTITY — two legs differing in either
    compute a different step. Before this round `ab_merge.py`'s hand-kept
    tuple lacked both, so a clip-on jammi leg merged against a clip-off
    torch leg and printed PASS. Every test here drives `ab_merge.main`
    against fixture legs, the same way the premise tests above do.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        return rc, merged

    def write_pair(self, raw_dir, jammi_overrides=None, torch_overrides=None, torch_sdpa_ok=True):
        write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}, attention_arm="eager"))
        write_leg(
            raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES, **(jammi_overrides or {}))
        )
        write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
        if torch_sdpa_ok:
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(**(torch_overrides or {})))
        else:
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", exit_code=1, stderr="CUDA out of memory")

    def test_jammi_clip_on_vs_torch_clip_off_is_refused(self):
        """THE B1 REPRODUCTION: jammi ran `--max-grad-norm 1.0` (and counted
        26 clip calls: 20 steps + 5 warmup + 1 pre-step), torch ran with the
        flag absent. Used to PASS.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir, jammi_overrides={"max_grad_norm": 1.0, "clip_invocations": 26})
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(
            any("max_grad_norm" in v and "differs" in v for v in cfg["leg_premise_violations"]),
            cfg["leg_premise_violations"],
        )
        self.assertTrue(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_torch_clip_on_vs_jammi_clip_off_is_refused(self):
        """The mirror image — the refusal is symmetric, not a jammi-only
        rule."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir, torch_overrides={"max_grad_norm": 1.0, "clip_invocations": 26})
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("max_grad_norm" in v and "differs" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_different_max_grad_norm_values_are_refused(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(
                raw_dir,
                jammi_overrides={"max_grad_norm": 1.0, "clip_invocations": 26},
                torch_overrides={"max_grad_norm": 0.5, "clip_invocations": 26},
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("max_grad_norm" in v and "differs" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_matching_clip_on_both_sides_is_not_refused(self):
        """Positive control: a genuinely matched clip-on pair (same bound,
        both counted) must not false-fail."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(
                raw_dir,
                jammi_overrides={"max_grad_norm": 1.0, "clip_invocations": 26},
                torch_overrides={"max_grad_norm": 1.0, "clip_invocations": 26},
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertFalse(cfg["verdict"].startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)
        self.assertEqual(cfg["provenance"]["jammi"]["jammi_clip_invocations"], 26)
        self.assertEqual(cfg["provenance"]["torch"]["torch_clip_invocations"], 26)

    def test_clip_off_on_both_sides_null_is_a_value_not_missing(self):
        """`max_grad_norm: null` on BOTH legs is the sweep's default (clip
        OFF) and must compare as a matching VALUE — the round-4 null-folds-
        to-MISSING rule is deliberately NOT applied to this field
        (`identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`)."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertEqual(rc, 0)

    def test_max_grad_norm_absent_from_a_leg_is_still_missing(self):
        """A jammi binary built before the field existed: the KEY is absent
        (not null). Must refuse as MISSING — a producer that cannot state
        its clip premise is not a matching one."""
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES)
            del report["tiers"]["finetune_step"]["max_grad_norm"]
            del report["tiers"]["finetune_step"]["clip_invocations"]
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}, attention_arm="eager"))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("max_grad_norm" in v and "missing" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_clip_requested_but_never_counted_is_refused(self):
        """B2: the counted fact must back the request. `max_grad_norm: 1.0`
        with `clip_invocations: 0` is a row claiming a clip that never
        ran."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(
                raw_dir,
                jammi_overrides={"max_grad_norm": 1.0, "clip_invocations": 0},
                torch_overrides={"max_grad_norm": 1.0, "clip_invocations": 26},
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("jammi-fused" in v and "clip never ran" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_clip_counted_but_not_requested_is_refused(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir, torch_overrides={"max_grad_norm": None, "clip_invocations": 26})
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("torch-sdpa" in v and "ran anyway" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_clip_request_without_a_counted_fact_is_refused(self):
        """`max_grad_norm` present but `clip_invocations` absent: a clip
        claim with nothing counted behind it."""
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES, max_grad_norm=1.0)
            del report["tiers"]["finetune_step"]["clip_invocations"]
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}, attention_arm="eager"))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=report)
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs(max_grad_norm=1.0, clip_invocations=26))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("clip_invocations" in v and "absent" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_torch_sdpa_oom_fallback_to_torch_eager_is_not_comparable_not_invalid(self):
        """PR #381 re-audit, face A2: `build_report` falls back to the
        `torch-eager` leg for PROVENANCE when `torch-sdpa` OOM'd — a
        documented NON-gating outcome. That leg is the other attention
        reference class by construction, so the identity check is SKIPPED
        for the row (never refused as an `attention_arm` mismatch, which
        would have turned every torch-OOM config into INVALID + exit 1),
        the reason is recorded, and the verdict/ratio logic is unchanged.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir, torch_sdpa_ok=False)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["leg_premise_violations"])
        self.assertIsNone(cfg["leg_premise_checked_legs"])
        self.assertIn("torch-eager is a fallback for torch-sdpa (OOM)", cfg["leg_premise_not_comparable"])
        self.assertIsNone(cfg["ratio_jammi_fused_over_torch_sdpa"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(cfg["provenance"]["torch"]["torch_attn_implementation"], "eager")
        self.assertEqual(rc, 0)

    def test_jammi_fused_failed_fallback_to_jammi_eager_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}, attention_arm="eager"))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", exit_code=1, stderr="CUDA out of memory")
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["leg_premise_violations"])
        self.assertIn("jammi-eager is a fallback for jammi-fused (OOM)", cfg["leg_premise_not_comparable"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID (leg premise"), cfg["verdict"])

    def test_preferred_legs_record_no_not_comparable_reason(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["leg_premise_not_comparable"])
        self.assertEqual(cfg["leg_premise_violations"], [])

    def test_jammi_fused_leg_with_attention_disabled_against_torch_sdpa_is_refused(self):
        """A `jammi-fused` leg whose operator's `JAMMI_KERNELS_DISABLE`
        named an attention base (leaked into the fused leg's environment)
        reads `attention_arm: "eager"` — the REQUEST, not the counters —
        and must not pair with torch-sdpa."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(
                raw_dir,
                jammi_overrides={
                    "attention_arm": "eager",
                    "kernels_disabled_requested": ["attention_block"],
                    "kernels_disabled_fired": ["attention_block"],
                },
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(
            any("attention_arm" in v and "jammi='eager' torch='fused'" in v for v in cfg["leg_premise_violations"]),
            cfg["leg_premise_violations"],
        )
        self.assertEqual(rc, 1)

    def test_domain_declined_counters_do_not_make_a_fused_leg_eager(self):
        """PR #381 re-audit, face A1: a jammi-fused leg on a checkpoint the
        fused attention predicate DECLINES BY DOMAIN (e.g. head_dim != 64)
        has eager attention_block counters but `attention_arm: "fused"`
        (nothing was disabled). The identity check must NOT refuse it; the
        counters remain `fused_proof`'s business (this fixture's proof
        fails for other reasons, and that is the ONLY thing allowed to
        turn the verdict INVALID here)."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(
                raw_dir,
                jammi_overrides={"attention_block": None, "attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 840},
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertNotIn("leg premise mismatch", str(cfg["verdict"]))

    def test_warmup_mismatch_between_legs_is_refused(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir, jammi_overrides={"warmup": 5}, torch_overrides={"warmup": 0})
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("warmup" in v and "differs" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)

    def test_attention_arm_absent_from_torch_leg_is_missing(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            torch_report = torch_fs()
            del torch_report["finetune_step"]["attention_arm"]
            write_leg(raw_dir, "b8-s128-d0", "jammi-eager", report=jammi_fs({}, attention_arm="eager"))
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES))
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(any("attention_arm" in v and "missing" in v for v in cfg["leg_premise_violations"]))
        self.assertEqual(rc, 1)


class SharedIdentityDeclarationTests(unittest.TestCase):
    """The ONE shared declaration (`identity_fields.FINETUNE_IDENTITY_FIELDS`)
    is what `ab_merge` iterates AND what both producers must emit. These
    tests pin the three ends together statically: `ab_merge` carries no
    tuple of its own; the torch producer's report literal names every
    member; and jammi's `FinetuneStepTier` names every member (its own
    Rust-side pin, `finetune_step_tier_emits_every_shared_identity_field`,
    reads the same tuple from the same file — this test reads the struct
    source so the pin holds from BOTH languages).
    """

    REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

    def test_ab_merge_iterates_the_shared_tuple_not_its_own(self):
        import identity_fields

        self.assertIs(ab_merge.FINETUNE_IDENTITY_FIELDS, identity_fields.FINETUNE_IDENTITY_FIELDS)
        with open(ab_merge.__file__) as fh:
            src = fh.read()
        self.assertNotIn("FINETUNE_IDENTITY_FIELDS = (", src, "ab_merge.py must not redeclare the identity tuple")
        self.assertIn("max_grad_norm", identity_fields.FINETUNE_IDENTITY_FIELDS)
        self.assertIn("attention_arm", identity_fields.FINETUNE_IDENTITY_FIELDS)

    def test_torch_producer_emits_every_shared_identity_field(self):
        path = os.path.join(self.REPO, "crates", "jammi-bench", "reference", "torch_finetune_step.py")
        with open(path) as fh:
            src = fh.read()
        # The report literal: `"args": {` ... and `"finetune_step": {` ...
        # Both blocks' keys, by the `"<name>":` spelling the literal uses.
        start = src.index('        report = {')
        end = src.index("\n        }\n", start)
        literal = src[start:end]
        emitted = set(re.findall(r'^\s*"([a-z_0-9]+)":', literal, flags=re.MULTILINE))
        # `checkpoint_*` come in via `**checkpoint_identity_fields` — the
        # `checkpoint_identity` helper's own literal.
        ci_start = src.index("def checkpoint_identity(")
        ci_end = src.index("\ndef ", ci_start + 1)
        emitted |= set(re.findall(r'"([a-z_0-9]+)":', src[ci_start:ci_end]))
        missing = [f for f in ab_merge.FINETUNE_IDENTITY_FIELDS if f not in emitted]
        self.assertEqual(missing, [], f"torch_finetune_step.py's report literal does not emit {missing}")

    def test_jammi_tier_struct_names_every_shared_identity_field(self):
        path = os.path.join(self.REPO, "crates", "jammi-bench", "src", "report.rs")
        with open(path) as fh:
            src = fh.read()
        start = src.index("pub struct FinetuneStepTier {")
        end = src.index("\n}\n", start)
        fields = set(re.findall(r"^\s*pub ([a-z_0-9]+):", src[start:end], flags=re.MULTILINE))
        missing = [f for f in ab_merge.FINETUNE_IDENTITY_FIELDS if f not in fields]
        self.assertEqual(missing, [], f"FinetuneStepTier does not carry {missing}")


class EmptyRawDirTests(unittest.TestCase):
    def test_no_leg_output_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        self.assertEqual(rc, 1)


class TorchIdentityFieldsAgainstADryRunDumpTests(unittest.TestCase):
    """Unification contract C3.5: `torch_finetune_step.py::TORCH_IDENTITY_FIELDS`
    must actually be present, and non-null where NOT declared nullable, in the
    JSON this producer emits.

    ONE REQUIRED leg + one best-effort leg — deliberately NOT symmetric,
    because `torch_finetune_step.py`'s own module doc already states this
    script is "never invoked from CI" (a human/CI *operator* runs it BY HAND
    on a rented GPU pod, next to `jammi-bench finetune-step`); wiring a real
    `--dry-run` execution into automated CI would reverse that pre-existing,
    deliberate design decision, not merely wire a test in. So:

    - `test_static_source_covers_every_declared_field` is the REQUIRED,
      ALWAYS-RUNNING oracle (stdlib-only `ast`, no torch needed — this is
      what actually executes in the `guard` matrix leg, a plain shallow
      checkout with no Python ML stack). It is scoped to exactly the THREE
      places this producer actually assembles emitted JSON —
      `provenance()`'s own `info = {...}` dict, `checkpoint_identity()`'s own
      return dict (the `**checkpoint_identity_fields` unpack inside the
      `finetune_step` block — an unpack is not a literal string key `ast.Dict.
      keys` sees, so its SOURCE function is walked directly instead), and the
      ONE `report = {...}` literal inside `run()` (identified structurally,
      by the dict literal that carries BOTH a `"finetune_step"` AND a
      `"provenance"` key, so a future rename cannot silently re-target the
      wrong dict) — never every `ast.Dict` in the module. Round-2 audit (B3)
      caught the vacuous
      shape this replaces: collecting keys from EVERY dict in the module
      also swept in `TORCH_IDENTITY_FIELDS_NULL_MEANS` (the classification
      table declared two lines below `TORCH_IDENTITY_FIELDS` itself, whose
      keys are the SAME field names) — a field declared in
      `TORCH_IDENTITY_FIELDS` and named ONLY in `NULL_MEANS`, never actually
      assigned anywhere the producer emits, still passed. Mutation check:
      add `"max_grad_norm"` to `TORCH_IDENTITY_FIELDS` and to
      `TORCH_IDENTITY_FIELDS_NULL_MEANS` ONLY (never to `provenance()`'s or
      `run()`'s own dict literals) — this leg now goes RED, where the
      pre-fix version stayed green.
    - `test_real_dry_run_dump_names_every_field` is a best-effort SUPPLEMENT,
      not the enforcement mechanism: it actually spawns
      `torch_finetune_step.py --dry-run` and checks every `NonNull` entry is
      `is not None` (never mere presence) and every `TORCH_IDENTITY_FIELDS_
      NULL_MEANS` entry is at least present — but it SKIPS, never RED,
      when `torch`/`transformers`/`peft` are not importable (this
      environment, and the plain `guard` matrix leg, never have them). This
      is intentional, not a zero-execution violation of the "zero-execution
      is RED, not a skip" convention (`.github/workflows/ci.yml` guard
      matrix's own doc): that convention governs REQUIRED gates: this leg
      was never wired as one — the static leg above is, and it has no skip
      path at all.
    """

    TORCH_SCRIPT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "crates", "jammi-bench",
        "reference", "torch_finetune_step.py",
    )

    def _module_source(self) -> str:
        path = os.path.abspath(self.TORCH_SCRIPT)
        self.assertTrue(os.path.isfile(path), f"missing: {path}")
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def _declared_fields(self, tree):
        import ast

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "TORCH_IDENTITY_FIELDS"
            ):
                return [elt.value for elt in node.value.elts]
        return None

    def _string_dict_keys(self, node):
        import ast

        keys = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                for key in sub.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys

    def _emitted_keys(self, tree):
        """Keys this producer's report literals ACTUALLY carry — scoped to
        `provenance()`'s own dict and the ONE `report = {...}` literal
        inside `run()` (identified by structure — see the class doc). Never
        every `ast.Dict` in the module."""
        import ast

        keys = set()
        provenance_fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "provenance"),
            None,
        )
        self.assertIsNotNone(provenance_fn, "torch_finetune_step.py's provenance() not found — RED at base")
        keys |= self._string_dict_keys(provenance_fn)

        # `run()`'s report["finetune_step"] block does `**checkpoint_identity_
        # fields,` (a dict-UNPACK, not a literal string key `ast.Dict.keys`
        # can see) — those three fields (checkpoint_config_sha256/
        # checkpoint_weights_sha256/checkpoint_weights_size_bytes) arrive from
        # `checkpoint_identity()`'s own return dict, so that function's keys
        # are ALSO part of what this producer emits.
        checkpoint_identity_fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_identity"),
            None,
        )
        self.assertIsNotNone(
            checkpoint_identity_fn, "torch_finetune_step.py's checkpoint_identity() not found — RED at base"
        )
        keys |= self._string_dict_keys(checkpoint_identity_fn)

        run_fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run"), None
        )
        self.assertIsNotNone(run_fn, "torch_finetune_step.py's run() not found — RED at base")

        report_dict = None
        for sub in ast.walk(run_fn):
            if isinstance(sub, ast.Dict):
                str_keys = {
                    k.value
                    for k in sub.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                if "finetune_step" in str_keys and "provenance" in str_keys:
                    report_dict = sub
                    break
        self.assertIsNotNone(
            report_dict,
            "no dict literal carrying both 'provenance' and 'finetune_step' keys found inside "
            "run() — the report = {...} literal this leg targets was not found (structure changed, "
            "or this scoping needs updating for a genuine reformat)",
        )
        keys |= self._string_dict_keys(report_dict)
        return keys

    def test_static_source_covers_every_declared_field(self):
        import ast

        tree = ast.parse(self._module_source())
        declared = self._declared_fields(tree)
        self.assertIsNotNone(
            declared, "TORCH_IDENTITY_FIELDS not found in torch_finetune_step.py — RED at base"
        )
        self.assertEqual(len(declared), len(set(declared)), "TORCH_IDENTITY_FIELDS has a duplicate")

        emitted = self._emitted_keys(tree)
        missing = sorted(set(declared) - emitted)
        self.assertFalse(
            missing,
            f"TORCH_IDENTITY_FIELDS names field(s) that never appear as a key in provenance()'s own "
            f"dict or the report={{...}} literal inside run(): {missing} — declaring a field in "
            f"TORCH_IDENTITY_FIELDS_NULL_MEANS does NOT count as emitting it",
        )

    def test_real_dry_run_dump_names_every_field(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError:
            self.skipTest(
                "torch/transformers/peft not installed in this environment — best-effort "
                "supplement, never the enforcement mechanism (see class doc); "
                "test_static_source_covers_every_declared_field is the REQUIRED oracle and does "
                "not skip"
            )

        sys.path.insert(0, os.path.dirname(os.path.abspath(self.TORCH_SCRIPT)))
        import torch_finetune_step as tfs  # noqa: E402

        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as out_dir:
            out_path = os.path.join(out_dir, "dry_run.json")
            rc = tfs.main(
                [
                    "--dry-run",
                    "--batch",
                    "2",
                    "--seq",
                    "6",
                    "--steps",
                    "1",
                    "--warmup",
                    "0",
                    "--out",
                    out_path,
                ]
            )
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                dump = json.load(fh)

        def _resolve(field):
            """Round-3 audit (advisory 4): the OLD form stopped at the
            FIRST block (`provenance`/`args`/`finetune_step`, in that
            order) that merely CONTAINED `field` — a `NonNull` field that
            read `null` in an EARLIER block (or absent) masked a genuine
            non-null value sitting in a LATER block this function never
            looked at, and "absent from every block" collapsed onto the
            exact same return value as "present-but-null in the one block
            that has it" (both `False`, no diagnostic). This version
            checks ALL THREE blocks and returns a per-block status dict
            (`"absent"` / `"null"` / `"present"`) alongside the verdict, so
            a failure names exactly where the field stood in EACH block.
            """
            per_block = {}
            satisfied = False
            for block_name in ("provenance", "args", "finetune_step"):
                block = dump.get(block_name, {})
                if not isinstance(block, dict) or field not in block:
                    per_block[block_name] = "absent"
                    continue
                if block[field] is None:
                    per_block[block_name] = "null"
                else:
                    per_block[block_name] = "present"
                    satisfied = True
            if not satisfied and field in tfs.TORCH_IDENTITY_FIELDS_NULL_MEANS:
                # A NullMeans field is satisfied by PRESENCE alone (even a
                # null value) in at least one block — null is the declared
                # state, not a finding.
                satisfied = any(status != "absent" for status in per_block.values())
            return satisfied, per_block

        results = {f: _resolve(f) for f in tfs.TORCH_IDENTITY_FIELDS}
        missing = {f: blocks for f, (ok, blocks) in results.items() if not ok}
        self.assertFalse(
            missing,
            f"TORCH_IDENTITY_FIELDS entries unsatisfied in a real --dry-run dump — per-field, "
            f"per-block status (absent/null/present) across provenance/args/finetune_step: "
            f"{missing}",
        )


# ============================================================================
# unit 63 H4b — finetune-run A/B merger tests (docs-ci domain).
# ============================================================================


# Unit-63 adversarial-audit finding 2 (merger half): the finetune-run tier
# now ALSO emits finetune-step's exact `*_fused_dispatches`/
# `*_eager_dispatches` (and, for the one `CASCADE_BASES` member,
# `*_declined_dispatches`) counter pairs, verbatim field names.
#
# Unit-63 round-3 audit, docs-ci class fix: this dict used to be
# HAND-TYPED, and its own hand-typing is exactly how `adamw_{fused,eager}_
# dispatches` fell out of coverage here (block 1's own reproduction) even
# though `report.rs`/`finetune_step.rs` have emitted it for months, and how
# the "alloff" shape below used to assert EVERY pair reads `fused == 0` --
# a hand-rolled assumption `fixtures/finetune_run_golden/modernbert_alloff.json`
# (a REAL alloff leg) directly contradicts: `finetune_run_ab.sh`'s own
# documented convention disables ONLY `attention_block_flash` and
# `adamw_step_fused`, so a real alloff leg's `ln`/`rope`/`softmax`/`geglu`/
# `lora_linear` all stay FUSED. Both entries below are now read DIRECTLY off
# the two committed goldens' own dispatch-counter fields (see
# `_golden_dispatch_counters`) rather than hand-kept literals -- a producer
# field addition changes these DERIVED dicts automatically the next time the
# golden is regenerated, never silently leaving a NEW field uncovered again.
def _golden_dispatch_counters(name):
    """Every `*_fused_dispatches`/`*_eager_dispatches`/`*_declined_dispatches`
    key golden `name`'s own `finetune_run` tier carries, read DIRECTLY off
    that REAL, committed report -- see `load_golden`'s own doc.
    """
    tier = load_golden(name)["tiers"]["finetune_run"]
    return {k: v for k, v in tier.items() if k.endswith(("_fused_dispatches", "_eager_dispatches", "_declined_dispatches"))}


_FINETUNE_RUN_DISPATCH_COUNTERS = {
    # `modernbert_fused.json` -- CONTRACT 63 Frame pre-registers the flash
    # cascade as the `fused` arm's own admitted branch (coordinator
    # correction, unit-63 round-3 audit): `attention_block_flash=840/0`
    # fires, `attention_block=0/0` is ABSORBED (its own `admit` call is
    # never reached -- `report.rs`'s own field doc), `ln`/`geglu`/
    # `lora_linear` independently fused, `adamw=6720/0`. Real counts
    # composited from the committed CUDA artifacts named in this golden's
    # own `PROVENANCE.md` ("Coordinator correction" section) -- no CUDA
    # device exists in this environment to run a genuine `flash_compiled:
    # true` leg against, so the counter fields are cited, never
    # hand-invented.
    "fused": _golden_dispatch_counters("modernbert_fused"),
    # `modernbert_alloff.json` (corrected, same PROVENANCE.md section) --
    # `ln=1710/0`, `rope=0/0`, `softmax=0/0`, `geglu=840/0`,
    # `lora_linear=3360/0` (all unaffected by either disable -- the
    # class-fix discovery `ALLOFF_DISABLED_OP_BASES`'s own doc explains);
    # `attention_block=840/0` -- the disabled flash cascade falls through
    # to the block arm's own, still-ACTIVE fused kernel (the positive
    # training-path proof for this arm, per the coordinator's cascade-
    # absorption correction -- NOT the eager shape an earlier, tiny
    # head_dim=16 fixture produced); `attention_block_flash=0/840` and
    # `adamw=0/6720` are the real disabled-kernel fallback counts.
    "alloff": _golden_dispatch_counters("modernbert_alloff"),
}


def _finetune_run_tier(arm="fused", **overrides):
    """A fully-populated `FinetuneRunTier`-shaped dict — every
    `FINETUNE_RUN_IDENTITY_FIELDS` entry, every `PROVENANCE_FIELDS` entry
    (report.rs), the three premise legs (`admission_is_dense`,
    `train_probe_series` -- amendment 2026-08-29b: the MERGER derives
    `learning_happened_delta` from this raw series, never a pre-derived
    scalar -- `tie_fraction`), the measurement fields
    (`final_epoch`, `held_out_example_mean`, `held_out_count`,
    `final_loss_diagnostic`, `trajectory`), and (unit-63 audit finding 2) a
    CLEAN `_FINETUNE_RUN_DISPATCH_COUNTERS` set matching `arm`. Defaults to
    MNRL (`margin=None`, `temperature=20.0`) and a CLEAN premise (padded
    transport, learning happened, no ties) — every mutant test below
    overrides exactly the one field it means to break.

    Unit-63 round-3 audit, docs-ci class fix: the STRUCTURAL base is now
    `load_golden("bert_fused")` -- a REAL, committed `jammi-bench
    finetune-run` report — rather than a second hand-typed field list; every
    field that report's own struct serializes is therefore present here by
    construction (see `load_golden`'s own doc for the exact class of bug
    this closes). The identity/provenance/premise/measurement literal below
    then overrides every field to this suite's own predictable-for-testing
    values, UNCHANGED from before this fix — the risk this golden closes is
    a MISSING field name, never a specific numeric value.
    """
    tier = copy.deepcopy(load_golden("bert_fused")["tiers"]["finetune_run"])
    tier.update({
        "seed": 42,
        "batch": 32,
        "seq": 64,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.05,
        "margin": None,
        "target_modules": ["Wqkv", "Wo", "Wi"],
        # unit-63 round-4 audit F-1: the fused arm's own dispatch counters
        # (`_FINETUNE_RUN_DISPATCH_COUNTERS["fused"]`, folded in below)
        # claim a positive `attention_block_flash_fused_dispatches` --
        # `flash_capability_gates` admits BF16 only, so `bf16` is the ONLY
        # dtype this tier's own counters can be self-consistent under
        # (`finetune_run_dispatch_proof_violations`'s new arm-agnostic
        # consistency premise). `finetune_run_ab.sh` now passes
        # `--backbone-dtype bf16` unconditionally on every real leg for
        # exactly this reason.
        "backbone_dtype": "bf16",
        "checkpoint_config_sha256": "cfg-sha",
        "checkpoint_weights_sha256": "weights-sha",
        "checkpoint_weights_size_bytes": 12345,
        "max_grad_norm": None,
        "warmup": None,
        "row_lengths": None,
        "epochs": 1,
        "lr": 0.0002,
        "schedule": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "grad_accum": 1,
        "validation_fraction": 0.1,
        "train_pairs_file_sha256": "train-pairs-file-sha",
        "heldout_ids_sha256": "heldout-ids-sha",
        "heldout_pairs_sha256": "heldout-pairs-sha",
        "heldout_batch_partition_sha256": "partition-sha",
        "embedding_loss": "mnrl",
        "temperature": 20.0,
        "matryoshka_dims": [],
        "early_stopping_patience": 10000,
        "early_stopping_metric": "val_loss",
        "eval_cadence": 1,
        # provenance (PROVENANCE_FIELDS)
        "arm": arm,
        "device_name": "cuda:0-fixture",
        "kernels_disabled_requested": [] if arm == "fused" else ["attention_block_flash", "adamw_step_fused"],
        "kernels_disabled_fired": [] if arm == "fused" else ["attention_block_flash", "adamw_step_fused"],
        "flash_compiled": True,
        "build_features": ["cuda"],
        "attention_arm": "fused" if arm == "fused" else "eager",
        "split_rule": "positional_fraction_split",
        "batched_forward": True,
        "steps_measured": 100,
        # premise legs (amendment 2026-08-29b: train_probe_series, index 0
        # the untrained-init probe, one entry per epoch -- this tier's own
        # default "epochs": 1 means 2 entries; the merger derives
        # learning_happened_delta = series[0] - series[-1] = 0.05)
        "admission_is_dense": False,
        "train_probe_series": [0.55, 0.5],
        "tie_fraction": 0.0,
        # measurements
        "final_epoch": 0,
        "held_out_example_mean": 0.5,
        "held_out_count": 128,
        "final_loss_diagnostic": 0.4,
        "trajectory": [
            {
                "epoch": 0,
                "held_out_mean": 0.5,
                "held_out_tie_fraction": 0.0,
                "held_out_batch_partition_sha256": "partition-sha",
            }
        ],
    })
    tier.update(_FINETUNE_RUN_DISPATCH_COUNTERS.get(arm, _FINETUNE_RUN_DISPATCH_COUNTERS["fused"]))
    tier.update(overrides)
    return tier


def _write_finetune_run_leg(raw_dir, seed, arm, repeat, tier, exit_code="0"):
    base = os.path.join(raw_dir, f"seed{seed}__{arm}__{repeat}")
    with open(base + ".json", "w") as fh:
        json.dump({"tiers": {"finetune_run": tier}}, fh)
    with open(base + ".exit", "w") as fh:
        fh.write(exit_code)
    with open(base + ".stderr", "w") as fh:
        fh.write("")


class GoldenProducerAnchoredFieldSetTests(unittest.TestCase):
    """The unit-63 round-3 audit's own class fix, pinned mechanically: the
    SET of `*_fused_dispatches`/`*_eager_dispatches`/`*_declined_dispatches`
    base names a REAL, committed `jammi-bench finetune-run` report carries
    must equal exactly what `ab_merge.ALL_BASES` classifies -- neither side
    a strict subset of the other. `adamw` fell out of `ALL_BASES` for
    months despite every real report emitting it (block 1's own
    reproduction); this test REDs the instant that gap reopens, for THIS
    base or a future one, rather than waiting for a real leg to hit
    `dispatch_pairs`'s own `KeyError` in a live sweep.

    All three committed goldens are read (`bert_fused`, `modernbert_fused`,
    `modernbert_alloff` — see the latter two's own `PROVENANCE.md` section
    for why their dispatch-counter fields are a documented composite of
    real committed CUDA artifacts rather than a fresh CPU-hermetic run) —
    a single golden would still catch a MISSING field (every
    `FinetuneRunTier` field is unconditionally serialized regardless of
    architecture, see `report.rs`), but reading all three is a stronger,
    still entirely real-data pin: no ONE golden alone could silently drift
    to "only ever has 8 of the 9 real bases" without another golden's own
    set disagreeing with it.
    """

    def test_golden_dispatch_pair_bases_equal_all_bases(self):
        for name in ("bert_fused", "modernbert_fused", "modernbert_alloff"):
            tier = load_golden(name)["tiers"]["finetune_run"]
            discovered = {
                key[: -len("_fused_dispatches")]
                for key in tier
                if key.endswith("_fused_dispatches")
            }
            self.assertEqual(
                discovered,
                ab_merge.ALL_BASES,
                f"{name}.json's own *_fused_dispatches base set no longer matches "
                f"ab_merge.ALL_BASES -- a producer field addition/removal REDs here "
                f"(regenerate the golden and update ALL_BASES together, never one alone): "
                f"golden-only={discovered - ab_merge.ALL_BASES!r} "
                f"ALL_BASES-only={ab_merge.ALL_BASES - discovered!r}",
            )

    def test_golden_dispatch_pairs_classify_cleanly_via_dispatch_pairs(self):
        """Not just the base-name SET (above) -- `ab_merge.dispatch_pairs`
        itself, the REAL function a merge calls, must not raise on either
        golden's own `finetune_run` tier (the exact mechanism `KeyError`d on
        `adamw` before block 1's fix).
        """
        for name in ("bert_fused", "modernbert_fused", "modernbert_alloff"):
            tier = load_golden(name)["tiers"]["finetune_run"]
            pairs = ab_merge.dispatch_pairs(tier)  # must not raise
            self.assertEqual({base for base, _fused, _fallback in pairs}, ab_merge.ALL_BASES)

    def test_golden_modernbert_composites_clear_the_dispatch_proof_gate(self):
        """Unit-63 round-5 audit (H5), renaming/re-scoping round-4's
        `test_golden_modernbert_legs_clear_the_full_dispatch_proof_gate`:
        `modernbert_fused.json`/`modernbert_alloff.json` are STAGED-CLOSURE
        composites, NOT real, producer-emittable legs (see
        `PROVENANCE.md`'s "Emittability status" section -- their
        `checkpoint_config_sha256` names a `head_dim=16` checkpoint that
        cannot arithmetically produce their own nonzero flash/
        attention-block counters, and their `batch`/`seq`/`steps_measured`
        cannot arithmetically produce those counters either). This test
        pins only that the FIELD-NAME SET these composites carry clears
        `finetune_run_dispatch_proof_violations` -- the merger's
        schema-shape acceptance gate -- run each one DIRECTLY off the
        committed JSON (never `_finetune_run_tier`'s own hand-overridden
        literal), including the arm-agnostic counters-vs-`backbone_dtype`
        consistency premise and the fused arm's own bf16 premise. It does
        NOT, and never did, certify that a real invocation could emit
        either file; `PROVENANCE.md`'s "Supersession plan" names the one
        thing that will (a real head_dim-64 probe leg), which also replaces
        these two committed files verbatim. `bert_fused.json` is
        deliberately excluded here -- a real CPU-hermetic BERT leg with no
        fused attention-block kernel at all (`flash_compiled: false`) was
        never claimed to clear the flash-cascade-arm's own premise; it is
        this suite's structural field-set base (`_finetune_run_tier`), not
        a leg exercising this gate.
        """
        fused_tier = load_golden("modernbert_fused")["tiers"]["finetune_run"]
        self.assertEqual(ab_merge.finetune_run_dispatch_proof_violations("fused", fused_tier), [])
        alloff_tier = load_golden("modernbert_alloff")["tiers"]["finetune_run"]
        self.assertEqual(ab_merge.finetune_run_dispatch_proof_violations("alloff", alloff_tier), [])


class SignTestMirrorTests(unittest.TestCase):
    """`ab_merge.sign_test` is a Python u128-equivalent mirror of
    `jammi_numerics::stats::sign_test::sign_test` (branch
    `numerics/63-sign-test`) — every case here is transcribed directly from
    that module's own `tests/it/stats.rs` cases, hand-computation comments
    included, so a divergence between the two implementations is caught by
    literally re-running the SAME arithmetic this suite's Rust twin already
    pins.
    """

    def test_golden_n12_k11_pinned_cell(self):
        # CONTRACT H2 / PLAN v2 delta 3's pre-registered decision cell:
        # t=11, tail=C(12,11)+C(12,12)=12+1=13, p=2*13/4096=13/2048.
        diffs = [1.0] * 11 + [-1.0]
        r = ab_merge.sign_test(diffs)
        self.assertEqual((r["n"], r["n_pos"], r["n_neg"], r["ties"]), (12, 11, 1, 0))
        expected = 13.0 / 2048.0
        self.assertEqual(
            r["p_value"].hex(),
            expected.hex(),
            f"golden cell (n=12, k=11) must equal 13/2048 bit-for-bit, got {r['p_value']}",
        )
        self.assertLess(abs(r["p_value"] - 0.0064), 0.0005)

    def test_exact_small_n_all_same_sign(self):
        r = ab_merge.sign_test([2.0, 3.0, 1.0, 0.5, 7.0])
        self.assertEqual((r["n"], r["n_pos"], r["n_neg"], r["ties"]), (5, 5, 0, 0))
        self.assertEqual(r["p_value"].hex(), (2.0 / 32.0).hex())

    def test_exact_small_n_one_dissent(self):
        r = ab_merge.sign_test([2.0, 3.0, 1.0, 0.5, -7.0])
        self.assertEqual((r["n"], r["n_pos"], r["n_neg"], r["ties"]), (5, 4, 1, 0))
        self.assertEqual(r["p_value"].hex(), (12.0 / 32.0).hex())

    def test_exact_small_n_balanced_saturates_at_one(self):
        r = ab_merge.sign_test([1.0, 1.0, -1.0, -1.0])
        self.assertEqual((r["n"], r["n_pos"], r["n_neg"], r["ties"]), (4, 2, 2, 0))
        self.assertEqual(r["p_value"], 1.0)

    def test_two_sided_symmetry(self):
        diffs = [3.0, 1.5, -2.0, 4.0, 0.2, -0.1, 5.0, -6.0, 0.9, 2.2, -1.1, 8.0]
        flipped = [-d for d in diffs]
        r = ab_merge.sign_test(diffs)
        rf = ab_merge.sign_test(flipped)
        self.assertEqual(r["n"], rf["n"])
        self.assertEqual(r["n_pos"], rf["n_neg"])
        self.assertEqual(r["n_neg"], rf["n_pos"])
        self.assertEqual(r["p_value"].hex(), rf["p_value"].hex())

    def test_reports_ties_without_dropping_them(self):
        r = ab_merge.sign_test([1.0, 1.0, 1.0, 0.0, 0.0, -1.0])
        self.assertEqual(r["n"], 4, "ties must be excluded from n, not folded in")
        self.assertEqual((r["n_pos"], r["n_neg"], r["ties"]), (3, 1, 2))
        self.assertEqual(r["p_value"].hex(), (10.0 / 16.0).hex())

    def test_refuses_empty_input(self):
        with self.assertRaises(ab_merge.SignTestError) as ctx:
            ab_merge.sign_test([])
        self.assertIn("n=0", str(ctx.exception))

    def test_refuses_all_ties_with_a_distinct_message(self):
        with self.assertRaises(ab_merge.SignTestError) as ctx:
            ab_merge.sign_test([0.0, 0.0, 0.0])
        msg = str(ctx.exception)
        self.assertTrue("tie" in msg or "tied" in msg, msg)

    def test_refuses_nan(self):
        with self.assertRaises(ab_merge.SignTestError):
            ab_merge.sign_test([1.0, float("nan"), -1.0])

    def test_admits_infinite_as_a_well_defined_sign(self):
        r = ab_merge.sign_test([float("inf"), float("-inf"), 1.0, 2.0, 3.0])
        self.assertEqual((r["n"], r["n_pos"], r["n_neg"], r["ties"]), (5, 4, 1, 0))

    def test_negative_control_balanced_is_not_significant(self):
        diffs = [1.0] * 6 + [-1.0] * 6
        r = ab_merge.sign_test(diffs)
        self.assertEqual(r["n"], 12)
        self.assertGreater(r["p_value"], 0.5)

    def test_is_invariant_to_input_order(self):
        ordered = [1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        shuffled = list(reversed(ordered))
        shuffled = shuffled[3:] + shuffled[:3]  # rotate_left(3)
        self.assertNotEqual(ordered, shuffled)
        a = ab_merge.sign_test(ordered)
        b = ab_merge.sign_test(shuffled)
        self.assertEqual((a["n"], a["n_pos"], a["n_neg"], a["ties"]), (b["n"], b["n_pos"], b["n_neg"], b["ties"]))
        self.assertEqual(a["p_value"].hex(), b["p_value"].hex())


class FinetuneRunArmPremiseMutantTests(unittest.TestCase):
    """`finetune_run_arm_premise_violations` — one mutant per premise leg
    (CONTRACT Frame / H4: admission_is_dense / learning-happened / tie cap /
    schedule, conjunctive). A clean tier clears all four; each mutation
    trips exactly the leg it targets, proving none of the four checks is
    vacuous.
    """

    def test_clean_tier_has_no_violations(self):
        self.assertEqual(ab_merge.finetune_run_arm_premise_violations("fused", _finetune_run_tier()), [])

    def test_dense_leg_is_a_violation(self):
        tier = _finetune_run_tier(admission_is_dense=True)
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("admission_is_dense" in m for m in v), v)

    def test_decaying_schedule_is_a_violation(self):
        # unit-63 round-7 audit advisory (d): amendment 2026-08-29b item 4's
        # decaying-schedule ban had no mechanical enforcement -- `schedule`
        # is already recorded on the tier, so a non-"constant" schedule must
        # be refused here, citing that item.
        tier = _finetune_run_tier(schedule="cosine")
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("schedule" in m and "item 4" in m for m in v), v)

    # amendment 2026-08-29b: the learning-happened premise is now DERIVED
    # from the raw `train_probe_series` (series[0] - series[-1] > floor),
    # never read off a pre-derived scalar. One mutant per typed refusal
    # (floor-fail, missing-series, short-series, non-finite, v1-scalar-only,
    # length-vs-epochs mismatch) -- proving none of the six is vacuous.

    def test_floor_fail_series_is_a_violation(self):
        tier = _finetune_run_tier(train_probe_series=[0.5, 0.5])  # delta == floor, not strictly >
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("does not clear the floor" in m for m in v), v)

        tier2 = _finetune_run_tier(train_probe_series=[0.5, 0.51])  # delta negative (got WORSE)
        v2 = ab_merge.finetune_run_arm_premise_violations("fused", tier2)
        self.assertTrue(any("does not clear the floor" in m for m in v2), v2)

    def test_missing_series_is_a_violation(self):
        tier = _finetune_run_tier(train_probe_series=None)
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("missing" in m for m in v), v)

    def test_short_series_is_a_violation(self):
        tier = _finetune_run_tier(train_probe_series=[0.5])
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("need at least 2" in m for m in v), v)

    def test_non_finite_series_entry_is_a_violation(self):
        tier = _finetune_run_tier(train_probe_series=[0.5, float("nan")])
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("non-finite" in m for m in v), v)

        tier2 = _finetune_run_tier(train_probe_series=[0.5, float("inf")])
        v2 = ab_merge.finetune_run_arm_premise_violations("fused", tier2)
        self.assertTrue(any("non-finite" in m for m in v2), v2)

    def test_length_equals_epochs_series_is_a_violation(self):
        # unit-63 round-7 audit finding 3: a series whose length equals
        # `epochs` itself (never `epochs + 1`) is the v1 probe bug's EXACT
        # shape (the baseline excluded the init point) -- this used to clear
        # the SHORT check (>= 2 entries) unchallenged whenever epochs >= 2.
        tier = _finetune_run_tier(epochs=3, train_probe_series=[0.55, 0.53, 0.51])  # len == epochs, not epochs+1
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("init-anchored" in m and "epochs=3" in m for m in v), v)

    def test_length_epochs_plus_two_series_is_a_violation(self):
        # The over-long direction: len(series) == epochs + 2, a
        # truncation/duplication producer bug in the OTHER direction, also
        # refused (never silently trusted just because it is longer, not
        # shorter, than the SHORT check's floor).
        tier = _finetune_run_tier(epochs=3, train_probe_series=[0.55, 0.53, 0.51, 0.50, 0.49])  # epochs+2
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("init-anchored" in m and "epochs=3" in m for m in v), v)

    def test_length_epochs_plus_one_series_is_clean(self):
        # The correct, init-anchored shape: exactly epochs+1 entries.
        tier = _finetune_run_tier(epochs=3, train_probe_series=[0.55, 0.53, 0.51, 0.50])  # epochs+1, delta=0.05>0
        self.assertEqual(ab_merge.finetune_run_arm_premise_violations("fused", tier), [])

    def test_v1_scalar_only_leg_is_invalid_never_readjudicated(self):
        # A leg carrying the OLD scalar field with no series at all -- a
        # producer-version mismatch (a v1-era report, e.g. the committed
        # campaign-v1 evidence), never silently re-adjudicated under the
        # corrected series-derived rule even though the OLD scalar itself
        # would have cleared the floor.
        tier = _finetune_run_tier(learning_happened_delta=0.05, train_probe_series=None)
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("producer-version mismatch" in m for m in v), v)

    def test_saturated_tie_fraction_is_a_violation(self):
        # C16's own hinge-saturation shape: tie_fraction -> 1.0.
        tier = _finetune_run_tier(tie_fraction=1.0)
        v = ab_merge.finetune_run_arm_premise_violations("fused", tier)
        self.assertTrue(any("tie_fraction" in m for m in v), v)

    def test_tie_fraction_just_under_cap_is_clean(self):
        tier = _finetune_run_tier(tie_fraction=ab_merge.FINETUNE_RUN_TIE_FRACTION_CAP - 0.01)
        self.assertEqual(ab_merge.finetune_run_arm_premise_violations("fused", tier), [])


class FinetuneRunDispatchProofMutantTests(unittest.TestCase):
    """`finetune_run_dispatch_proof_violations` (unit-63 audit finding 2's
    merger half) — one mutant per arm, plus the "missing entirely" and
    "malformed pair" carve-outs.
    """

    def test_clean_fused_tier_has_no_violations(self):
        self.assertEqual(
            ab_merge.finetune_run_dispatch_proof_violations("fused", _finetune_run_tier(arm="fused")), []
        )

    def test_clean_alloff_tier_has_no_violations(self):
        self.assertEqual(
            ab_merge.finetune_run_dispatch_proof_violations("alloff", _finetune_run_tier(arm="alloff")), []
        )

    def test_fused_arm_failing_required_pair_is_a_violation(self):
        # ln reads (0, 0) -- REQUIRED_PAIRS demands fused > 0.
        tier = _finetune_run_tier(arm="fused", ln_fused_dispatches=0)
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("fused-dispatch proof" in m for m in v), v)

    def test_fused_arm_with_a_real_eager_fallback_is_a_violation(self):
        # Rule 1: ANY pair with a nonzero fallback count is a hard fail.
        tier = _finetune_run_tier(arm="fused", ln_eager_dispatches=3)
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("fused-dispatch proof" in m for m in v), v)

    def test_fused_arm_with_flash_compiled_false_is_an_invalid_premise(self):
        # Unit-63 round-3 audit, coordinator correction: CONTRACT 63 Frame
        # pre-registers the flash cascade as this arm's own admitted
        # branch -- a build that cannot compile it in can never exercise
        # the pre-registered differential, an INVALID premise regardless of
        # what the (otherwise clean) dispatch counters read.
        tier = _finetune_run_tier(arm="fused", flash_compiled=False)
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("flash_compiled=False" in m for m in v), v)

    def test_fused_arm_with_non_bf16_dtype_and_positive_flash_counter_is_a_contradiction(self):
        # unit-63 round-4 audit F-1, check 0 (arm-agnostic): the DEFAULT
        # fused tier's own dispatch counters already claim
        # `attention_block_flash_fused_dispatches=840` -- overriding only
        # `backbone_dtype` (never the counters) exercises the
        # counters-vs-declared-premise contradiction directly, before the
        # fused arm's own (separate) dtype premise check below is ever
        # reached.
        tier = _finetune_run_tier(arm="fused", backbone_dtype="f32")
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(
            any("counters claim a dispatch the declared dtype forbids" in m for m in v), v
        )

    def test_alloff_arm_with_non_bf16_dtype_and_positive_flash_counter_is_a_contradiction(self):
        # Same check 0, exercised on the OTHER arm -- arm-agnostic means
        # arm-agnostic: an `alloff` leg whose own counters somehow claim a
        # positive flash-fused dispatch is caught here, before ever
        # reaching the alloff-specific disabled-op proof below (which would
        # ALSO flag this leg, for a different reason -- 'disabled but
        # fired' -- but this check fires first and is the one actually
        # exercised, since it is unconditional on arm).
        tier = _finetune_run_tier(arm="alloff", backbone_dtype="f32", attention_block_flash_fused_dispatches=5)
        v = ab_merge.finetune_run_dispatch_proof_violations("alloff", tier)
        self.assertTrue(
            any("counters claim a dispatch the declared dtype forbids" in m for m in v), v
        )

    def test_fused_arm_with_non_bf16_dtype_is_an_invalid_premise_even_with_clean_counters(self):
        # unit-63 round-4 audit F-1, the FUSED arm's own defining premise
        # (independent of check 0 above): force `attention_block_flash_
        # fused_dispatches` to 0 (the block arm's own absorption picking up
        # the slack instead, same shape as
        # `test_fused_arm_with_flash_never_dispatched_is_a_violation`) so
        # check 0 does NOT fire, isolating the fused-specific dtype premise
        # check that DOES fire regardless of the (otherwise clean-looking)
        # counters.
        tier = _finetune_run_tier(
            arm="fused",
            backbone_dtype="f32",
            attention_block_flash_fused_dispatches=0,
            attention_block_flash_declined_dispatches=0,
            attention_block_fused_dispatches=840,
        )
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("fused: backbone_dtype='f32'" in m for m in v), v)
        self.assertTrue(any("INVALID premise" in m for m in v), v)

    def test_fused_arm_with_flash_never_dispatched_is_a_violation(self):
        # `fused_proof`'s own absorption rule tolerates EITHER the flash
        # cascade or the block arm firing -- correct for finetune-step's
        # own flash-vs-block A/B, but the finetune-run `fused` arm
        # specifically claims to run the flash-cascade branch. A leg where
        # the block arm picked up the slack instead (flash never fired,
        # attention_block fired FUSED on its own) must still fail this
        # arm's own, stricter proof.
        tier = _finetune_run_tier(
            arm="fused",
            attention_block_flash_fused_dispatches=0,
            attention_block_flash_declined_dispatches=0,
            attention_block_fused_dispatches=840,
        )
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("attention_block_flash_fused_dispatches=0" in m for m in v), v)

    def test_alloff_arm_with_real_production_dispatch_shape_is_clean(self):
        # unit-63 round-3 audit, class-fix discovery: the default alloff
        # base is now the REAL `modernbert_alloff.json` golden's own
        # dispatch shape (ln/rope/softmax/geglu/lora_linear all FUSED,
        # attention_block/adamw/attention_block_flash all correctly
        # disabled) -- no longer "all zero fused", but still clean.
        self.assertEqual(
            ab_merge.finetune_run_dispatch_proof_violations("alloff", _finetune_run_tier(arm="alloff")), []
        )

    def test_alloff_arm_with_attention_block_fused_zero_is_a_violation(self):
        # unit-63 round-3 audit block 4, coordinator correction: the
        # positive training-path proof for `alloff` is `attention_block`'s
        # own FUSED count (it is NOT itself named in the disable list, only
        # `attention_block_flash` is, so it must remain an ACTIVE,
        # undisabled fused kernel on a real checkpoint) -- a leg where it
        # reads `fused == 0` (the disabled flash cascade failing to fall
        # through to a live fused kernel) is a violation, never tolerated
        # as "maybe it fell back to eager instead" (an EARLIER, incorrect
        # shape of this same proof).
        tier = _finetune_run_tier(
            arm="alloff", attention_block_fused_dispatches=0, attention_block_eager_dispatches=4
        )
        v = ab_merge.finetune_run_dispatch_proof_violations("alloff", tier)
        self.assertTrue(any("attention_block_fused_dispatches=0" in m for m in v), v)

    def test_alloff_arm_with_ln_and_geglu_fused_is_not_a_violation(self):
        # unit-63 round-3 audit, class-fix discovery: `ln`/`geglu` are NOT
        # among `ALLOFF_DISABLED_OP_BASES` -- `finetune_run_ab.sh`'s own
        # documented `alloff` convention disables ONLY `attention_block_flash`
        # and `adamw_step_fused` -- so a real alloff leg's `ln`/`geglu`
        # staying fused (exactly what `fixtures/finetune_run_golden/
        # modernbert_alloff.json`, a REAL leg, shows) is not this arm's
        # business at all. The pre-fix blanket "every pair must be
        # fused == 0" rule would have wrongly flagged this.
        tier = _finetune_run_tier(arm="alloff", ln_fused_dispatches=2, geglu_fused_dispatches=5)
        v = ab_merge.finetune_run_dispatch_proof_violations("alloff", tier)
        self.assertEqual(v, [])

    def test_alloff_arm_with_multiple_disabled_ops_leaking_fused_names_each(self):
        # Both genuinely-disabled ops (`attention_block_flash`,
        # `adamw_step_fused`) leaking a fused count at once must each be
        # named, independently -- not merged into one opaque message.
        tier = _finetune_run_tier(
            arm="alloff", attention_block_flash_fused_dispatches=3, adamw_fused_dispatches=7
        )
        v = ab_merge.finetune_run_dispatch_proof_violations("alloff", tier)
        self.assertTrue(any("attention_block_flash shows 3" in m for m in v), v)
        self.assertTrue(any("adamw shows 7" in m for m in v), v)

    def test_missing_dispatch_counters_entirely_is_a_violation_never_assumed_good(self):
        # An older-producer leg predating this emission: strip every
        # counter field -- `dispatch_pairs` then discovers nothing at all,
        # which must NOT be silently treated as "ran the claimed arm
        # cleanly" for either arm.
        base_tier = _finetune_run_tier(arm="fused")
        stripped = {k: v for k, v in base_tier.items() if not k.endswith(("_fused_dispatches", "_eager_dispatches"))}
        v_fused = ab_merge.finetune_run_dispatch_proof_violations("fused", stripped)
        self.assertTrue(any("no *_fused_dispatches" in m for m in v_fused), v_fused)

        alloff_tier = _finetune_run_tier(arm="alloff")
        stripped_alloff = {
            k: v for k, v in alloff_tier.items() if not k.endswith(("_fused_dispatches", "_eager_dispatches"))
        }
        v_alloff = ab_merge.finetune_run_dispatch_proof_violations("alloff", stripped_alloff)
        self.assertTrue(any("no *_fused_dispatches" in m for m in v_alloff), v_alloff)

    def test_solo_counter_schema_error_is_caught_not_propagated(self):
        tier = _finetune_run_tier(arm="fused")
        del tier["ln_eager_dispatches"]  # a fused key with no fallback sibling
        v = ab_merge.finetune_run_dispatch_proof_violations("fused", tier)
        self.assertTrue(any("dispatch-pair schema error" in m for m in v), v)


class FinetuneRunCrossSeedHomogeneityTests(unittest.TestCase):
    """`finetune_run_cross_seed_homogeneity_violations` (unit-63 audit
    finding 3): every OTHER identity check in this section compares
    fused-vs-alloff WITHIN one seed only; this one compares every leg
    entering the decision against every OTHER leg, `seed` itself excepted.
    """

    def _identity(self, seed, arm, **overrides):
        return (
            f"seed {seed} {arm} r1",
            ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm=arm, seed=seed, **overrides)),
        )

    def test_fewer_than_two_legs_is_clean(self):
        self.assertEqual(ab_merge.finetune_run_cross_seed_homogeneity_violations([]), [])
        self.assertEqual(
            ab_merge.finetune_run_cross_seed_homogeneity_violations([self._identity(1, "fused")]), []
        )

    def test_homogeneous_twelve_seeds_is_clean(self):
        legs = [self._identity(seed, arm) for seed in range(1, 13) for arm in ("fused", "alloff")]
        self.assertEqual(ab_merge.finetune_run_cross_seed_homogeneity_violations(legs), [])

    def test_two_fixture_split_is_a_violation(self):
        # Empirical reproduction: 6 seeds run against one held-out text, 6
        # against a different one -- each seed's own fused/alloff pair
        # internally agrees (heldout_pairs_sha256 matches within a seed), so
        # the existing per-seed check alone would see nothing wrong.
        legs = []
        for seed in range(1, 7):
            for arm in ("fused", "alloff"):
                legs.append(self._identity(seed, arm, heldout_pairs_sha256="fixture-a"))
        for seed in range(7, 13):
            for arm in ("fused", "alloff"):
                legs.append(self._identity(seed, arm, heldout_pairs_sha256="fixture-b"))
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(legs)
        self.assertTrue(any("heldout_pairs_sha256" in m for m in v), v)

    def test_single_divergent_field_on_one_seed_names_it(self):
        legs = [self._identity(seed, arm) for seed in range(1, 13) for arm in ("fused", "alloff")]
        # Seed 7's own two legs both agree with EACH OTHER (so the existing
        # cross-arm check is clean) but diverge from every other seed.
        legs = [
            leg
            if not leg[0].startswith("seed 7 ")
            else self._identity(7, leg[0].split(" ")[2], checkpoint_weights_sha256="different-checkpoint")
            for leg in legs
        ]
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(legs)
        self.assertEqual(len(v), 1)
        self.assertIn("checkpoint_weights_sha256", v[0])
        self.assertIn("seed 7", v[0])

    def test_seed_field_itself_is_never_compared(self):
        legs = [self._identity(seed, arm) for seed in range(1, 5) for arm in ("fused", "alloff")]
        self.assertEqual(ab_merge.finetune_run_cross_seed_homogeneity_violations(legs), [])

    def test_missing_field_on_one_leg_is_its_own_divergent_group(self):
        clean = [self._identity(seed, arm) for seed in (1, 2, 3) for arm in ("fused", "alloff")]
        missing_field_tier = _finetune_run_tier(arm="fused", seed=4)
        del missing_field_tier["lora_dropout"]
        clean.append((f"seed 4 fused r1", ab_merge.finetune_run_leg_identity(missing_field_tier)))
        clean.append((f"seed 4 alloff r1", ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="alloff", seed=4))))
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(clean)
        self.assertTrue(any("lora_dropout" in m for m in v), v)

    # unit-63 round-3 audit block 3 -- `lr` is IDENTITY FIELD #17; the lr0
    # RED control's own legs run at `--lr 0` BY CONSTRUCTION, so comparing
    # `lr` across the FULL combined pool the way every other field is
    # compared would make ANY nonempty `lr0_labels` set unconditionally
    # INVALID. `lr0_labels` names which `leg_identities` entries are the
    # control's own -- `lr` is then compared WITHIN each of the main/
    # lr0-control pools separately, never across that boundary.

    def _lr0_identity(self, seed, arm, **overrides):
        overrides.setdefault("lr", 0.0)
        return (
            f"lr0 seed {seed} {arm}",
            ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm=arm, seed=seed, **overrides)),
        )

    def test_lr0_control_legs_diverging_on_lr_from_main_legs_is_not_a_violation(self):
        main_legs = [self._identity(seed, arm) for seed in range(1, 3) for arm in ("fused", "alloff")]
        lr0_legs = [self._lr0_identity(101, arm) for arm in ("fused", "alloff")]
        lr0_labels = {label for label, _fields in lr0_legs}
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(main_legs + lr0_legs, lr0_labels=lr0_labels)
        self.assertEqual(v, [], v)

    def test_lr0_control_leg_diverging_on_a_non_lr_field_is_still_a_violation(self):
        # The `lr` exception is NARROW -- an lr0-control leg diverging on
        # any OTHER field (its own defining premise aside) still collapses
        # this check, exactly like a main leg would.
        main_legs = [self._identity(seed, arm) for seed in range(1, 3) for arm in ("fused", "alloff")]
        lr0_legs = [
            self._lr0_identity(101, "fused"),
            self._lr0_identity(101, "alloff", checkpoint_weights_sha256="different-checkpoint"),
        ]
        lr0_labels = {label for label, _fields in lr0_legs}
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(main_legs + lr0_legs, lr0_labels=lr0_labels)
        self.assertTrue(any("checkpoint_weights_sha256" in m for m in v), v)

    def test_lr0_control_legs_diverging_on_lr_among_themselves_is_a_violation(self):
        # The `lr` exception drops the CROSS-group comparison only -- two
        # lr0-control legs must still agree with EACH OTHER on `lr`.
        main_legs = [self._identity(seed, arm) for seed in range(1, 3) for arm in ("fused", "alloff")]
        lr0_legs = [
            self._lr0_identity(101, "fused"),
            self._lr0_identity(101, "alloff", lr=0.5),  # a lr0-control leg NOT actually at lr=0
        ]
        lr0_labels = {label for label, _fields in lr0_legs}
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(main_legs + lr0_legs, lr0_labels=lr0_labels)
        self.assertTrue(any("'lr' diverges within the lr0-control pool" in m for m in v), v)

    def test_main_legs_diverging_on_lr_among_themselves_is_still_a_violation(self):
        # The `lr` exception is about the lr0-vs-main BOUNDARY only -- main
        # legs must still all agree with each other on `lr`.
        main_legs = [
            self._identity(1, "fused"),
            self._identity(1, "alloff"),
            self._identity(2, "fused", lr=0.001),
            self._identity(2, "alloff", lr=0.001),
        ]
        v = ab_merge.finetune_run_cross_seed_homogeneity_violations(main_legs)
        self.assertTrue(any("'lr' diverges within the main pool" in m for m in v), v)


class FinetuneRunCrossArmIdentityTests(unittest.TestCase):
    """Cross-arm identity check (`generic_leg_premise_violations` over
    `FINETUNE_RUN_IDENTITY_FIELDS`) — the fused/alloff legs of ONE seed must
    share every identity field; `arm`/`attention_arm` (provenance) are
    allowed, indeed expected, to differ.
    """

    def test_matching_identity_is_clean(self):
        fused = ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="fused"))
        alloff = ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="alloff"))
        self.assertEqual(
            ab_merge.generic_leg_premise_violations(
                ab_merge.FINETUNE_RUN_IDENTITY_FIELDS, fused, alloff, "fused", "alloff"
            ),
            [],
        )

    def test_differing_heldout_ids_sha256_is_a_violation(self):
        fused = ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="fused"))
        alloff = ab_merge.finetune_run_leg_identity(
            _finetune_run_tier(arm="alloff", heldout_ids_sha256="different-sha")
        )
        v = ab_merge.generic_leg_premise_violations(
            ab_merge.FINETUNE_RUN_IDENTITY_FIELDS, fused, alloff, "fused", "alloff"
        )
        self.assertTrue(any("heldout_ids_sha256" in m for m in v), v)

    def test_null_is_a_value_for_margin_and_temperature(self):
        # Both arms MNRL (margin=None both sides) must match cleanly --
        # a present null on BOTH sides is the stated premise, never MISSING.
        fused = ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="fused"))
        alloff = ab_merge.finetune_run_leg_identity(_finetune_run_tier(arm="alloff"))
        self.assertIsNot(fused["margin"], ab_merge._MISSING)
        self.assertIsNone(fused["margin"])
        self.assertEqual(
            ab_merge.generic_leg_premise_violations(
                ab_merge.FINETUNE_RUN_IDENTITY_FIELDS, fused, alloff, "fused", "alloff"
            ),
            [],
        )


class FinetuneRunDeterminismFloorTests(unittest.TestCase):
    """`build_finetune_run_report`'s r1/r2 determinism-floor reporting
    (CONTRACT H4/PLAN.md v2 delta 6): the delta is ALWAYS measured and
    reported; it is RED (a `determinism_floor.findings` entry, and the
    overall `status` collapses to `INVALID`) only when it exceeds the
    cross-seed spread of `d_i`.
    """

    def _write_seed(self, raw_dir, seed, fused_mean, alloff_mean, fused_r2_mean=None, alloff_r2_mean=None):
        fused_r2_mean = fused_mean if fused_r2_mean is None else fused_r2_mean
        alloff_r2_mean = alloff_mean if alloff_r2_mean is None else alloff_r2_mean
        _write_finetune_run_leg(
            raw_dir, seed, "fused", "r1", _finetune_run_tier(arm="fused", seed=seed, held_out_example_mean=fused_mean)
        )
        _write_finetune_run_leg(
            raw_dir,
            seed,
            "fused",
            "r2",
            _finetune_run_tier(arm="fused", seed=seed, held_out_example_mean=fused_r2_mean),
        )
        _write_finetune_run_leg(
            raw_dir,
            seed,
            "alloff",
            "r1",
            _finetune_run_tier(arm="alloff", seed=seed, held_out_example_mean=alloff_mean),
        )
        _write_finetune_run_leg(
            raw_dir,
            seed,
            "alloff",
            "r2",
            _finetune_run_tier(arm="alloff", seed=seed, held_out_example_mean=alloff_r2_mean),
        )

    def test_identical_r1_r2_never_reds(self):
        # unit-63 audit finding 1: the decision rule now requires exactly
        # the pre-registered 12 premise-clean seeds (else INVALID) --
        # 6-vs-6 keeps both n_pos/n_neg well under the 11-of-12 threshold,
        # so GREEN is still the right read regardless of mean sign.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 7):
                self._write_seed(raw_dir, seed, 0.30, 0.50)  # fused better: d = -0.20
            for seed in range(7, 13):
                self._write_seed(raw_dir, seed, 0.55, 0.45)  # alloff better: d = +0.10
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["determinism_floor"]["findings"], [])
        self.assertEqual(merged["determinism_floor"]["max_delta"], 0.0)
        self.assertEqual(merged["decision"]["clean_seed_count"], 12)
        self.assertEqual(merged["decision"]["concordant_direction"], "none")
        self.assertEqual(merged["status"], "GREEN")

    def test_delta_exceeding_cross_seed_spread_reds(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            # Three seeds whose d_i are tightly clustered (small cross-seed
            # spread), then one seed's OWN r1/r2 repeat disagrees by far
            # more than that spread -- the exact shape the floor exists to
            # catch (a nondeterminism the seed spread would otherwise mask).
            self._write_seed(raw_dir, 1, 0.400, 0.500)
            self._write_seed(raw_dir, 2, 0.401, 0.501)
            self._write_seed(raw_dir, 3, 0.399, 0.499, fused_r2_mean=0.700)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1, 2, 3], allow_missing_lr0_control=True)
        self.assertGreater(len(merged["determinism_floor"]["findings"]), 0)
        self.assertTrue(any("seed 3" in f and "fused" in f for f in merged["determinism_floor"]["findings"]))
        self.assertEqual(merged["status"], "INVALID")

    def test_fewer_than_two_d_values_falls_back_to_zero_spread(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_seed(raw_dir, 1, 0.40, 0.50, fused_r2_mean=0.41)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1], allow_missing_lr0_control=True)
        self.assertEqual(merged["cross_seed_spread"], 0.0)
        self.assertGreater(len(merged["determinism_floor"]["findings"]), 0)


class BuildFinetuneRunReportEndToEndTests(unittest.TestCase):
    """`build_finetune_run_report` / `ab_merge.main(["finetune-run", ...])`
    — the REAL entry point `finetune_run_ab.sh` invokes, driven against
    fixture leg directories (never a hand-rolled call to `sign_test` alone
    standing in for the merge).
    """

    def _write_clean_seed(self, raw_dir, seed, fused_mean, alloff_mean):
        for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                )

    def test_end_to_end_green_matches_direct_sign_test(self):
        # unit-63 audit finding 1: the decision rule is pre-registered FOR
        # exactly 12 premise-clean seeds (else INVALID) -- 7-vs-5 (fused
        # wins 7, alloff wins 5) keeps both n_pos/n_neg under the 11-of-12
        # threshold, so this stays GREEN regardless of the mean's sign
        # (the SAME shape `sign_test` itself is exercised over, just with
        # a real premise-clean seed count this time).
        seeds = list(range(1, 13))
        means = {
            1: (0.30, 0.50),
            2: (0.32, 0.48),
            3: (0.29, 0.55),
            4: (0.31, 0.47),
            5: (0.28, 0.52),
            6: (0.33, 0.49),
            7: (0.27, 0.53),
            8: (0.55, 0.40),
            9: (0.52, 0.38),
            10: (0.58, 0.42),
            11: (0.50, 0.35),
            12: (0.54, 0.39),
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                self._write_clean_seed(raw_dir, seed, *means[seed])
            merged, table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)

        self.assertEqual(merged["status"], "GREEN")
        self.assertEqual(merged["decision"]["clean_seed_count"], 12)
        self.assertEqual(merged["decision"]["n_pos"], 5)
        self.assertEqual(merged["decision"]["n_neg"], 7)
        self.assertEqual(merged["decision"]["concordant_direction"], "none")
        expected_d = [means[s][0] - means[s][1] for s in seeds]
        expected = ab_merge.sign_test(expected_d)
        self.assertEqual(merged["sign_test"]["n"], expected["n"])
        self.assertEqual(merged["sign_test"]["n_pos"], expected["n_pos"])
        self.assertEqual(merged["sign_test"]["n_neg"], expected["n_neg"])
        self.assertEqual(merged["sign_test"]["p_value"].__class__, float)
        self.assertEqual(merged["sign_test"]["p_value"], expected["p_value"])
        self.assertIn("sign_test:", table)
        self.assertEqual(len(merged["per_seed"]), len(seeds))
        for seed in seeds:
            self.assertIn("fused", merged["per_seed"][str(seed)]["trajectory"])
            self.assertIn("alloff", merged["per_seed"][str(seed)]["trajectory"])

    def test_a_single_seed_premise_violation_invalidates_the_whole_merge(self):
        seeds = [1, 2, 3]
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_clean_seed(raw_dir, 1, 0.30, 0.50)
            self._write_clean_seed(raw_dir, 2, 0.31, 0.49)
            # seed 3's fused leg saturates its tie fraction -- a single
            # tripped premise leg on ONE seed collapses the whole merge's
            # status, mirroring build_report's own INVALID carve-out.
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir, 3, "fused", repeat, _finetune_run_tier(arm="fused", seed=3, tie_fraction=1.0)
                )
                _write_finetune_run_leg(
                    raw_dir, 3, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=3)
                )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)

        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(merged["per_seed"]["3"]["leg_premise_violations"])
        # seeds 1/2 stay premise-clean and still contribute their own d_i to
        # the record, even though the OVERALL merge is INVALID.
        self.assertIsNotNone(merged["per_seed"]["1"]["d_i"])
        self.assertIsNotNone(merged["per_seed"]["2"]["d_i"])
        self.assertNotIn("3", merged["d_values"])

    def test_main_finetune_run_dispatch_writes_report_and_exits_0_on_green(self):
        # unit-63 audit finding 1: needs exactly the pre-registered 12
        # premise-clean seeds (else INVALID) -- 6-vs-6 stays under the
        # 11-of-12 threshold, so GREEN/exit-0 is still the right read.
        seeds = list(range(1, 13))
        means = [(0.30, 0.50), (0.32, 0.48), (0.29, 0.55), (0.31, 0.47), (0.28, 0.52), (0.33, 0.49)]
        means += [(0.55, 0.40), (0.52, 0.38), (0.58, 0.42), (0.50, 0.35), (0.54, 0.39), (0.56, 0.41)]
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed, (fm, am) in zip(seeds, means):
                self._write_clean_seed(raw_dir, seed, fm, am)
            rc = ab_merge.main(
                ["finetune-run", raw_dir, out_dir, ",".join(str(s) for s in seeds), "", "--allow-missing-lr0-control"]
            )
            self.assertEqual(rc, 0)
            report_path = os.path.join(out_dir, "finetune_run_ab_report.json")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path) as fh:
                merged = json.load(fh)
            self.assertEqual(merged["status"], "GREEN")

    def test_main_finetune_run_dispatch_exits_1_on_invalid(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir, 1, "fused", repeat, _finetune_run_tier(arm="fused", seed=1, admission_is_dense=True)
                )
                _write_finetune_run_leg(raw_dir, 1, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=1))
            rc = ab_merge.main(["finetune-run", raw_dir, out_dir, "1"])
            self.assertEqual(rc, 1)

    def test_empty_raw_dir_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            rc = ab_merge.main(["finetune-run", raw_dir, out_dir, "1,2,3"])
        self.assertEqual(rc, 1)


class BuildFinetuneRunReportDispatchProofEndToEndTests(unittest.TestCase):
    """`finetune_run_dispatch_proof_violations` wired into
    `build_finetune_run_report` (unit-63 audit finding 2's merger half) —
    driven through the REAL merge entry point, never the bare function
    alone.
    """

    def test_fused_leg_failing_dispatch_proof_invalidates_its_seed(self):
        seeds = [1, 2, 3]
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_overrides = {"ln_fused_dispatches": 0} if seed == 2 else {}
                for repeat in ("r1", "r2"):
                    _write_finetune_run_leg(
                        raw_dir, seed, "fused", repeat, _finetune_run_tier(arm="fused", seed=seed, **fused_overrides)
                    )
                    _write_finetune_run_leg(raw_dir, seed, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=seed))
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(
            any("fused-dispatch proof" in v for v in merged["per_seed"]["2"]["leg_premise_violations"]),
            merged["per_seed"]["2"]["leg_premise_violations"],
        )
        # seeds 1/3 stay dispatch-proof clean.
        self.assertEqual(merged["per_seed"]["1"]["leg_premise_violations"], [])
        self.assertEqual(merged["per_seed"]["3"]["leg_premise_violations"], [])

    def test_fused_leg_with_flash_compiled_false_invalidates_its_seed(self):
        # Unit-63 round-3 audit, coordinator correction, end-to-end: a
        # `fused` leg built without flash-attn compiled in is an INVALID
        # premise the moment it reaches the real merge entry point, never
        # merely a `fused_proof` warning buried in a table column.
        seeds = [1, 2]
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_overrides = {"flash_compiled": False} if seed == 1 else {}
                for repeat in ("r1", "r2"):
                    _write_finetune_run_leg(
                        raw_dir, seed, "fused", repeat, _finetune_run_tier(arm="fused", seed=seed, **fused_overrides)
                    )
                    _write_finetune_run_leg(raw_dir, seed, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=seed))
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(
            any("flash_compiled=False" in v for v in merged["per_seed"]["1"]["leg_premise_violations"]),
            merged["per_seed"]["1"]["leg_premise_violations"],
        )
        self.assertEqual(merged["per_seed"]["2"]["leg_premise_violations"], [])

    def test_alloff_leg_with_attention_block_fused_zero_invalidates_its_seed(self):
        # unit-63 round-3 audit block 4, coordinator correction:
        # `attention_block` reading `fused == 0` on an alloff leg (the
        # disabled flash cascade failing to fall through to a live fused
        # kernel) is caught by the positive training-path proof -- never
        # the pre-fix blanket "every pair must be fused == 0" rule, which
        # would have also flagged this golden's OWN real ln/rope/softmax/
        # geglu/lora_linear fused counts as violations too.
        seeds = [1, 2]
        with tempfile.TemporaryDirectory() as raw_dir:
            alloff_overrides = {"attention_block_fused_dispatches": 0, "attention_block_eager_dispatches": 4}
            for seed in seeds:
                overrides = alloff_overrides if seed == 1 else {}
                for repeat in ("r1", "r2"):
                    _write_finetune_run_leg(raw_dir, seed, "fused", repeat, _finetune_run_tier(arm="fused", seed=seed))
                    _write_finetune_run_leg(
                        raw_dir, seed, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=seed, **overrides)
                    )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(
            any(
                "attention_block_fused_dispatches=0" in v
                for v in merged["per_seed"]["1"]["leg_premise_violations"]
            ),
            merged["per_seed"]["1"]["leg_premise_violations"],
        )
        self.assertEqual(merged["per_seed"]["2"]["leg_premise_violations"], [])


class BuildFinetuneRunReportCrossSeedHomogeneityEndToEndTests(unittest.TestCase):
    """`finetune_run_cross_seed_homogeneity_violations` wired into
    `build_finetune_run_report` (unit-63 audit finding 3) — driven through
    the REAL merge entry point.
    """

    _MEANS = {
        1: (0.30, 0.50),
        2: (0.32, 0.48),
        3: (0.29, 0.55),
        4: (0.31, 0.47),
        5: (0.28, 0.52),
        6: (0.33, 0.49),
        7: (0.27, 0.53),
        8: (0.55, 0.40),
        9: (0.52, 0.38),
        10: (0.58, 0.42),
        11: (0.50, 0.35),
        12: (0.54, 0.39),
    }

    def test_two_fixture_split_end_to_end_is_invalid(self):
        # Empirical reproduction this finding fixes: 6 seeds against
        # `heldout_pairs_sha256="fixture-a"`, 6 against `"fixture-b"` --
        # each seed's own fused/alloff pair internally agrees, and the
        # 7-vs-5 sign-test shape used to read GREEN (see
        # `test_end_to_end_green_matches_direct_sign_test`, the SAME means).
        # This must now be INVALID, naming the diverging field.
        seeds = list(range(1, 13))
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fixture = "fixture-a" if seed <= 6 else "fixture-b"
                fused_mean, alloff_mean = self._MEANS[seed]
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir,
                            seed,
                            arm,
                            repeat,
                            _finetune_run_tier(
                                arm=arm, seed=seed, held_out_example_mean=mean, heldout_pairs_sha256=fixture
                            ),
                        )
            merged, table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(
            any("heldout_pairs_sha256" in v for v in merged["cross_seed_identity_violations"]),
            merged["cross_seed_identity_violations"],
        )
        self.assertIn("cross_seed_identity_violations", table)

    def test_homogeneous_twelve_end_to_end_is_unaffected(self):
        # Same fixture as `test_end_to_end_green_matches_direct_sign_test` --
        # a genuinely homogeneous 12-seed sweep must stay exactly as before
        # this fix.
        seeds = list(range(1, 13))
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_mean, alloff_mean = self._MEANS[seed]
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "GREEN")
        self.assertEqual(merged["cross_seed_identity_violations"], [])

    def test_single_divergent_seed_end_to_end_is_invalid_naming_the_field(self):
        seeds = list(range(1, 13))
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_mean, alloff_mean = self._MEANS[seed]
                overrides = {"checkpoint_weights_sha256": "different-checkpoint"} if seed == 5 else {}
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir,
                            seed,
                            arm,
                            repeat,
                            _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean, **overrides),
                        )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertEqual(len(merged["cross_seed_identity_violations"]), 1)
        self.assertIn("checkpoint_weights_sha256", merged["cross_seed_identity_violations"][0])
        self.assertIn("seed 5", merged["cross_seed_identity_violations"][0])

    def _write_lr0_control_seed(self, raw_dir, seed):
        # `--lr 0` BY CONSTRUCTION (block 3's own premise) -- every OTHER
        # identity field left at `_finetune_run_tier`'s own default, so it
        # matches the main A/B seeds on everything except `lr`/`seed`.
        for arm in ("fused", "alloff"):
            _write_finetune_run_leg(
                raw_dir,
                seed,
                arm,
                ab_merge.FINETUNE_RUN_LR0_REPEAT,
                _finetune_run_tier(arm=arm, seed=seed, lr=0.0, train_probe_series=[0.0, 0.0]),
            )

    def test_lr0_seeds_1_and_2_end_to_end_is_not_invalid(self):
        # unit-63 round-3 audit block 3's own end-to-end pin: 12 premise-
        # clean, cross-seed-homogeneous main seeds PLUS a real lr0 control
        # (seeds 1, 2) -- before block 3's fix, the lr0 legs' own `lr=0`
        # (vs. the main seeds' real, nonzero `lr`) would have unconditionally
        # collapsed `cross_seed_identity_violations`, making ANY nonempty
        # `lr0_seeds` list INVALID no matter how clean everything else was.
        seeds = list(range(1, 13))
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_mean, alloff_mean = self._MEANS[seed]
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
            self._write_lr0_control_seed(raw_dir, 1)
            self._write_lr0_control_seed(raw_dir, 2)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, lr0_seeds=[1, 2])
        self.assertNotEqual(merged["status"], "INVALID", merged)
        self.assertEqual(merged["cross_seed_identity_violations"], [])
        self.assertEqual(merged["lr0_control"]["violations"], [])

    def test_lr0_seed_diverging_on_a_non_lr_field_end_to_end_is_invalid(self):
        # The mutant: an lr0-control leg diverging on a field OTHER than
        # `lr` (its own defining premise) must still collapse the merge --
        # the block 3 exception is narrow, never a blanket exemption for
        # lr0-control legs.
        seeds = list(range(1, 13))
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in seeds:
                fused_mean, alloff_mean = self._MEANS[seed]
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
            self._write_lr0_control_seed(raw_dir, 1)
            # seed 2's lr0-control legs diverge on `schedule` -- NOT `lr`.
            for arm in ("fused", "alloff"):
                _write_finetune_run_leg(
                    raw_dir,
                    2,
                    arm,
                    ab_merge.FINETUNE_RUN_LR0_REPEAT,
                    _finetune_run_tier(arm=arm, seed=2, lr=0.0, train_probe_series=[0.0, 0.0], schedule="cosine"),
                )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, seeds, lr0_seeds=[1, 2])
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(
            any("schedule" in v for v in merged["cross_seed_identity_violations"]),
            merged["cross_seed_identity_violations"],
        )


class FinetuneRunDecisionRuleMutantTests(unittest.TestCase):
    """The pre-registered decision rule itself (unit-63 audit finding 1):
    `build_finetune_run_report` used to compute the sign test and then
    hardcode `status = "GREEN"` regardless of what it found -- EVERY mutant
    below would have read GREEN under that pre-fix code (any premise-clean,
    non-tied, non-empty `d_values` produced a `sign_result`, and a
    `sign_result` alone was sufficient for GREEN); the fixed rule reads each
    one correctly instead. One mutant per arm of `FINETUNE_RUN_DECISION_RULE_TEXT`'s
    own predicate -- `n_pos`/`n_neg` >= `FINETUNE_RUN_DECISION_THRESHOLD` (11
    of `FINETUNE_RUN_GATE_SEED_COUNT`, 12) AND the mean's sign agreeing with
    that concordant direction, plus the `clean_seed_count != 12` -> INVALID
    carve-out -- proves none of the rule's own conjuncts is vacuous.
    """

    def _write_r1(self, raw_dir, seed, fused_mean, alloff_mean):
        # r1-only (no r2): the decision rule only ever reads r1 -- see
        # `build_finetune_run_report`'s own doc -- and omitting r2 keeps
        # every mutant here from ever touching the (separately tested)
        # determinism floor.
        for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
            _write_finetune_run_leg(
                raw_dir, seed, arm, "r1", _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
            )

    def test_11_of_12_degradation_is_red(self):
        # 11 seeds with fused WORSE than alloff (d_i > 0, degradation-
        # concordant), 1 dissenting seed -- exactly the golden (12, 11)
        # sign-test shape, mean necessarily > 0 since the dissent is the
        # SAME magnitude as the majority. Under the pre-fix code this read
        # GREEN (a sign_result existed); the fixed rule reads RED.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 12):
                self._write_r1(raw_dir, seed, 0.60, 0.50)  # fused worse: d = +0.10
            self._write_r1(raw_dir, 12, 0.50, 0.60)  # dissent: d = -0.10
            merged, table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["decision"]["n_pos"], 11)
        self.assertEqual(merged["decision"]["n_neg"], 1)
        self.assertGreater(merged["decision"]["mean_d"], 0.0)
        self.assertEqual(merged["decision"]["concordant_direction"], "degradation")
        self.assertEqual(merged["status"], "RED")
        self.assertIn("status: RED", table)

    def test_11_of_12_improvement_is_red_for_investigation(self):
        # Mirror image: fused BEATS alloff on 11 of 12 (d_i < 0,
        # improvement-concordant) -- anomalous improvement is investigated,
        # never silently celebrated as GREEN.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 12):
                self._write_r1(raw_dir, seed, 0.50, 0.60)  # fused better: d = -0.10
            self._write_r1(raw_dir, 12, 0.60, 0.50)  # dissent: d = +0.10
            merged, table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["decision"]["n_pos"], 1)
        self.assertEqual(merged["decision"]["n_neg"], 11)
        self.assertLess(merged["decision"]["mean_d"], 0.0)
        self.assertEqual(merged["decision"]["concordant_direction"], "improvement")
        self.assertEqual(merged["status"], "RED_FOR_INVESTIGATION")
        self.assertIn("status: RED_FOR_INVESTIGATION", table)

    def test_10_of_12_is_green(self):
        # Below the 11-of-12 threshold on EITHER side -- the rule is
        # pre-registered for >=11, never a bare majority.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 11):
                self._write_r1(raw_dir, seed, 0.60, 0.50)  # fused worse: d = +0.10
            for seed in (11, 12):
                self._write_r1(raw_dir, seed, 0.50, 0.60)  # dissent: d = -0.10
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["decision"]["n_pos"], 10)
        self.assertEqual(merged["decision"]["n_neg"], 2)
        self.assertEqual(merged["decision"]["concordant_direction"], "none")
        self.assertEqual(merged["status"], "GREEN")

    def test_mean_sign_disagreement_at_11_is_green(self):
        # The AND bites: n_pos=11 (degradation-concordant by COUNT) but one
        # huge dissenting seed pulls mean(d_i) negative -- the count
        # threshold alone is NOT sufficient; the mean's sign must also
        # agree, or the rule falls through to GREEN rather than RED.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 12):
                self._write_r1(raw_dir, seed, 0.51, 0.50)  # small positive: d = +0.01 each
            self._write_r1(raw_dir, 12, 0.10, 1.10)  # huge dissent: d = -1.00
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["decision"]["n_pos"], 11)
        self.assertEqual(merged["decision"]["n_neg"], 1)
        # 11 * 0.01 - 1.00 = -0.89 < 0 -- disagrees with the "degradation"
        # direction n_pos=11 alone would otherwise indicate.
        self.assertLess(merged["decision"]["mean_d"], 0.0)
        self.assertEqual(merged["decision"]["concordant_direction"], "none")
        self.assertEqual(merged["status"], "GREEN")

    def test_11_seeds_is_invalid_never_rescaled(self):
        # Only 11 seeds were ever passed to the merger (a real short sweep,
        # e.g. one seed simply never dispatched) -- ALL 11 are premise-clean
        # and unanimous in sign, exactly the shape that would have read
        # GREEN (indeed, the pre-fix code's own hardcoded GREEN) at n=11.
        # The rule is pre-registered FOR 12; a different count is INVALID,
        # never silently rescaled to fit whatever n happened to show up.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 12):
                self._write_r1(raw_dir, seed, 0.60, 0.50)  # unanimous: d = +0.10
            merged, table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 12)), allow_missing_lr0_control=True)
        self.assertEqual(merged["decision"]["clean_seed_count"], 11)
        self.assertEqual(merged["wrong_seed_count"], True)
        self.assertEqual(merged["status"], "INVALID")
        self.assertIn("status: INVALID", table)


class FinetuneRunLr0ControlTests(unittest.TestCase):
    """The lr=0 RED control (unit-63 audit advisory (b)):
    `finetune_run_lr0_control_seed_violations` / its wiring into
    `build_finetune_run_report`'s own `lr0_control` section -- a clean
    control leg FAILS learning-happened and is never counted into the A/B
    set; a control leg that PASSES (or never ran) is a violation that
    collapses `status` to `INVALID`.
    """

    def _write_ab_seeds(self, raw_dir, n=12):
        # A clean, GREEN-shaped 12-seed A/B set (6-vs-6, under the decision
        # threshold) so any INVALID seen in these tests is attributable to
        # the lr0 control alone, never the A/B set itself.
        for seed in range(1, n // 2 + 1):
            for arm, mean in (("fused", 0.30), ("alloff", 0.50)):
                _write_finetune_run_leg(
                    raw_dir, seed, arm, "r1", _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                )
        for seed in range(n // 2 + 1, n + 1):
            for arm, mean in (("fused", 0.55), ("alloff", 0.40)):
                _write_finetune_run_leg(
                    raw_dir, seed, arm, "r1", _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                )

    def _write_lr0_leg(self, raw_dir, seed, arm, learning_happened_delta, **overrides):
        # unit-63 round-3 audit block 3: real lr0-control legs run at
        # `--lr 0` BY CONSTRUCTION (`finetune_run_ab.sh`'s own lr=0 loop
        # passes `"0"` explicitly) -- setting it here (rather than leaving
        # it at `_finetune_run_tier`'s own main-A/B default) is what
        # actually exercises `finetune_run_cross_seed_homogeneity_violations`'s
        # own `lr` exception (block 3's fix); every test in this class
        # previously left `lr` at the SAME default the main A/B seeds use,
        # which never genuinely exercised the divergence this control's own
        # premise creates.
        #
        # amendment 2026-08-29b: `learning_happened_delta` (the caller's own
        # desired DERIVED value, for readability at every call site below)
        # is realized as a 2-entry `train_probe_series` -- `[delta, 0.0]` --
        # so the merger's own `series[0] - series[-1]` derivation reproduces
        # exactly the delta this test asks for, unless the caller already
        # supplies its own `train_probe_series` override.
        overrides.setdefault("lr", 0.0)
        overrides.setdefault("train_probe_series", [learning_happened_delta, 0.0])
        _write_finetune_run_leg(
            raw_dir,
            seed,
            arm,
            ab_merge.FINETUNE_RUN_LR0_REPEAT,
            _finetune_run_tier(arm=arm, seed=seed, **overrides),
        )

    # unit-63 round-4 audit F-2: `finetune_run_lr0_control_seed_violations`'s
    # own new positive fact -- an OK lr0-control leg's reported `lr` must
    # equal `0.0` EXACTLY. Exercised directly (never only end-to-end) so a
    # divergence from `learning_happened_delta`'s own, independent check is
    # unambiguous.

    def test_control_leg_at_lr_0_001_is_a_violation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0, lr=0.001)
            self._write_lr0_leg(raw_dir, 101, "alloff", 0.0)
            violations, per_arm, _identities = ab_merge.finetune_run_lr0_control_seed_violations(raw_dir, 101)
        self.assertTrue(
            any("reported lr=0.001" in v and "not exactly 0.0" in v for v in violations), violations
        )
        self.assertEqual(per_arm["fused"]["lr"], 0.001)

    def test_control_leg_at_lr_0_777_with_clean_delta_is_still_a_violation(self):
        # The `lr` fact and the `learning_happened_delta` fact are
        # INDEPENDENT -- a control leg can pass the (unrelated)
        # learning-happened calibration check while still failing the
        # 'did this leg actually run at lr=0' fact this round adds.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0, lr=0.777)
            self._write_lr0_leg(raw_dir, 101, "alloff", 0.0)
            violations, per_arm, _identities = ab_merge.finetune_run_lr0_control_seed_violations(raw_dir, 101)
        self.assertTrue(
            any("reported lr=0.777" in v and "not exactly 0.0" in v for v in violations), violations
        )
        # The learning-happened check itself stays clean for this leg --
        # proving the two checks are independent, not one masking the other.
        self.assertFalse(any("unexpectedly CLEARS the floor" in v for v in violations), violations)
        self.assertEqual(per_arm["fused"]["learning_happened_delta"], 0.0)

    def test_control_leg_at_lr_0_0_is_clean_on_the_lr_fact(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0)
            self._write_lr0_leg(raw_dir, 101, "alloff", 0.0)
            violations, per_arm, _identities = ab_merge.finetune_run_lr0_control_seed_violations(raw_dir, 101)
        self.assertEqual(violations, [])
        self.assertEqual(per_arm["fused"]["lr"], 0.0)
        self.assertEqual(per_arm["alloff"]["lr"], 0.0)

    def test_control_leg_diverging_on_lr_end_to_end_invalidates(self):
        # The end-to-end path (`build_finetune_run_report`, the REAL merge
        # stage `finetune_run_ab.sh` drives) -- a control leg's own lr
        # divergence must collapse `status` to INVALID exactly like every
        # other lr0_control violation does, never silently absorbed.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0, lr=0.001)
            self._write_lr0_leg(raw_dir, 101, "alloff", 0.0)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), lr0_seeds=[101])
        self.assertTrue(
            any("not exactly 0.0" in v for v in merged["lr0_control"]["violations"]),
            merged["lr0_control"]["violations"],
        )
        self.assertEqual(merged["status"], "INVALID")

    def test_clean_lr0_control_fails_learning_happened_and_is_green(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            for seed in (101, 102):
                for arm in ("fused", "alloff"):
                    self._write_lr0_leg(raw_dir, seed, arm, 0.0)  # no learning under lr=0 -- clean
            merged, table = ab_merge.build_finetune_run_report(
                raw_dir, list(range(1, 13)), lr0_seeds=[101, 102]
            )
        self.assertEqual(merged["lr0_control"]["violations"], [])
        self.assertEqual(merged["status"], "GREEN")
        self.assertIn("lr0_control: seeds=", table)
        # NEVER counted into the A/B set.
        self.assertNotIn("101", merged["d_values"])
        self.assertNotIn("102", merged["d_values"])

    def test_passing_lr0_control_leg_is_a_violation_and_invalidates(self):
        # A "passing" control (learning_happened_delta clears the floor
        # despite lr=0) is a finding against the FLOOR=0.0 premise-leg
        # ruling itself -- never silently passed through.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0)
            self._write_lr0_leg(raw_dir, 101, "alloff", 0.05)  # unexpectedly clears the floor
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), lr0_seeds=[101])
        self.assertTrue(
            any("unexpectedly CLEARS the floor" in v for v in merged["lr0_control"]["violations"]),
            merged["lr0_control"]["violations"],
        )
        self.assertEqual(merged["status"], "INVALID")

    def test_missing_lr0_control_leg_is_a_violation_and_invalidates(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            self._write_lr0_leg(raw_dir, 101, "fused", 0.0)
            # alloff's own lr0 leg for seed 101 never written -- MISSING.
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), lr0_seeds=[101])
        self.assertTrue(
            any("not OK" in v for v in merged["lr0_control"]["violations"]), merged["lr0_control"]["violations"]
        )
        self.assertEqual(merged["status"], "INVALID")

    def test_no_lr0_seeds_with_allow_flag_is_a_deliberate_no_op(self):
        # unit-63 round-3 audit block 5: an empty lr0_seeds list is a
        # DELIBERATE, visible opt-out only when `allow_missing_lr0_control`
        # is explicitly passed -- see the sibling refusal test below for the
        # DEFAULT (no flag) behavior.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            merged, table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["lr0_control"]["seeds"], [])
        self.assertEqual(merged["lr0_control"]["violations"], [])
        self.assertIs(merged["lr0_control"]["allow_missing_lr0_control"], True)
        self.assertEqual(merged["status"], "GREEN")
        self.assertNotIn("lr0_control: seeds=", table)

    def test_no_lr0_seeds_without_allow_flag_is_invalid(self):
        # unit-63 round-3 audit block 5: CONTRACT Frame's own RED control is
        # pre-registered, not optional -- the DEFAULT (no
        # `allow_missing_lr0_control`) refuses rather than silently skipping
        # it, the class fix for `gpu-howwell.yml`'s own `|| ''` collapse
        # (see that workflow's own "Resolve" step).
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_ab_seeds(raw_dir)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)))
        self.assertIs(merged["lr0_control"]["allow_missing_lr0_control"], False)
        self.assertTrue(
            any("allow_missing_lr0_control was not set" in v for v in merged["lr0_control"]["violations"]),
            merged["lr0_control"]["violations"],
        )
        self.assertEqual(merged["status"], "INVALID")

    def test_main_finetune_run_dispatch_honours_allow_missing_lr0_control_flag(self):
        # The REAL entry point `finetune_run_ab.sh` calls -- proves the CLI
        # flag itself (never just the Python-level kwarg) round-trips into
        # `build_finetune_run_report` and the merged artifact.
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            self._write_ab_seeds(raw_dir)
            seeds_s = ",".join(str(s) for s in range(1, 13))
            rc = ab_merge.main(["finetune-run", raw_dir, out_dir, seeds_s, "", "--allow-missing-lr0-control"])
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 0)
        self.assertEqual(merged["status"], "GREEN")
        self.assertIs(merged["lr0_control"]["allow_missing_lr0_control"], True)

    def test_main_finetune_run_dispatch_without_flag_refuses(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            self._write_ab_seeds(raw_dir)
            seeds_s = ",".join(str(s) for s in range(1, 13))
            rc = ab_merge.main(["finetune-run", raw_dir, out_dir, seeds_s])
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 1)
        self.assertEqual(merged["status"], "INVALID")
        self.assertIs(merged["lr0_control"]["allow_missing_lr0_control"], False)


class PremiseFailureDiagnosticTests(unittest.TestCase):
    """`premise_failure_diagnostic` (amendment 2026-08-29b item 1(c)):
    ALWAYS present in the merged artifact, non-parameterised, and never
    itself decisional -- it can only ever RECORD which premise leg(s) failed
    on which leg, with that leg's raw `train_probe_series`, never promote an
    INVALID verdict to GREEN.
    """

    def _write_clean_seed(self, raw_dir, seed, fused_mean=0.30, alloff_mean=0.50):
        for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                )

    def test_diagnostic_key_always_present_and_empty_on_a_clean_run(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_clean_seed(raw_dir, 1)
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1], allow_missing_lr0_control=True)
        self.assertIn("premise_failure_diagnostic", merged)
        self.assertEqual(merged["premise_failure_diagnostic"]["failed_seeds"], [])
        self.assertEqual(merged["premise_failure_diagnostic"]["failing_legs"], [])

    def test_campaign_v1_shaped_floor_fail_is_recorded_with_its_raw_series(self):
        # Mirrors the committed campaign-v1 evidence's own root cause
        # (docs/plans/63-how-well/measurements/campaign-v1/README.md): one
        # seed's alloff leg fails the learning-happened premise while every
        # other seed stays clean -- the diagnostic must name exactly that
        # leg, exactly that premise, and carry its raw series.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_clean_seed(raw_dir, 1)
            self._write_clean_seed(raw_dir, 2)
            # seed 3's alloff leg: the series goes UP (spiked, never
            # recovered net) -- a real floor-fail shape, not a contrived one.
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir,
                    3,
                    "alloff",
                    repeat,
                    _finetune_run_tier(arm="alloff", seed=3, train_probe_series=[3.2276, 3.4211, 3.2447]),
                )
                _write_finetune_run_leg(raw_dir, 3, "fused", repeat, _finetune_run_tier(arm="fused", seed=3))
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1, 2, 3], allow_missing_lr0_control=True)

        self.assertEqual(merged["status"], "INVALID")
        diag = merged["premise_failure_diagnostic"]
        self.assertEqual(diag["failed_seeds"], [3])
        self.assertEqual(len(diag["failing_legs"]), 1)
        entry = diag["failing_legs"][0]
        self.assertEqual(entry["label"], "seed 3 alloff r1")
        self.assertEqual(entry["pool"], "main")
        self.assertEqual(entry["failing_premises"], ["learning_happened"])
        self.assertEqual(entry["train_probe_series"], [3.2276, 3.4211, 3.2447])
        # never decisional -- the diagnostic's own presence changes nothing
        # about the (independently computed) status above.
        self.assertIn("NEVER promote", diag["note"])
        self.assertIn("NO operator override", diag["note"])

    def test_v1_scalar_only_seed_is_invalid_and_recorded_in_diagnostic(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_clean_seed(raw_dir, 1)
            for repeat in ("r1", "r2"):
                _write_finetune_run_leg(
                    raw_dir,
                    2,
                    "fused",
                    repeat,
                    _finetune_run_tier(arm="fused", seed=2, learning_happened_delta=0.05, train_probe_series=None),
                )
                _write_finetune_run_leg(raw_dir, 2, "alloff", repeat, _finetune_run_tier(arm="alloff", seed=2))
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1, 2], allow_missing_lr0_control=True)

        self.assertEqual(merged["status"], "INVALID")
        diag = merged["premise_failure_diagnostic"]
        self.assertEqual(diag["failed_seeds"], [2])
        entry = diag["failing_legs"][0]
        self.assertEqual(entry["label"], "seed 2 fused r1")
        self.assertEqual(entry["failing_premises"], ["learning_happened"])
        self.assertIsNone(entry["train_probe_series"])
        self.assertTrue(
            any("producer-version mismatch" in v for v in merged["per_seed"]["2"]["leg_premise_violations"])
        )

    def test_lr0_control_learning_happened_failure_is_recorded_as_its_own_pool(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_clean_seed(raw_dir, 1)
            for arm, series in (("fused", [0.0, 0.0]), ("alloff", [0.05, 0.0])):
                _write_finetune_run_leg(
                    raw_dir,
                    101,
                    arm,
                    ab_merge.FINETUNE_RUN_LR0_REPEAT,
                    _finetune_run_tier(arm=arm, seed=101, lr=0.0, train_probe_series=series),
                )
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, [1], lr0_seeds=[101])

        self.assertEqual(merged["status"], "INVALID")
        diag = merged["premise_failure_diagnostic"]
        self.assertIn(101, diag["failed_seeds"])
        entry = next(e for e in diag["failing_legs"] if e["pool"] == "lr0-control")
        self.assertEqual(entry["label"], "lr0 seed 101 alloff")
        self.assertEqual(entry["failing_premises"], ["learning_happened"])
        self.assertEqual(entry["train_probe_series"], [0.05, 0.0])


def _mutant_tier(arm="fused", **overrides):
    """A mutant leg's own tier -- a normal, premise-clean `fused` leg
    (`_finetune_run_tier`) plus the three producer-stamped fields
    mutants/README.md's own on-pod procedure records per leg (unit-63
    round-7 audit finding 1: `mutant_id`/`mutant_base_sha`/
    `mutant_patch_sha256`, the `FinetuneRunTier` field names the
    `--mutant-id`/`--mutant-base-sha`/`--mutant-patch-sha256` CLI flags
    stamp -- renamed from this suite's own earlier `base_sha`/`patch_sha256`
    names to match the producer's real field names)."""
    tier = _finetune_run_tier(arm=arm)
    tier.update({
        "mutant_id": "M2",
        "mutant_base_sha": "4257cde6d51184475b3e798f5d7e9c3885a763ca",
        "mutant_patch_sha256": "eps0-02-patch-sha",
    })
    tier.update(overrides)
    return tier


def _write_mutant_leg(raw_dir, seed, dose_label, tier):
    _write_finetune_run_leg(raw_dir, seed, "fused", ab_merge.mutant_leg_repeat_tag(dose_label), tier)


class MutantDoseLadderTests(unittest.TestCase):
    """`build_mutant_dose_column`/`mutant_dose_ladder_sensitivity` (amendment
    2026-08-29b item 3): each dose column merges the mutant-substituted
    fused arm against the SAME campaign alloff legs under the SAME
    `>=11/12`+mean-sign rule the primary decision uses; mutant legs never
    enter the primary A/B set.
    """

    PATCH_SHA = "eps0-02-patch-sha"

    def _write_alloff(self, raw_dir, seed, mean=0.50):
        _write_finetune_run_leg(
            raw_dir, seed, "alloff", "r1", _finetune_run_tier(arm="alloff", seed=seed, held_out_example_mean=mean)
        )

    def test_repeat_tag_never_collides_with_r1_r2_or_lr0(self):
        for label in ("eps0.02", "", "r1", "r2", "lr0"):
            tag = ab_merge.mutant_leg_repeat_tag(label)
            self.assertNotIn(tag, ab_merge.FINETUNE_RUN_REPEATS)
            self.assertNotEqual(tag, ab_merge.FINETUNE_RUN_LR0_REPEAT)

    def test_detected_red_when_threshold_and_direction_match(self):
        # 11 of 12 mutant legs read WORSE (higher held-out loss) than their
        # SAME-SEED alloff leg -- the pre-registered degradation-concordant
        # shape.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 13):
                self._write_alloff(raw_dir, seed, mean=0.50)
                mutant_mean = 0.70 if seed != 12 else 0.30  # seed 12 dissents
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=mutant_mean))
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, list(range(1, 13)))
        self.assertEqual(col["detected"], "RED")
        self.assertEqual(col["n_pos"], 11)
        self.assertEqual(col["n_neg"], 1)
        self.assertGreater(col["mean_d"], 0.0)
        self.assertEqual(col["clean_pair_count"], 12)
        self.assertEqual(col["violations"], [])

    def test_detected_red_for_investigation_when_threshold_and_direction_are_improvement(self):
        # unit-63 round-8 audit finding 2: 11 of 12 mutant legs read BETTER
        # (lower held-out loss) than their SAME-SEED alloff leg -- the
        # improvement-concordant shape the two-sided-falsification cell
        # (+0.50) needs a real, reportable state for. Before this fix, this
        # exact shape collapsed into "not-detected" and the confirming
        # outcome could never be reported.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 13):
                self._write_alloff(raw_dir, seed, mean=0.50)
                mutant_mean = 0.30 if seed != 12 else 0.70  # seed 12 dissents
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=mutant_mean))
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, list(range(1, 13)))
        self.assertEqual(col["detected"], "RED_FOR_INVESTIGATION")
        self.assertEqual(col["n_pos"], 1)
        self.assertEqual(col["n_neg"], 11)
        self.assertLess(col["mean_d"], 0.0)
        self.assertEqual(col["clean_pair_count"], 12)
        self.assertEqual(col["violations"], [])

    def test_sign_flipping_transient_is_not_detected(self):
        # mutants/README.md's own M1 finding, reproduced generically: an
        # 8/12 split (well under the 11/12 threshold) reads not-detected,
        # never RED -- "detects movement" is explicitly not this gate's claim.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 13):
                self._write_alloff(raw_dir, seed, mean=0.50)
                mutant_mean = 0.55 if seed <= 8 else 0.45
                _write_mutant_leg(raw_dir, seed, "m1-shaped", _mutant_tier(seed=seed, held_out_example_mean=mutant_mean))
            col = ab_merge.build_mutant_dose_column(raw_dir, "m1-shaped", self.PATCH_SHA, list(range(1, 13)))
        self.assertEqual(col["detected"], "not-detected")
        self.assertEqual(col["n_pos"], 8)
        self.assertEqual(col["n_neg"], 4)

    def test_wrong_clean_pair_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 12):  # only 11, not the pre-registered 12
                self._write_alloff(raw_dir, seed, mean=0.50)
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, list(range(1, 13)))
        self.assertEqual(col["detected"], "INVALID")
        self.assertEqual(col["clean_pair_count"], 11)
        self.assertTrue(any("expected exactly" in v for v in col["violations"]), col["violations"])

    def test_missing_provenance_field_is_a_violation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_alloff(raw_dir, 1, mean=0.50)
            _write_mutant_leg(
                raw_dir, 1, "eps0.50", _mutant_tier(seed=1, held_out_example_mean=0.70, mutant_id=None)
            )
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, [1])
        self.assertTrue(any("mutant_id" in v for v in col["violations"]), col["violations"])
        self.assertIsNone(col["per_seed"]["1"]["d_i"])

    def test_whitespace_only_provenance_fields_are_treated_as_empty(self):
        # unit-63 round-8 audit finding 4 (merger half): a whitespace-only
        # value (" ") for any of the three producer-stamped fields is
        # exactly as absent as "" or None -- the pre-fix bare
        # `if not tier.get(field)` check passed it straight through as
        # though it were present.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_alloff(raw_dir, 1, mean=0.50)
            _write_mutant_leg(
                raw_dir,
                1,
                "eps0.50",
                _mutant_tier(
                    seed=1,
                    held_out_example_mean=0.70,
                    mutant_id=" ",
                    mutant_base_sha="\t",
                    mutant_patch_sha256="  ",
                ),
            )
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, [1])
        for field in ("mutant_id", "mutant_base_sha", "mutant_patch_sha256"):
            self.assertTrue(
                any(f"{field!r}" in v and "missing/empty" in v for v in col["violations"]),
                col["violations"],
            )
        self.assertIsNone(col["per_seed"]["1"]["d_i"])

    def test_sha_comparison_uses_stripped_values(self):
        # unit-63 round-8 audit finding 4 (merger half): incidental
        # leading/trailing whitespace on either side of the sha comparison
        # must never be reported as a labeling-error mismatch.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_alloff(raw_dir, 1, mean=0.50)
            _write_mutant_leg(
                raw_dir,
                1,
                "eps0.50",
                _mutant_tier(seed=1, held_out_example_mean=0.70, mutant_patch_sha256=f"  {self.PATCH_SHA}\t"),
            )
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", f" {self.PATCH_SHA} ", [1])
        # per-leg violations only -- the column-level "expected exactly 12"
        # count violation (this test only runs 1 seed) is orthogonal to the
        # sha-comparison claim under test here.
        self.assertEqual(col["per_seed"]["1"]["violations"], [])
        self.assertIsNotNone(col["per_seed"]["1"]["d_i"])

    def test_sha_comparison_is_case_insensitive_in_every_case_combination(self):
        # unit-63 round-10 audit F2: sha hex is case-insensitive by domain.
        # The producer now canonicalizes its own stamped mutant_patch_sha256
        # to lowercase (round-9 advisory (b)) -- an upper/upper pair that
        # matched before that change must still match, and every other
        # (leg-case, caller-case) combination must match too, since the
        # comparison itself now folds case on both sides.
        base_sha = self.PATCH_SHA  # already all-lowercase
        case_cells = [
            ("lower", "lower", base_sha, base_sha),
            ("lower", "upper", base_sha, base_sha.upper()),
            ("upper", "lower", base_sha.upper(), base_sha),
            ("upper", "upper", base_sha.upper(), base_sha.upper()),
        ]
        for leg_case, caller_case, leg_sha, caller_sha in case_cells:
            with self.subTest(leg_case=leg_case, caller_case=caller_case):
                with tempfile.TemporaryDirectory() as raw_dir:
                    self._write_alloff(raw_dir, 1, mean=0.50)
                    _write_mutant_leg(
                        raw_dir,
                        1,
                        "eps0.50",
                        _mutant_tier(seed=1, held_out_example_mean=0.70, mutant_patch_sha256=leg_sha),
                    )
                    col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", caller_sha, [1])
                self.assertEqual(col["per_seed"]["1"]["violations"], [])
                self.assertIsNotNone(col["per_seed"]["1"]["d_i"])

    def test_patch_sha256_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_alloff(raw_dir, 1, mean=0.50)
            _write_mutant_leg(
                raw_dir,
                1,
                "eps0.50",
                _mutant_tier(seed=1, held_out_example_mean=0.70, mutant_patch_sha256="some-other-sha"),
            )
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, [1])
        self.assertTrue(any("does not match this dose column" in v for v in col["violations"]), col["violations"])
        self.assertIsNone(col["per_seed"]["1"]["d_i"])

    def test_premise_failing_alloff_partner_excludes_the_pair(self):
        # unit-63 round-7 audit finding 2: the main pool premise-checks BOTH
        # arms, but this column used to premise-check only the mutant
        # (fused-shaped) leg -- the REUSED alloff partner never got the same
        # check. A v1-seed-4-shaped alloff partner (train_probe_series
        # giving a negative learning_happened_delta, mirroring the REAL
        # campaign-v1 seed-4 alloff leg's own -0.1125 floor breach --
        # measurements/campaign-v1/README.md) must exclude the PAIR from
        # this dose column, never silently count as a clean partner.
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed in range(1, 13):
                if seed == 4:
                    _write_finetune_run_leg(
                        raw_dir,
                        seed,
                        "alloff",
                        "r1",
                        _finetune_run_tier(
                            arm="alloff", seed=seed, held_out_example_mean=0.50, train_probe_series=[0.5, 0.6]
                        ),
                    )
                else:
                    self._write_alloff(raw_dir, seed, mean=0.50)
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            col = ab_merge.build_mutant_dose_column(raw_dir, "eps0.50", self.PATCH_SHA, list(range(1, 13)))
        self.assertEqual(col["detected"], "INVALID")
        self.assertEqual(col["clean_pair_count"], 11)
        self.assertIsNone(col["per_seed"]["4"]["d_i"])
        self.assertTrue(
            any("learning_happened_delta" in v and "seed 4" in v for v in col["violations"]),
            col["violations"],
        )

    def test_mutant_leg_never_leaks_into_the_ab_set(self):
        # A mutant leg is written under the SAME raw_dir, SAME seeds, as a
        # clean 12-seed main A/B sweep -- the merger's own decision must be
        # bit-for-bit identical to a run with NO mutant legs present at all.
        # 6-vs-6 (mixed sign) keeps the decision GREEN, isolating this test's
        # own claim (no leakage) from the unrelated decision-rule mutants.
        means = {s: (0.30, 0.50) for s in range(1, 7)}
        means.update({s: (0.55, 0.40) for s in range(7, 13)})
        with tempfile.TemporaryDirectory() as raw_dir:
            for seed, (fused_mean, alloff_mean) in means.items():
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                # a mutant leg for this SAME seed, wildly different mean --
                # if it ever leaked into the A/B loader, this would change
                # d_values/the decision.
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=9.99))
            merged, _table = ab_merge.build_finetune_run_report(raw_dir, list(range(1, 13)), allow_missing_lr0_control=True)
        self.assertEqual(merged["status"], "GREEN")
        for seed in range(1, 7):
            self.assertAlmostEqual(merged["d_values"][str(seed)], -0.20)
        for seed in range(7, 13):
            self.assertAlmostEqual(merged["d_values"][str(seed)], 0.15)

    def test_sensitivity_finds_the_straddling_pair_ordered_by_magnitude_not_caller_order(self):
        # unit-63 round-7 audit finding 4 / addendum 2026-08-29c: the
        # SCHEDULED ladder runs ascending eps (-0.50, -0.10, +0.50) -- the
        # LARGER-magnitude degradation dose (-0.50) is passed to this
        # function FIRST, in caller order. A signed ladder with -0.50 RED
        # (only) and -0.10 not-detected must still report the straddle,
        # magnitude-ordered within the negative branch (-0.10 -> -0.50), not
        # `None` (what the pre-fix caller-order scan would have returned:
        # neither adjacent caller-order pair reads (not-detected, RED)).
        neg50 = {"dose_label": "eps-0.50", "detected": "RED"}
        neg10 = {"dose_label": "eps-0.10", "detected": "not-detected"}
        pos50 = {"dose_label": "eps0.50", "detected": "not-detected"}
        sensitivity = ab_merge.mutant_dose_ladder_sensitivity([neg50, neg10, pos50])  # caller/scheduled order
        self.assertEqual(sensitivity, {"lower": "eps-0.10", "higher": "eps-0.50"})

    def test_sensitivity_is_none_when_no_negative_dose_is_detected(self):
        columns = [
            {"dose_label": "eps-0.50", "detected": "not-detected"},
            {"dose_label": "eps-0.10", "detected": "not-detected"},
            {"dose_label": "eps0.50", "detected": "not-detected"},
        ]
        self.assertIsNone(ab_merge.mutant_dose_ladder_sensitivity(columns))

    def test_positive_eps_never_enters_the_negative_branch(self):
        # A single negative dose can never straddle anything by itself; a
        # positive-eps dose reading RED must not be borrowed to complete a
        # pair, regardless of its own detected value.
        columns = [
            {"dose_label": "eps-0.10", "detected": "not-detected"},
            {"dose_label": "eps0.50", "detected": "RED"},
        ]
        self.assertIsNone(ab_merge.mutant_dose_ladder_sensitivity(columns))

    def test_cross_sign_detection_is_a_falsification_finding_not_sensitivity(self):
        # unit-63 round-7 audit finding 4: a POSITIVE-eps dose reading RED
        # (the two-sided falsification cell, addendum 2026-08-29c) must be
        # reported separately, never folded into 'sensitivity' as though a
        # cross-sign (-0.10 not-detected, +0.50 RED) pair were a degradation
        # straddle.
        neg50 = {"dose_label": "eps-0.50", "detected": "not-detected"}
        neg10 = {"dose_label": "eps-0.10", "detected": "not-detected"}
        pos50 = {"dose_label": "eps0.50", "detected": "RED"}
        columns = [neg50, neg10, pos50]
        self.assertIsNone(ab_merge.mutant_dose_ladder_sensitivity(columns))
        falsification = ab_merge.mutant_dose_ladder_two_sided_falsification(columns)
        self.assertEqual(
            falsification,
            [
                {
                    "dose_label": "eps0.50",
                    "eps": 0.50,
                    "detected": "RED",
                    "finding": "secant refuted (degradation at +eps)",
                }
            ],
        )

    def test_positive_eps_red_for_investigation_is_the_confirming_falsification_arm(self):
        # unit-63 round-8 audit finding 1/2: a positive-eps dose reading
        # RED_FOR_INVESTIGATION (improvement-concordant) is the CONFIRMING
        # outcome for the held-out-improvement prediction -- never described
        # as a "refutation" (that word belongs to the RED/degradation arm
        # instead, the exact polarity inversion round-8 finding 1 corrects).
        columns = [{"dose_label": "eps0.50", "detected": "RED_FOR_INVESTIGATION"}]
        falsification = ab_merge.mutant_dose_ladder_two_sided_falsification(columns)
        self.assertEqual(
            falsification,
            [
                {
                    "dose_label": "eps0.50",
                    "eps": 0.50,
                    "detected": "RED_FOR_INVESTIGATION",
                    "finding": "secant confirmed (improvement at +eps)",
                }
            ],
        )

    def test_red_for_investigation_never_enters_the_sensitivity_branch(self):
        # RED_FOR_INVESTIGATION is an improvement-concordant reading; it must
        # never be treated as a degradation-direction detection even for a
        # negative-eps dose_label (a negative dose reading RED_FOR_INVESTIGATION
        # would be an anomalous improvement under DEFLATION, still not the
        # DEGRADATION `sensitivity` statistic is scoped to).
        columns = [
            {"dose_label": "eps-0.50", "detected": "not-detected"},
            {"dose_label": "eps-0.10", "detected": "RED_FOR_INVESTIGATION"},
        ]
        self.assertIsNone(ab_merge.mutant_dose_ladder_sensitivity(columns))

    def test_negative_red_for_investigation_is_a_dose_anomaly(self):
        # unit-63 round-9 audit finding 3: a negative-eps dose reading
        # RED_FOR_INVESTIGATION is an anomalous improvement under
        # deflation -- it must never silently vanish from sensitivity AND
        # anomalies both; sensitivity correctly returns None (unchanged
        # test above), and this is the entry that names the anomaly.
        columns = [
            {"dose_label": "eps-0.50", "detected": "not-detected"},
            {"dose_label": "eps-0.10", "detected": "RED_FOR_INVESTIGATION"},
        ]
        anomalies = ab_merge.mutant_dose_ladder_anomalies(columns)
        self.assertEqual(
            anomalies,
            [
                {
                    "dose_label": "eps-0.10",
                    "eps": -0.10,
                    "detected": "RED_FOR_INVESTIGATION",
                    "finding": "anomalous improvement under deflation (eps < 0)",
                }
            ],
        )

    def test_positive_red_for_investigation_is_never_a_dose_anomaly(self):
        # The ORDINARY, predicted two-sided-falsification confirming arm
        # (a positive-eps dose reading RED_FOR_INVESTIGATION) must never be
        # misclassified as an anomaly.
        columns = [{"dose_label": "eps0.50", "detected": "RED_FOR_INVESTIGATION"}]
        self.assertEqual(ab_merge.mutant_dose_ladder_anomalies(columns), [])

    def test_negative_red_and_not_detected_are_never_dose_anomalies(self):
        columns = [
            {"dose_label": "eps-0.50", "detected": "RED"},
            {"dose_label": "eps-0.10", "detected": "not-detected"},
        ]
        self.assertEqual(ab_merge.mutant_dose_ladder_anomalies(columns), [])

    def test_positive_eps_not_detected_is_not_a_falsification_finding(self):
        columns = [{"dose_label": "eps0.50", "detected": "not-detected"}]
        self.assertEqual(ab_merge.mutant_dose_ladder_two_sided_falsification(columns), [])

    def test_unparseable_dose_label_is_refused(self):
        # unit-63 round-7 audit finding 4: a dose_label that cannot be
        # placed in either branch must be refused, never silently skipped
        # or silently treated as a positive/negative default.
        columns = [{"dose_label": "bogus", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_sensitivity(columns)
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_two_sided_falsification(columns)

    def test_non_finite_zero_and_out_of_domain_eps_labels_are_refused(self):
        # unit-63 round-8 audit finding 3: `_dose_label_eps` parses
        # successfully for nan/0.0/-0.0/inf/an out-of-domain magnitude, but
        # `eps < 0.0`/`eps > 0.0` both silently reject each of these --
        # they must never vanish from BOTH findings with a clean exit, so
        # each one is refused loudly at parse time instead.
        for dose_label in ("epsnan", "eps0.0", "eps-0.0", "epsinf", "eps-inf", "eps10.0"):
            with self.subTest(dose_label=dose_label):
                with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
                    ab_merge._dose_label_eps(dose_label)

    def test_whitespace_or_explicit_plus_in_eps_substring_is_refused(self):
        # unit-63 round-9 audit advisory (a): float() is more permissive
        # than the raw leg file name lookup this label is used VERBATIM
        # for -- a whitespace-padded or explicit-plus-signed eps substring
        # parses fine but could silently diverge from the on-disk file
        # name, so it is refused here rather than accepted.
        for dose_label in ("eps 0.50", "eps0.50 ", "eps +0.50", "eps+0.50", "eps0.5\t0"):
            with self.subTest(dose_label=dose_label):
                with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
                    ab_merge._dose_label_eps(dose_label)

    def test_epsnan_is_refused_not_silently_dropped_from_both_findings(self):
        columns = [{"dose_label": "epsnan", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_sensitivity(columns)
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_two_sided_falsification(columns)

    def test_eps0_0_is_refused(self):
        columns = [{"dose_label": "eps0.0", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_sensitivity(columns)

    def test_eps_negative_zero_is_refused(self):
        columns = [{"dose_label": "eps-0.0", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_sensitivity(columns)

    def test_epsinf_is_refused(self):
        columns = [{"dose_label": "epsinf", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_two_sided_falsification(columns)

    def test_eps_magnitude_over_the_family_domain_is_refused(self):
        # eps=10.0 parses as a float fine, but its magnitude vastly exceeds
        # the scheduled ladder's own domain (|eps| <= 1.0) -- refused, never
        # silently accepted as though it were a real dose.
        columns = [{"dose_label": "eps10.0", "detected": "RED"}]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
            ab_merge.mutant_dose_ladder_sensitivity(columns)

    def test_eps_domain_boundary_oracle(self):
        # unit-63 round-9 audit finding 2: the domain is ASYMMETRIC --
        # `eps <= -1.0` (multiplier sign bound, exclusive of -1.0 itself)
        # and `eps > 1.0` (the family-sanity cap) are refused on the high
        # end, and `0 < abs(eps) < 0.01` (below the 0.01 sanity floor,
        # itself set deliberately BELOW the smallest ever-SCHEDULED dose of
        # `|eps| = 0.10` -- unit-63 round-10 audit advisory (a)) is refused
        # on the low end. A single symmetric
        # `abs(eps) > 1.0` check would have wrongly ACCEPTED eps=-1.0 (a
        # zero-update leg) as though it were a real degradation dose.
        refused = ("eps-1.0", "eps1.01", "eps0.009")
        accepted = {"eps-0.99": -0.99, "eps1.0": 1.0, "eps-0.01": -0.01}
        for dose_label in refused:
            with self.subTest(dose_label=dose_label, expect="refused"):
                with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError):
                    ab_merge._dose_label_eps(dose_label)
        for dose_label, expected_value in accepted.items():
            with self.subTest(dose_label=dose_label, expect="accepted"):
                self.assertAlmostEqual(ab_merge._dose_label_eps(dose_label), expected_value)

    def test_aliased_eps_labels_are_refused_regardless_of_order(self):
        # unit-63 round-10 audit F1: `eps-0.1` / `eps-0.100` / `eps-.10` /
        # `eps-1e-1` all parse to the identical eps=-0.1 float while filing
        # FOUR distinct leg files -- a stable abs(eps)-sort would otherwise
        # break the tie by caller order, silently emitting a zero-width
        # straddle between two aliases of the SAME dose. Both orderings of
        # each aliased pair must refuse -- never just one.
        aliased_pairs = [
            ("eps-0.1", "eps-0.100"),
            ("eps-0.1", "eps-.10"),
            ("eps-0.1", "eps-1e-1"),
        ]
        for label_a, label_b in aliased_pairs:
            for ordered in ((label_a, label_b), (label_b, label_a)):
                with self.subTest(order=ordered):
                    columns = [{"dose_label": lbl, "detected": "not-detected"} for lbl in ordered]
                    with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError) as ctx:
                        ab_merge.mutant_dose_ladder_reject_duplicate_doses(columns)
                    message = str(ctx.exception)
                    self.assertIn(ordered[0], message)
                    self.assertIn(ordered[1], message)
                    self.assertIn("-0.1", message)

    def test_distinct_eps_dose_set_is_unaffected_by_the_duplicate_guard(self):
        columns = [
            {"dose_label": "eps-0.50", "detected": "RED"},
            {"dose_label": "eps-0.10", "detected": "not-detected"},
            {"dose_label": "eps0.50", "detected": "RED_FOR_INVESTIGATION"},
        ]
        ab_merge.mutant_dose_ladder_reject_duplicate_doses(columns)  # must not raise

    def test_duplicate_literal_dose_label_is_refused_even_if_unparseable(self):
        # a literal-label duplicate is refused BEFORE eps parsing is ever
        # attempted -- two identically-spelled labels are refused even when
        # that label itself would fail to parse as a signed eps value.
        columns = [
            {"dose_label": "bogus", "detected": "RED"},
            {"dose_label": "bogus", "detected": "not-detected"},
        ]
        with self.assertRaises(ab_merge.MutantDoseLadderSensitivityError) as ctx:
            ab_merge.mutant_dose_ladder_reject_duplicate_doses(columns)
        self.assertIn("bogus", str(ctx.exception))
        self.assertIn("more than once", str(ctx.exception))

    def test_cli_wiring_refuses_aliased_dose_labels(self):
        # unit-63 round-10 audit F1 (CLI wiring): `--mutant-legs` supplied
        # twice with two aliased labels (eps-0.1 / eps-0.100) is a
        # merge-level refusal (exit 1, recorded in 'sensitivity_error'),
        # never a script crash and never a silently order-dependent
        # straddle.
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                _write_mutant_leg(raw_dir, seed, "eps-0.1", _mutant_tier(seed=seed, held_out_example_mean=0.70))
                _write_mutant_leg(raw_dir, seed, "eps-0.100", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    f"eps-0.1:{self.PATCH_SHA}:{seeds_s}",
                    "--mutant-legs",
                    f"eps-0.100:{self.PATCH_SHA}:{seeds_s}",
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 1)
        self.assertEqual(merged["status"], "GREEN")
        self.assertIsNone(merged["mutant_dose_ladder"]["sensitivity"])
        self.assertIsNotNone(merged["mutant_dose_ladder"]["sensitivity_error"])
        self.assertIn("eps-0.1", merged["mutant_dose_ladder"]["sensitivity_error"])
        self.assertIn("eps-0.100", merged["mutant_dose_ladder"]["sensitivity_error"])

    def test_cli_wiring_folds_the_dose_ladder_into_the_same_artifact(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                # 6-vs-6 (mixed sign) -- a GREEN main decision, isolating
                # this test's own claim (the dose ladder folds into the
                # SAME artifact, exit 0) from the decision-rule mutants.
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                mutant_mean = 0.70 if seed != 12 else 0.30
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=mutant_mean))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            mutant_spec = f"eps0.50:{self.PATCH_SHA}:{seeds_s}"
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    mutant_spec,
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 0)  # GREEN main decision + a RED (successfully-detecting) dose is not itself a script FAIL
        # unit-63 round-10 audit advisory (c): pin the main decision's own
        # status, per this file's own convention -- this test's isolation
        # claim (the dose ladder alone drives `rc`/the dose_anomalies etc.)
        # depends on the main decision actually being GREEN, not merely
        # "whatever it happened to compute" from the 6-vs-6 fixture above.
        self.assertEqual(merged["status"], "GREEN")
        self.assertIn("mutant_dose_ladder", merged)
        self.assertEqual(len(merged["mutant_dose_ladder"]["doses"]), 1)
        self.assertEqual(merged["mutant_dose_ladder"]["doses"][0]["detected"], "RED")

    def test_cli_wiring_fails_on_an_invalid_dose_column(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                # 6-vs-6 (mixed sign) -- a GREEN main decision, isolating
                # this test's own claim (an INVALID dose column alone fails
                # the merge's exit code) from the main decision rule.
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                if seed <= 11:  # only 11 mutant legs -- wrong clean pair count
                    _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            mutant_spec = f"eps0.50:{self.PATCH_SHA}:{seeds_s}"
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    mutant_spec,
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 1)
        # unit-63 round-10 audit advisory (c): pin the main decision's own
        # status, per this file's own convention -- this test's isolation
        # claim (an INVALID dose column ALONE fails the exit code) depends
        # on the main decision actually being GREEN.
        self.assertEqual(merged["status"], "GREEN")

    def test_cli_wiring_fails_on_a_negative_eps_red_for_investigation_dose(self):
        # unit-63 round-9 audit finding 3: a negative-eps (deflation) dose
        # reading RED_FOR_INVESTIGATION (11/12 seeds read BETTER than
        # alloff, an anomalous improvement under deflation) must non-zero
        # the merge's own exit code -- before this fix it silently yielded
        # sensitivity=null, sensitivity_error=null, exit 0.
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                # 6-vs-6 (mixed sign) -- a GREEN main decision, isolating
                # this test's own claim (a negative-eps dose_anomaly alone
                # fails the merge's exit code) from the main decision rule.
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                # mutant mean below BOTH branches' own alloff_mean (0.50 /
                # 0.40) so every non-dissenting seed reads improvement;
                # seed 12 dissents (above its own 0.40 alloff_mean) for
                # 11/12, mirroring the RED-detecting test's own shape.
                mutant_mean = 0.30 if seed != 12 else 0.70
                _write_mutant_leg(raw_dir, seed, "eps-0.10", _mutant_tier(seed=seed, held_out_example_mean=mutant_mean))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            mutant_spec = f"eps-0.10:{self.PATCH_SHA}:{seeds_s}"
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    mutant_spec,
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 1)
        # unit-63 round-10 audit advisory (c): pin the main decision's own
        # status, per this file's own convention -- this test's isolation
        # claim (a negative-eps dose_anomaly ALONE fails the exit code)
        # depends on the main decision actually being GREEN.
        self.assertEqual(merged["status"], "GREEN")
        self.assertEqual(merged["mutant_dose_ladder"]["doses"][0]["detected"], "RED_FOR_INVESTIGATION")
        self.assertIsNone(merged["mutant_dose_ladder"]["sensitivity"])
        self.assertEqual(
            merged["mutant_dose_ladder"]["dose_anomalies"],
            [
                {
                    "dose_label": "eps-0.10",
                    "eps": -0.10,
                    "detected": "RED_FOR_INVESTIGATION",
                    "finding": "anomalous improvement under deflation (eps < 0)",
                }
            ],
        )

    def test_cli_wiring_fails_on_an_unparseable_dose_label(self):
        # unit-63 round-7 audit finding 4: an operator-typo'd dose_label
        # that does not parse as a signed eps value is a merge-level
        # refusal (exit 1, recorded in the artifact's own
        # 'sensitivity_error'), never a script crash and never silently
        # skipped.
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                _write_mutant_leg(raw_dir, seed, "bogus", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            mutant_spec = f"bogus:{self.PATCH_SHA}:{seeds_s}"
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    mutant_spec,
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertEqual(rc, 1)
        self.assertIsNone(merged["mutant_dose_ladder"]["sensitivity"])
        self.assertIsNotNone(merged["mutant_dose_ladder"]["sensitivity_error"])

    def test_cli_wiring_strips_the_mutant_legs_sha_before_comparison(self):
        # unit-63 round-8 audit finding 4 (merger half): the CLI's own
        # `--mutant-legs DOSE_LABEL:PATCH_SHA256:SEEDS` sha is stripped
        # before it is used, so a whitespace-padded sha on the command line
        # is not silently reported as a labeling-error mismatch against a
        # leg's own (clean) recorded sha, and is recorded stripped in the
        # merged artifact.
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for seed in range(1, 13):
                fused_mean, alloff_mean = (0.30, 0.50) if seed <= 6 else (0.55, 0.40)
                for arm, mean in (("fused", fused_mean), ("alloff", alloff_mean)):
                    for repeat in ("r1", "r2"):
                        _write_finetune_run_leg(
                            raw_dir, seed, arm, repeat, _finetune_run_tier(arm=arm, seed=seed, held_out_example_mean=mean)
                        )
                _write_mutant_leg(raw_dir, seed, "eps0.50", _mutant_tier(seed=seed, held_out_example_mean=0.70))
            seeds_s = ",".join(str(s) for s in range(1, 13))
            mutant_spec = f"eps0.50: {self.PATCH_SHA} :{seeds_s}"
            rc = ab_merge.main(
                [
                    "finetune-run",
                    raw_dir,
                    out_dir,
                    seeds_s,
                    "",
                    "--allow-missing-lr0-control",
                    "--mutant-legs",
                    mutant_spec,
                ]
            )
            with open(os.path.join(out_dir, "finetune_run_ab_report.json")) as fh:
                merged = json.load(fh)
        dose = merged["mutant_dose_ladder"]["doses"][0]
        self.assertEqual(dose["patch_sha256"], self.PATCH_SHA)
        self.assertEqual(dose["violations"], [])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
