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

    `test_real_flash_on_fixture_is_valid_true` /
    `test_real_flash_off_fixture_is_valid_reference_leg` drive the two
    REAL, committed raw-run reports (`fixtures/p6_fa2_dense_raw_runs/`,
    provenance in that directory's own `PROVENANCE.md`) through
    `ab_merge.main` unmodified -- never a hand-rolled dict standing in for
    what that branch's own binary actually emitted. Every other test here
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

    def test_real_flash_on_fixture_is_valid_true(self):
        """THE BUG REPRODUCTION: before this fix, this raised `KeyError`
        (via `dispatch_pairs`) on this exact fixture, caught per-leg by
        `build_report` and surfaced as an `"ERROR: ..."` string, never
        `True`. `s128_flash_on_1.json` reads `attention_block_flash_fused_
        dispatches: 840`, `..._declined_dispatches: 0`,
        `attention_block_fused_dispatches: 0` -- the flash-ON leg.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-on", load_fixture_finetune_step("s128_flash_on_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-on"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_real_flash_off_fixture_is_valid_reference_leg(self):
        """THE BUG REPRODUCTION, the reference-leg side: before this fix,
        this ALSO raised `KeyError` on the exact same missing-sibling shape
        (`attention_block_flash_fused_dispatches` present,
        `..._eager_dispatches` absent -- the fallback key is
        `..._declined_dispatches` here too, just nonzero: `840`).
        `s128_flash_off_1.json` reads `attention_block_flash_fused_
        dispatches: 0`, `..._declined_dispatches: 840`,
        `attention_block_fused_dispatches: 840`,
        `kernels_disabled_requested == kernels_disabled_fired ==
        ["attention_block_flash"]` -- the JAMMI_KERNELS_DISABLE=
        attention_block_flash reference leg, and its `declined: 840` must
        NOT be treated as a silent fallback (rule 1's exemption).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-off", load_fixture_finetune_step("s128_flash_off_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-off"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)

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
        adds to the module docstring's determinant table.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(
                raw_dir, "b8-s128-flash-on", load_fixture_finetune_step("s128_flash_on_1")
            )
            out_dir = tempfile.mkdtemp()
            rc = ab_merge.main([raw_dir, out_dir, "25", "5", "0.9"])
            self.assertEqual(rc, 0)
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


if __name__ == "__main__":
    unittest.main()
