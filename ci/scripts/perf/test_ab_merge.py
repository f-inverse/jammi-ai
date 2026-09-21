#!/usr/bin/env python3
"""Fixture-directory tests for `ab_merge.py` — the merge/table stage
`finetune_ab.sh` invokes as `python3 "$DIR/ab_merge.py" ...`.

`AB_DRY_RUN=1` alone only exercises the DRY_RUN arm, never a real report
shape, so every test here builds a fixture directory shaped
EXACTLY like `run_leg`'s own `.exit`/`.json`/`.stderr` triples, then drives
`ab_merge.main(argv)` — the REAL entry point `finetune_ab.sh` calls — never
`fused_proof()` or `dispatch_pairs()` in isolation with literal tuples
standing in for a report.

Stdlib-only (`unittest`), no external dependency — same footing
`torch_finetune_step.py`'s own "never a Cargo dependency, never a pinned
requirements file" stance (crates/jammi-bench/reference/README.md): this is a
CI-adjacent script, not a package, and nothing here should ever tempt CI into
enforcing a Python requirements file against a crate that has no Python
toolchain.

`CascadePairFixtureTests` additionally drives two REAL, committed raw-run
reports (`fixtures/p6_fa2_dense_raw_runs/*.json`, provenance in that
directory's own `PROVENANCE.md`) through this same real entry point — never
a hand-rolled dict standing in for what a real `finetune-step` binary
emitted.

`OptionalNonCascadePairFixtureTests` covers `ab_merge.py`'s
`OPTIONAL_NON_CASCADE_PAIRS` classification (`gelu`): a `(0, 0)` reading is
legitimate (the dense erf-GELU seam is never called on a ModernBERT leg,
whose GeGLU MLP is a different, already-covered pair), a live `(n, 0)`
reading validates like any ordinary pair, a genuine `(0, n)` eager fallback
still hard-fails, and a wholly separate unclassified base still raises the
loud schema-drift error `dispatch_pairs` always has.

Run directly: `python3 ci/scripts/perf/test_ab_merge.py`
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import re
import subprocess
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
# building this fixture set). `adamw` is the multi-tensor AdamW pair. This
# list is hand-kept, so `GoldenProducerAnchoredFieldSetTests` pins it against
# a REAL producer report rather than trusting it alone.
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
        # `seed` (`crates/jammi-bench/src/report.rs`'s
        # `FinetuneStepTier::seed` field): the leg-premise check below reads
        # it directly off THIS sub-block for a jammi leg (jammi's own
        # `finetune_step.rs` carries it inline, unlike torch's — see
        # `torch_fs`'s own doc for why torch's lives one level up).
        "seed": 42,
        "backbone_dtype": "bf16",
        # Checkpoint content identity, lora_alpha, and margin. Same literal
        # defaults as `torch_fs` below so a matching-premise pair (the
        # overwhelming default use of both fixtures) stays matching without
        # every call site having to override them.
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
        # The clip's REQUEST (`null` = clip off, the sweep's default), its
        # COUNTED fact (`0` when off), and the attention reference class —
        # all three on every real `FinetuneStepTier`. `attention_arm` defaults
        # to `"fused"` here (the jammi-fused leg is the one the premise check
        # reads); a test that writes an eager or clip-on leg overrides them
        # explicitly.
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
        # (a producer that does not emit it) deletes it from the returned
        # dict directly.
        if value is None and key not in ab_merge.FINETUNE_NULL_IS_A_VALUE_FIELDS:
            del fs[key]
    return {"tiers": {"finetune_step": fs}}


def drop_dispatch_keys(*bases):
    """An `overrides` dict for `jammi_fs` that DELETES both
    `_fused_dispatches` and `_eager_dispatches` for each given base
    entirely (`jammi_fs`'s own `None`-deletes-the-key convention) --
    simulates the base being ABSENT from the schema (a field renamed,
    deleted, or feature-gated off), never merely reading `(0, 0)`.
    `fused_proof` requires every `ALL_BASES` member to be PRESENT; this is
    how the fixtures below construct the "classified base vanished from the
    schema entirely" regression it catches.
    """
    overrides = {}
    for base in bases:
        overrides[f"{base}_fused_dispatches"] = None
        overrides[f"{base}_eager_dispatches"] = None
    return overrides


def flash_overrides(fused=0, declined=0, compiled=True, disabled_requested=None, disabled_fired=None):
    """An `overrides` dict for `jammi_fs` that adds the FlashAttention-2
    cascade fields (`attention_block_flash_fused_dispatches`/
    `..._declined_dispatches`/`flash_compiled`/`kernels_disabled_requested`/
    `kernels_disabled_fired`) — NONE of these are in `jammi_fs`'s own base
    dict (a report may carry no `flash` key at all), so this is additive,
    never a replacement of an existing key. `disabled_requested`/
    `disabled_fired` default to `[]` (not `None`) — `fs.get(...)` on a real
    report never reads `null` for these two list fields (their field doc:
    "Always present, even on an ordinary run with nothing disabled").
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
    committed `jammi-bench finetune-step` raw-run report (see that
    directory's own `PROVENANCE.md`) — and returns its FULL top-level dict (the
    same shape `write_leg` writes straight to a `.json` fixture file), never
    just the `finetune_step` sub-block in isolation.
    """
    with open(os.path.join(FIXTURES_DIR, f"{name}.json")) as fh:
        return json.load(fh)


def load_golden(name):
    """Reads `fixtures/finetune_run_golden/<name>.json` — a REAL, committed
    `jammi-bench finetune-run` report, run once by the actual compiled
    binary (see that directory's own `PROVENANCE.md` for the exact CLI
    invocation and git sha), never a hand-typed field list standing in for
    what the producer actually serializes, so a field a hand-written literal
    dict would forget (e.g. `adamw_{fused,eager}_dispatches`) is still
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
    `finetune_step` sub-block (e.g. `attn_implementation="eager"`).
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
        # Same trio as `jammi_fs`, same defaults, so
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


def write_two_run_marker(raw_dir):
    """`finetune_ab.sh`'s own `TWO_RUN_PROTOCOL_MARKER` file — see
    `ab_merge.TWO_RUN_PROTOCOL_MARKER`'s own doc. Written the SAME way the
    real script does (`touch`, empty file, presence-only signal).
    """
    open(os.path.join(raw_dir, ab_merge.TWO_RUN_PROTOCOL_MARKER), "w").close()


def write_second_run(raw_dir, slug, jammi_tps=750.0, torch_tps=700.0, jammi_overrides=None, torch_overrides=None):
    """The A,B,B,A protocol's SECOND run of the bar pair
    (`jammi-fused-2`/`torch-sdpa-2`) — `_CLEAN_YES_DISPATCHES`-shaped by
    default so `metrics()`'s own `dispatch_pairs()` call on the second
    jammi-fused leg never raises unless a caller deliberately overrides
    the dispatch counters.
    """
    jammi_overrides = dict(jammi_overrides or {})
    jammi_overrides.setdefault("triplets_per_s", {"value": jammi_tps, "unit": "triplets/s"})
    torch_overrides = dict(torch_overrides or {})
    torch_overrides.setdefault("triplets_per_s", {"value": torch_tps, "unit": "triplets/s"})
    write_leg(raw_dir, slug, "jammi-fused-2", report=jammi_fs(_CLEAN_YES_DISPATCHES, **jammi_overrides))
    write_leg(raw_dir, slug, "torch-sdpa-2", report=torch_fs(**torch_overrides))


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

        A `False` proof turns the verdict `INVALID` (never merely a `[WARN]`
        suffix on whatever ratio-based verdict would have applied) and the
        SWEEP's own exit code goes non-zero -- driven at the real `main()`
        entry point, not `build_report`'s internals in isolation.
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
        """A `_fused_dispatches` key with no `_eager_dispatches`
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
        # One bad leg must not abort the merge for the OTHER config --
        # both configs are still present and correctly classified in the
        # merged JSON, one bad leg does not silently swallow the other.
        self.assertTrue(merged["configs"]["b8-s128-solo"]["verdict"].startswith("INVALID"))
        self.assertIs(merged["configs"]["b8-s128-healthy"]["jammi_fused_dispatch_proof"], True)
        self.assertFalse(merged["configs"]["b8-s128-healthy"]["verdict"].startswith("INVALID"))
        # The error is visible in the printed table too, not just the JSON.
        self.assertIn("ERROR", table)
        # The errored config's own INVALID verdict is what gates the
        # SWEEP's exit code non-zero -- the healthy config
        # passing does not paper over it.
        self.assertEqual(rc, 1)

    def test_vanished_site_case_all_zero_except_attention_block(self):
        """Worst case for a blanket "(0, 0) is fine" rule: ln/rope/softmax/
        geglu/lora_epilogue/lora_linear ALL read (0, 0) and only
        attention_block reads (10, 0). This must be NO, because `ln`
        (REQUIRED, absorbed by nothing) reads (0, 0).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {"attention_block": (10, 0)})
            rc, merged, _table = self.run_merge(raw_dir)
        self.assertIs(merged["configs"]["b8-s128-d0"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-d0"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)

    def test_vanished_site_case_only_lora_epilogue_positive(self):
        """Only `lora_epilogue` reads (1, 0),
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
        standing in for a report.
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
        # rules.
        self.assertFalse(merged["configs"]["b8-s128-missing"]["verdict"].startswith("INVALID"))
        self.assertIs(merged["configs"]["b8-s128-empty"]["jammi_fused_dispatch_proof"], False)
        self.assertTrue(merged["configs"]["b8-s128-empty"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1, "the b8-s128-empty config's INVALID verdict must gate the sweep exit code")

    def test_geglu_zero_zero_now_fails_the_f5_reproduction(self):
        """`geglu = (0, 0)` (present, reading zero -- e.g. a
        deleted/feature-gated-off fused MLP) must fail the proof even when
        `ln`/`attention_block`/`lora_linear` each independently clear their
        own bar: `geglu` is a `REQUIRED_PAIRS` member (matching
        `finetune_ab.sh`'s header). Must be NO.
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
        """`fused_proof([('ln', 9, 0), ('lora_linear', 3, 0)])` --
        rope/softmax/geglu/attention_block ALL entirely ABSENT from the
        report's schema, not merely reading `(0, 0)`. Skipping an absent
        `ABSORBABLE_BY_ATTENTION_BLOCK` member would grant it a free pass no
        `REQUIRED_PAIRS` member gets. Must be NO: an ABSENT classified base
        is a hard fail for EVERY class.
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
        """The most extreme absence case: `fused_proof([('ln', 9, 0)])` --
        EVERY OTHER classified base entirely missing from the schema. Must
        be NO.
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
        silently-ignored/exempted base -- caught per-leg, never
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
        # An errored proof is INVALID, and INVALID gates the sweep's exit
        # code -- unlike the "one bad leg must not abort the merge for
        # another config" guarantee (which is about NOT
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
    """`attention_block_flash_fused_dispatches` has no
    `_eager_dispatches` sibling — its fallback counter is
    `_declined_dispatches` instead (`CASCADE_BASES`). Looking for an
    `_eager_dispatches` sibling would make `dispatch_pairs` raise `KeyError`
    on the committed fixtures below, and `build_report`'s per-leg
    `try`/`except` would turn every leg of every config `INVALID`.

    `test_real_flash_on_fixture_no_longer_keyerrors_but_predates_adamw` /
    `test_real_flash_off_fixture_no_longer_keyerrors_but_predates_adamw`
    drive the two REAL, committed raw-run reports
    (`fixtures/p6_fa2_dense_raw_runs/`, provenance in that directory's own
    `PROVENANCE.md`) through `ab_merge.main` unmodified -- never a
    hand-rolled dict standing in for what a real binary emitted (both read
    INVALID, not a `KeyError` crash -- see each test's own doc).
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
        """`dispatch_pairs` classifies `attention_block_flash` on this exact
        fixture cleanly -- no `KeyError`, no `"ERROR: ..."` string.

        `s128_flash_on_1.json` (see
        `fixtures/p6_fa2_dense_raw_runs/PROVENANCE.md`) PREDATES the
        multi-tensor AdamW counters, so this report carries no
        `adamw_{fused,eager}_dispatches` keys AT ALL. `adamw` is a
        `REQUIRED_PAIRS` member, and an ABSENT required base is a hard fail
        (see `fused_proof`'s own doc), never a silently-granted exemption for
        an older schema. The verdict is therefore a CORRECT INVALID naming a
        real schema-staleness fact about this fixture --
        `RealAdamwArtifactFixtureTests` below drives the actual GREEN
        (adamw-carrying) shape this proof exists to pass.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-on", load_fixture_finetune_step("s128_flash_on_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-on"]
        self.assertIsNot(cfg["jammi_fused_dispatch_proof"], None)
        self.assertNotIsInstance(
            cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"]
        )  # never an "ERROR: ..." KeyError string
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_real_flash_off_fixture_no_longer_keyerrors_but_predates_adamw(self):
        """The reference-leg side of the same missing-sibling shape
        (`attention_block_flash_fused_dispatches` present,
        `..._eager_dispatches` absent -- the fallback key is
        `..._declined_dispatches` here too, just nonzero: `840`).
        `s128_flash_off_1.json` reads `attention_block_flash_fused_
        dispatches: 0`, `..._declined_dispatches: 840`,
        `attention_block_fused_dispatches: 840`,
        `kernels_disabled_requested == kernels_disabled_fired ==
        ["attention_block_flash"]` -- the JAMMI_KERNELS_DISABLE=
        attention_block_flash reference leg, and its `declined: 840` is
        correctly NOT treated as a silent fallback (rule 1's exemption).

        This fixture also predates the AdamW counters -- see the flash-on
        sibling's own doc for why `REQUIRED_PAIRS`'s absence rule fails
        this leg too.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_jammi_fused_only(raw_dir, "b8-s128-flash-off", load_fixture_finetune_step("s128_flash_off_1"))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-flash-off"]
        self.assertNotIsInstance(
            cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"]
        )  # never an "ERROR: ..." KeyError string
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_nothing_ran_in_attention_arm_is_invalid(self):
        """Truth-table case 3: `attention_block_flash` reads `(0, 0)` AND
        `attention_block` ALSO reads `fused == 0` -- the whole attention
        arm dispatched nothing at all. Built from the real flash-off
        fixture with only the attention-arm counters and the (then
        inapplicable) disable-request fields zeroed out.
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
        """`flash_compiled is False` is NOT exempted from rule 1: `fused_proof`
        is SHARED by every sweep; a premise fact that voids ONE sweep (the
        finetune-run A/B's fused arm IS the flash cascade) belongs in THAT
        sweep's own premise check
        (`finetune_run_dispatch_proof_violations`'s `arm == "fused"`
        branch), never a silent, generic exemption inside the shared
        primitive. This is the regression pin: the exact capability-miss
        shape (a real disable request CLEARED, so this is genuinely a
        capability miss, not also a self-describing one, plus a synthetic
        `adamw` pair -- this fixture predates the multi-tensor AdamW
        counters) must hard-fail, unconditionally, exactly like an ordinary
        silent eager fallback.
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
        padding, wrong arch, `flash-attn` not compiled) must hard-fail
        exactly like an ordinary silent eager fallback. Built
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
        """A report with NO `attention_block_flash` key at all must treat
        `attention_block` EXACTLY as a `REQUIRED_PAIRS` member --
        `attention_block` reading `(0, 0)` with no flash key present is a
        hard fail, never silently exempted by the flash absorption rule.
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
        both recorded, never compared, the "provenance" row of the module
        docstring's determinant table. `leg_provenance`
        records the RAW counters unconditionally (it is never itself gated
        by `fused_proof`), so this still holds even though this exact real
        fixture predates the multi-tensor AdamW counters (`rc`/`verdict`
        read INVALID/1 for THIS config, since `adamw` is a `REQUIRED_PAIRS`
        member this schema-older report cannot supply -- see
        `RealAdamwArtifactFixtureTests` for the GREEN, adamw-carrying leg this
        proof exists to pass).
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


class OptionalNonCascadePairFixtureTests(unittest.TestCase):
    """`gelu_fused_dispatches`/`gelu_eager_dispatches` (an ORDINARY
    pair — a plain `_eager_dispatches` fallback, never the `CASCADE_BASES`
    `_declined_dispatches` shape) is unconditionally serialized by
    `FinetuneStepTier`, but its `fused > 0` half is architecture-conditional
    (`OPTIONAL_NON_CASCADE_PAIRS` — see that set's own module-level doc):
    the dense erf-GELU seam it counts is BERT's/DistilBERT's FFN only, and a
    ModernBERT leg's GeGLU MLP (already covered by the separate `geglu`
    REQUIRED_PAIRS member) never reaches it at all. An unclassified `gelu`
    would make `dispatch_pairs` raise `KeyError('gelu' ... not classified
    in ALL_BASES)` on every real `finetune_ab.sh` leg — exercised below via
    `jammi_fs`'s own `_CLEAN_YES_DISPATCHES`-shaped fixture plus the two
    `gelu_*` keys
    (`gelu` is deliberately NOT a member of this file's own local
    `ALL_BASES` tuple, so `jammi_fs` never adds it by default — every test
    here adds it explicitly via `overrides`, exactly mirroring how a real
    `FinetuneStepTier` report carries it alongside every other pair).

    Admission here is by THIS LEG'S OWN COUNTERS (tensor state), never a
    model-identity/architecture-name branch — no test below reads or
    fabricates a `"backbone"`/model-name field; a `(0, 0)` reading and an
    `(n, 0)` reading are both exercised purely via the counter VALUES.
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

    def test_gelu_pair_reading_zero_zero_is_not_invalid(self):
        """(a) A `FinetuneStepTier`-shaped fixture carrying the `gelu_*`
        pair at `(0, 0)` (the ModernBERT shape — the dense erf-GELU seam
        never called) must clear `dispatch_pairs` without raising, and the
        leg's own `fused_proof` must NOT be dragged down to `False`/INVALID
        by that reading alone — every OTHER pair in `_CLEAN_YES_DISPATCHES`
        already independently proves the leg.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES, gelu_fused_dispatches=0, gelu_eager_dispatches=0)
            self.write_jammi_fused_only(raw_dir, "b8-s128-gelu-zero", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-gelu-zero"]
        self.assertNotIsInstance(cfg["jammi_fused_dispatch_proof"], str, cfg["jammi_fused_dispatch_proof"])
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)
        fs = report["tiers"]["finetune_step"]
        pairs = dict((base, (fused, fallback)) for base, fused, fallback in ab_merge.dispatch_pairs(fs))
        self.assertEqual(pairs["gelu"], (0, 0))

    def test_gelu_pair_reading_live_validates_like_an_ordinary_pair(self):
        """(b) A BERT-family leg where the dense erf-GELU seam actually
        fired (`gelu_fused_dispatches > 0`, `gelu_eager_dispatches == 0`)
        validates cleanly — a live pair is not somehow rejected merely
        because this base's presence-with-zero reading is also tolerated.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES, gelu_fused_dispatches=12, gelu_eager_dispatches=0)
            self.write_jammi_fused_only(raw_dir, "b8-s128-gelu-live", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-gelu-live"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], True, cfg["jammi_fused_dispatch_proof"])
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 0)
        fs = report["tiers"]["finetune_step"]
        pairs = dict((base, (fused, fallback)) for base, fused, fallback in ab_merge.dispatch_pairs(fs))
        self.assertEqual(pairs["gelu"], (12, 0))

    def test_gelu_live_pair_eager_fallback_still_hard_fails(self):
        """Rule 1 is NOT relaxed for this base: a genuine
        `gelu_eager_dispatches > 0` (the admitted seam actually fell back)
        is a hard fail exactly like any ordinary pair's eager fallback —
        `OPTIONAL_NON_CASCADE_PAIRS` only tolerates `(0, 0)`, never a
        counted, nonzero fallback.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES, gelu_fused_dispatches=0, gelu_eager_dispatches=3)
            self.write_jammi_fused_only(raw_dir, "b8-s128-gelu-fallback", report)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-gelu-fallback"]
        self.assertIs(cfg["jammi_fused_dispatch_proof"], False, cfg["jammi_fused_dispatch_proof"])
        self.assertTrue(str(cfg["verdict"]).startswith("INVALID"), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_unknown_base_distinct_from_gelu_still_raises(self):
        """(c) RED control: the OPTIONAL_NON_CASCADE_PAIRS classification for
        `gelu` must not widen `ALL_BASES` into a silent catch-all — a WHOLLY
        DIFFERENT, still-unclassified base (`swiglu`, a plausible next real op,
        picked precisely because it is NOT `gelu`) landing in the same report
        as a legitimate `gelu` pair must raise the loud, per-leg schema-drift
        error.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            report = jammi_fs(_CLEAN_YES_DISPATCHES, gelu_fused_dispatches=0, gelu_eager_dispatches=0)
            report["tiers"]["finetune_step"]["swiglu_fused_dispatches"] = 4
            report["tiers"]["finetune_step"]["swiglu_eager_dispatches"] = 0
            self.write_jammi_fused_only(raw_dir, "b8-s128-swiglu-unknown", report)
            rc, merged = self.run_merge(raw_dir)
        proof = merged["configs"]["b8-s128-swiglu-unknown"]["jammi_fused_dispatch_proof"]
        self.assertIsInstance(proof, str)
        self.assertIn("ERROR", proof)
        self.assertIn("swiglu", proof)
        self.assertTrue(merged["configs"]["b8-s128-swiglu-unknown"]["verdict"].startswith("INVALID"))
        self.assertEqual(rc, 1)


# The committed real artifact, NOT copied into this crate's own `fixtures/`
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
    """Every real leg of this committed artifact carries the `adamw` pair,
    so `dispatch_pairs` must classify it (`adamw` in
    `ALL_BASES`/`REQUIRED_PAIRS`) rather than raise `KeyError('adamw')` —
    checked directly against `b8_s512_fused.r2.json.raw` and its siblings,
    never merely asserted. Every test here reads the REAL
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
        """This leg's own
        `kernels_disabled_requested`/`kernels_disabled_fired` are BOTH empty
        (the fused arm, no disable request at all) and every `REQUIRED_PAIRS`/
        `ABSORBABLE_BY_ATTENTION_BLOCK_FLASH`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
        `LORA_SITE_EXCLUSIVE_GROUP` member this leg's own schema carries
        shows a real, positive fused count (`ln`, `geglu`, `adamw`,
        `attention_block`, `lora_linear` all `> 0`; `rope`/`softmax`
        legitimately absorbed at `(0, 0)` via `attention_block`'s own
        `fused > 0`) — this is the GENUINE green shape `fused_proof` exists
        to pass, with `dispatch_pairs` classifying `adamw`.
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
        """The sibling shape at the OTHER committed seq length — a second
        real leg."""
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
        eager fallback on a REQUIRED pair), never a `KeyError`.
        This leg is never itself passed through `fused_proof` by
        `build_report` (only the `jammi-fused` leg is), so this test drives
        `dispatch_pairs` directly — the function that would raise.
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
        an unclassified `adamw` would raise on ALL EIGHT of these.
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
    """`ab_merge.py`'s premise-identity check: identity is a checked RECORD
    in the merged artifact, never an ASSUMPTION from `finetune_ab.sh`'s own
    matched CLI flags. Every test here uses `_CLEAN_YES_DISPATCHES`
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
        every other fixture test would be a false negative waiting to
        happen.
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
        """A jammi binary that does not emit `FinetuneStepTier::seed` -- the
        field is simply absent, not present-and-wrong. Must refuse just as
        loudly as a value mismatch, never silently skip the check because one
        side has nothing to compare.
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
        """Two legs measured at a DIFFERENT step count (e.g. `--steps 20` vs
        `--steps 5`, a mismatched per-leg override) must not merge to a
        "clean" ratio and PASS verdict -- `steps_measured` is recorded on
        BOTH sides and compared.
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
        determinant table covers, applied to this tier too.
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
        """Present-but-`null` (`None` in the fixture dict, matching a JSON
        `null` — e.g. `serde_json` serializing a NaN `lora_alpha`) on BOTH
        legs must not compare `None == None` and silently PASS, the same
        class `compare_grad_oracle.py` refuses on the grad-oracle side.
        `jammi_fs`'s
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
    """`--lora-init` is overridable, and the merged report records
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
    """Losses are bf16-sourced; the table must not print more decimal
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
    """`max_grad_norm` and the attention reference class are IDENTITY —
    two legs differing in either compute a different step, so a clip-on
    jammi leg must never merge against a clip-off torch leg and print PASS.
    Every test here drives `ab_merge.main`
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
        """jammi ran `--max-grad-norm 1.0` (and counted 26 clip calls: 20
        steps + 5 warmup + 1 pre-step), torch ran with the flag absent.
        Must be refused, never PASS.
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
        OFF) and must compare as a matching VALUE — the null-folds-to-MISSING
        rule is deliberately NOT applied to this field
        (`identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`)."""
        with tempfile.TemporaryDirectory() as raw_dir:
            self.write_pair(raw_dir)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertEqual(rc, 0)

    def test_max_grad_norm_absent_from_a_leg_is_still_missing(self):
        """A jammi binary that does not emit the field: the KEY is absent
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
        """The counted fact must back the request. `max_grad_norm: 1.0`
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
        """`build_report` falls back to the
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
        """A jammi-fused leg on a checkpoint the
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


class TorchIdentityFieldsInProducerSourceTests(unittest.TestCase):
    """Every `torch_finetune_step.py::TORCH_IDENTITY_FIELDS` entry is a key
    the producer actually emits, read off its source with stdlib `ast` (no
    torch). Whether each emitted VALUE is non-null is a run-time fact, held by
    `test_torch_finetune_step_dry_run.py` on a host with a torch venv.

    The scan is scoped to exactly the THREE places this producer assembles
    emitted JSON —
    `provenance()`'s own `info = {...}` dict, `checkpoint_identity()`'s own
    return dict (the `**checkpoint_identity_fields` unpack inside the
    `finetune_step` block — an unpack is not a literal string key `ast.Dict.
    keys` sees, so its SOURCE function is walked directly instead), and the
    ONE `report = {...}` literal inside `run()` (identified structurally,
    by the dict literal that carries BOTH a `"finetune_step"` AND a
    `"provenance"` key, so a rename cannot silently re-target the
    wrong dict) — never every `ast.Dict` in the module. Collecting keys
    from EVERY dict in the module would be vacuous: it also sweeps in
    `TORCH_IDENTITY_FIELDS_NULL_MEANS` (the classification
    table declared two lines below `TORCH_IDENTITY_FIELDS` itself, whose
    keys are the SAME field names) — a field declared in
    `TORCH_IDENTITY_FIELDS` and named ONLY in `NULL_MEANS`, never actually
    assigned anywhere the producer emits, would pass. Mutation check:
    add `"max_grad_norm"` to `TORCH_IDENTITY_FIELDS` and to
    `TORCH_IDENTITY_FIELDS_NULL_MEANS` ONLY (never to `provenance()`'s or
    `run()`'s own dict literals) — this test goes RED.
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


class GoldenProducerAnchoredFieldSetTests(unittest.TestCase):
    """The SET of `*_fused_dispatches`/`*_eager_dispatches`/
    `*_declined_dispatches` base names a REAL, committed `jammi-bench
    finetune-run` report carries must equal exactly what
    `ab_merge.ALL_BASES` classifies -- neither side a strict subset of the
    other. This REDs the instant a base the producer emits (e.g. `adamw`)
    is missing from `ALL_BASES`, rather than waiting for a real leg to hit
    `dispatch_pairs`'s own `KeyError` in a live sweep.

    All three committed goldens are read (`bert_fused`, `modernbert_fused`,
    `modernbert_alloff` — all three are real, single, producer-emitted
    `jammi-bench finetune-run` reports, never a composite; see the goldens'
    own `PROVENANCE.md` for each one's seed/git_sha) —
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
        golden's own `finetune_run` tier (the mechanism that would
        `KeyError` on an unclassified base such as `adamw`).
        """
        for name in ("bert_fused", "modernbert_fused", "modernbert_alloff"):
            tier = load_golden(name)["tiers"]["finetune_run"]
            pairs = ab_merge.dispatch_pairs(tier)  # must not raise
            self.assertEqual({base for base, _fused, _fallback in pairs}, ab_merge.ALL_BASES)


class FinetuneAbVerdictInvalidPrefixNamedConstantTests(unittest.TestCase):
    """Like the `MUTANT_DOSE_DETECTED_*`/`RED_PROOF_VERDICT_*` constants:
    `build_report`'s own `verdict` INVALID prefix (both production sites --
    the fused-dispatch-proof branch and the leg-premise-mismatch branch)
    and `main()`'s own `.startswith(...)` consumption of it all read
    `ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX`, never a re-typed
    `"INVALID"` literal. Pinned end-to-end through the real
    `ab_merge.main` entry point (the same fixture
    `FusedProofFixtureTests.test_all_zero_no` already exercises), proving
    the constant's OWN value is what both the producer and the consumer
    agree on, not merely that the constant equals the string `"INVALID"`
    in isolation.
    """

    def test_constant_is_the_literal_invalid(self):
        self.assertEqual(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX, "INVALID")

    def test_build_report_invalid_verdict_and_mains_own_exit_gate_agree_with_the_constant(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {})  # all-(0, 0) -> INVALID, see test_all_zero_no
            out_dir = tempfile.mkdtemp()
            rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
            with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
                merged = json.load(fh)
        self.assertTrue(
            merged["configs"]["b8-s128-d0"]["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX)
        )
        self.assertEqual(rc, 1, "main()'s own exit-code gate must agree with the same named prefix")


class OrderBalancedBarLegsTests(unittest.TestCase):
    """finetune_ab.sh's own A,B,B,A order-balanced bar-leg protocol
    (`jammi-fused`/`torch-sdpa` run twice per config) — drives
    `ab_merge.main` (the real entry point) against fixture `raw_dir`s that
    additionally carry `jammi-fused-2`/`torch-sdpa-2` legs
    (`ab_merge.BAR_SECOND_RUN_LEGS`). A config using `write_ok_config`
    ALONE (no `-2` legs at all — every OTHER test class in this file) is
    the regression guard for the single-run shape: `bar_ratio ==
    ratio_jammi_fused_over_torch_sdpa` and `bar_ratio_indeterminate is
    False` whenever the second run never ran, which every such fixture in
    this file asserts implicitly.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        with open(os.path.join(out_dir, "finetune_ab_table.txt")) as fh:
            table = fh.read()
        return rc, merged, table

    def write_second_run(self, raw_dir, slug, jammi_tps, torch_tps):
        """Writes the A,B,B,A protocol's SECOND run of the bar pair
        (`jammi-fused-2`/`torch-sdpa-2`) — `_CLEAN_YES_DISPATCHES`-shaped
        so `metrics()`'s own `dispatch_pairs()` call on the second
        jammi-fused leg never raises (this class is not exercising
        `fused_proof`, which stays keyed to the FIRST run only).
        """
        write_leg(
            raw_dir,
            slug,
            "jammi-fused-2",
            report=jammi_fs(_CLEAN_YES_DISPATCHES, triplets_per_s={"value": jammi_tps, "unit": "triplets/s"}),
        )
        write_leg(
            raw_dir,
            slug,
            "torch-sdpa-2",
            report=torch_fs(triplets_per_s={"value": torch_tps, "unit": "triplets/s"}),
        )

    def test_no_second_run_legs_falls_back_to_the_single_pair_ratio_unchanged(self):
        """Backward compatibility: an older `raw_dir` (this file's own
        244 pre-existing fixtures) carries no `-2` legs at all — `bar_ratio`
        must equal the ORIGINAL single-pair `ratio_jammi_fused_over_torch_sdpa`
        exactly, `bar_ratio_indeterminate` must be `False`, and `pair2_ratio`
        must be `None`.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["bar_pair_ratios"]["pair2_jammi_fused_2_over_torch_sdpa_2"])
        self.assertFalse(cfg["bar_ratio_indeterminate"])
        self.assertEqual(
            cfg["bar_ratio_min_of_two_least_favourable_to_jammi"],
            cfg["ratio_jammi_fused_over_torch_sdpa"],
        )
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_both_pairs_clear_the_bar_is_pass_using_the_min_of_the_two(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # pair1: 800/727 ~= 1.100
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=750.0, torch_tps=700.0)  # pair2 ~= 1.071
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        pair1 = cfg["bar_pair_ratios"]["pair1_jammi_fused_over_torch_sdpa"]
        pair2 = cfg["bar_pair_ratios"]["pair2_jammi_fused_2_over_torch_sdpa_2"]
        self.assertAlmostEqual(pair1, 800.0 / 727.0, places=6)
        self.assertAlmostEqual(pair2, 750.0 / 700.0, places=6)
        self.assertFalse(cfg["bar_ratio_indeterminate"])
        self.assertAlmostEqual(cfg["bar_ratio_min_of_two_least_favourable_to_jammi"], min(pair1, pair2), places=6)
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertIn("PASS", table)
        self.assertEqual(rc, 0)

    def test_both_pairs_miss_the_bar_is_fail_using_the_min_of_the_two(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                _CLEAN_YES_DISPATCHES,
                jammi_overrides={"triplets_per_s": {"value": 600.0, "unit": "triplets/s"}},
            )  # pair1: 600/727 ~= 0.825
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=610.0, torch_tps=730.0)  # pair2 ~= 0.836
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertFalse(cfg["bar_ratio_indeterminate"])
        self.assertLess(cfg["bar_ratio_min_of_two_least_favourable_to_jammi"], 0.9)
        self.assertTrue(cfg["verdict"].startswith("FAIL"), cfg["verdict"])
        self.assertIn("FAIL", table)
        # record-don't-gate: an ordinary ratio-based FAIL never gates exit code.
        self.assertEqual(rc, 0)

    def test_straddling_pair_ratios_are_indeterminate_never_pass_or_fail(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                _CLEAN_YES_DISPATCHES,
                jammi_overrides={"triplets_per_s": {"value": 950.0, "unit": "triplets/s"}},
                torch_overrides={"triplets_per_s": {"value": 1000.0, "unit": "triplets/s"}},
            )  # pair1 = 0.95 (>= 0.9)
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=800.0, torch_tps=1000.0)  # pair2 = 0.80 (< 0.9)
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(cfg["bar_ratio_indeterminate"])
        self.assertEqual(cfg["verdict"][: len(ab_merge.FINETUNE_AB_VERDICT_INDETERMINATE)], "INDETERMINATE")
        self.assertFalse(cfg["verdict"].startswith("PASS"))
        self.assertFalse(str(cfg["verdict"]).startswith("FAIL"))
        self.assertFalse(str(cfg["verdict"]).startswith("INVALID"))
        self.assertIn("INDETERMINATE", table)
        self.assertIn("pair1(jammi-fused/torch-sdpa)=0.950", cfg["verdict"])
        self.assertIn("pair2(jammi-fused-2/torch-sdpa-2)=0.800", cfg["verdict"])
        # record-don't-gate: INDETERMINATE never gates exit code either.
        self.assertEqual(rc, 0)

    def test_wide_spread_same_side_of_the_bar_is_also_indeterminate(self):
        """Both pair ratios clear 0.9 (no straddle) but disagree by far more
        than the combined estimate's own distance from the bar -- still
        INDETERMINATE, never a confident PASS.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir,
                "b8-s128-d0",
                _CLEAN_YES_DISPATCHES,
                jammi_overrides={"triplets_per_s": {"value": 950.0, "unit": "triplets/s"}},
                torch_overrides={"triplets_per_s": {"value": 1000.0, "unit": "triplets/s"}},
            )  # pair1 = 0.95
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=2000.0, torch_tps=1000.0)  # pair2 = 2.0
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        pair1 = cfg["bar_pair_ratios"]["pair1_jammi_fused_over_torch_sdpa"]
        pair2 = cfg["bar_pair_ratios"]["pair2_jammi_fused_2_over_torch_sdpa_2"]
        # No straddle: both ratios are >= 0.9.
        self.assertGreaterEqual(pair1, 0.9)
        self.assertGreaterEqual(pair2, 0.9)
        self.assertTrue(cfg["bar_ratio_indeterminate"])
        self.assertTrue(cfg["verdict"].startswith("INDETERMINATE"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_fused_proof_failure_still_invalidates_even_with_a_clean_second_pair(self):
        """The INVALID carve-out (fused_proof) still takes precedence over
        INDETERMINATE/PASS/FAIL — checked on the FIRST run only, unchanged.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", {})  # all-(0,0) fused_proof -> False -> INVALID
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=750.0, torch_tps=700.0)
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX))
        self.assertEqual(rc, 1)

    def test_second_run_table_rows_appear_only_when_the_second_run_ran(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged, table_no_second = self.run_merge(raw_dir)
        self.assertNotIn("jammi-fused-2", table_no_second)
        self.assertNotIn("torch-sdpa-2", table_no_second)

        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            self.write_second_run(raw_dir, "b8-s128-d0", jammi_tps=750.0, torch_tps=700.0)
            rc, merged, table_with_second = self.run_merge(raw_dir)
        self.assertIn("jammi-fused-2", table_with_second)
        self.assertIn("torch-sdpa-2", table_with_second)

    def test_jammi_eager_row_surfaces_kernels_disabled_requested_and_fired(self):
        """A: the negative control's own provenance surfaced on the
        jammi-eager row of the printed table.
        """
        disable_keys = [
            "layer_norm_fused",
            "geglu_fused",
            "attention_block_flash",
            "attention_block_fused",
            "rope_fused",
            "softmax_last_dim_fused",
            "lora_linear_fused",
            "adamw_step_fused",
        ]
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(
                raw_dir,
                "b8-s128-d0",
                "jammi-eager",
                report=jammi_fs(
                    {},
                    attention_arm="eager",
                    kernels_disabled_requested=list(disable_keys),
                    kernels_disabled_fired=list(disable_keys),
                ),
            )
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES))
            write_leg(raw_dir, "b8-s128-d0", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa", report=torch_fs())
            rc, merged, table = self.run_merge(raw_dir)
        self.assertIn("kernels_disabled_requested=", table)
        self.assertIn("kernels_disabled_fired=", table)
        for key in disable_keys:
            self.assertIn(key, table)

    def test_second_run_fused_leg_with_undeclared_flash_decline_invalidates_the_config(self):
        """Identity-completeness: the bar ratio consumes BOTH pair legs, so
        `jammi-fused-2` must clear `fused_proof` exactly like `jammi-fused`
        does. An UNDECLARED (`kernels_disabled_requested`/`_fired` both
        empty) `attention_block_flash_declined_dispatches > 0` on the
        SECOND run alone must refuse the whole config, never silently feed
        `pair2_ratio`/the bar ratio with no proof check at all.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # first run: clean
            write_leg(
                raw_dir,
                "b8-s128-d0",
                "jammi-fused-2",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, **flash_overrides(fused=0, declined=5)),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa-2", report=torch_fs())
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertFalse(cfg["jammi_fused_dispatch_proof_second_run"])
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX), cfg["verdict"])
        self.assertIn("second-run", cfg["verdict"])
        self.assertIn("jammi-fused-2", cfg["verdict"])
        self.assertIn("INVALID", table)
        self.assertEqual(rc, 1)

    def test_second_run_leg_premise_mismatch_invalidates_the_config(self):
        """Identity-completeness: `jammi-fused-2`/`torch-sdpa-2` must run
        under the SAME premise, exactly like `jammi-fused`/`torch-sdpa` —
        a mismatched `batch` on the second run alone must refuse the whole
        config, never silently feed a ratio computed off two different
        configurations.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # first run: clean, batch=8
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused-2", report=jammi_fs(_CLEAN_YES_DISPATCHES))  # batch=8
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa-2", report=torch_fs(batch=16))  # mismatched
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(cfg["leg_premise_violations_second_run"])
        self.assertTrue(any("batch" in v for v in cfg["leg_premise_violations_second_run"]))
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX), cfg["verdict"])
        # A batch mismatch confined to ONE second-run leg is
        # mathematically inseparable from ALSO tripping the cross-run
        # check on that same leg's own run-1/run-2 pair (a 4-cycle of
        # equality constraints — jammi run1/run2, torch run1/run2,
        # run1-same, run2-same — cannot have exactly one dirty edge), so
        # whichever override runs LAST (`cross-run`, in this module's own
        # ordering) determines the exact final string; both are checked
        # via the STRUCTURED field above, and the verdict is asserted only
        # to actually name a premise mismatch, not a specific one.
        self.assertIn("premise mismatch", cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_second_run_absent_never_triggers_the_second_run_carve_outs(self):
        """The single-run shape, restated for the second-run carve-outs
        specifically: no `-2` legs at all -> both second-run checks read
        `None` (not checked), never a spurious INVALID.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["jammi_fused_dispatch_proof_second_run"])
        self.assertIsNone(cfg["leg_premise_violations_second_run"])
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)


class AdversarialAuditFoldInTests(unittest.TestCase):
    """The A,B,B,A order-balanced bar-leg protocol's edge cases: a `None`
    pair ratio never crashes the merge, the two_run marker makes all four
    bar legs mandatory, and cross-run premise drift is refused.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        with open(os.path.join(out_dir, "finetune_ab_table.txt")) as fh:
            table = fh.read()
        return rc, merged, table

    # ---- bar_ratio_classification must never crash the merge -----------

    def test_first_run_torch_sdpa_oom_with_clean_second_run_never_crashes(self):
        """`pair1_ratio` (torch-sdpa OOM'd on the FIRST run) is `None`;
        `pair2_ratio` (a clean second run) is a real float. A bare
        `min(None, float)` would take down the ENTIRE merge, not just this
        one config's row. Proven
        here against a raw_dir with a SECOND, healthy config too, so a
        crash-turned-refusal (never a crash) is distinguished from "this
        one bad config poisoned every other row".
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "b8-s128-oom", "jammi-eager", report=jammi_fs({}))
            write_leg(raw_dir, "b8-s128-oom", "jammi-fused", report=jammi_fs(_CLEAN_YES_DISPATCHES))
            write_leg(raw_dir, "b8-s128-oom", "torch-eager", report=torch_fs(attn_implementation="eager"))
            write_leg(
                raw_dir, "b8-s128-oom", "torch-sdpa",
                exit_code=1, stderr="RuntimeError: CUDA error: out of memory",
            )
            write_second_run(raw_dir, "b8-s128-oom")  # clean second run

            write_ok_config(raw_dir, "b8-s128-healthy", _CLEAN_YES_DISPATCHES)
            write_second_run(raw_dir, "b8-s128-healthy")

            rc, merged, table = self.run_merge(raw_dir)

        oom_cfg = merged["configs"]["b8-s128-oom"]
        self.assertEqual(oom_cfg["legs"]["torch-sdpa"]["outcome"], "OOM")
        # Config-level refusal (a well-defined, non-crashing verdict),
        # never a Python exception surfacing all the way to main().
        self.assertIn("torch-sdpa itself did not fit", oom_cfg["verdict"])
        self.assertFalse(oom_cfg["verdict"].startswith("PASS"))

        # The OTHER config in the SAME raw_dir must be entirely unaffected
        # -- a raise here would be a WHOLE-MERGE crash, not merely a bad
        # row.
        healthy_cfg = merged["configs"]["b8-s128-healthy"]
        self.assertTrue(healthy_cfg["verdict"].startswith("PASS"), healthy_cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_bar_ratio_classification_never_raises_for_any_none_combination(self):
        """Direct unit coverage of the function itself, every combination
        of `None`s explicitly."""
        self.assertEqual(ab_merge.bar_ratio_classification(None, None, 0.9), (None, False, None))
        self.assertEqual(ab_merge.bar_ratio_classification(None, 1.0, 0.9), (1.0, False, None))
        self.assertEqual(ab_merge.bar_ratio_classification(1.0, None, 0.9), (1.0, False, None))
        bar, indeterminate, detail = ab_merge.bar_ratio_classification(1.0, 1.0, 0.9)
        self.assertEqual(bar, 1.0)
        self.assertFalse(indeterminate)

    # ---- the two_run marker makes all four bar legs mandatory ----------

    def test_two_run_marker_present_missing_second_run_leg_is_invalid(self):
        """The marker promises all four bar legs; a genuinely MISSING
        second-run leg (never attempted at all, not merely OOM/FAIL) is
        an INCOMPLETE sweep -- INVALID, with a named reason -- never
        silently degraded to the single-pair estimator the way an absent
        MARKER legitimately is.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_two_run_marker(raw_dir)
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            # No write_second_run() call at all -- both -2 legs MISSING.
            rc, merged, table = self.run_merge(raw_dir)
        self.assertTrue(merged["two_run_protocol"])
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNotNone(cfg["two_run_missing_leg_reason"])
        self.assertIn("MISSING", cfg["two_run_missing_leg_reason"])
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX), cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_two_run_marker_present_second_run_jammi_fused_oom_never_silently_passes(self):
        """`jammi-fused-2` OOM's (a REAL,
        attempted measurement outcome, not MISSING) under the two_run
        marker -- must FAIL (OOM where torch fits), never silently
        degrade to the single-pair PASS the first run's own clean ratio
        would otherwise have produced.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_two_run_marker(raw_dir)
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # run1: clean, would PASS alone
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused-2",
                exit_code=1, stderr="RuntimeError: CUDA error: out of memory",
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa-2", report=torch_fs())
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["bar_second_run_legs"]["jammi-fused-2"]["outcome"], "OOM")
        self.assertFalse(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertIn("FAIL", cfg["verdict"])
        self.assertIn("jammi-fused-2", cfg["verdict"])
        self.assertIn("OOM", cfg["verdict"])
        # record-don't-gate: an ordinary OOM'd-where-torch-fits FAIL never
        # gates exit code, same as the primary-run carve-out.
        self.assertEqual(rc, 0)

    def test_legacy_raw_dir_without_the_marker_regresses_to_single_run_mode(self):
        """No marker at all (a hand-built `raw_dir`) -- `two_run_protocol`
        reads `False`, and a MISSING second run degrades to the
        single-pair estimator, never an INVALID.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged, table = self.run_merge(raw_dir)
        self.assertFalse(merged["two_run_protocol"])
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["two_run_missing_leg_reason"])
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)

    # ---- cross-RUN premise (jammi-fused vs jammi-fused-2, etc.) --------

    def test_cross_run_seed_and_seq_mismatch_invalidates_the_config(self):
        """Run 1 at seed=42/seq=128 (the
        fixtures' own defaults), run 2 at seed=7/seq=1024 -- internally
        CONSISTENT on each side (so neither SAME-run premise check fires
        at all), but the seed/seq drifted ACROSS the two runs, which only
        the cross-run check catches.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # run1: seed=42, seq=128
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused-2",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, seed=7, seq=1024),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa-2", report=torch_fs(seed=7, seq=1024))
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        # Neither same-run check fires -- both are CHECKED and CLEAN
        # (an empty list, not None -- None would mean "not checked", the
        # SAME sentinel-vs-empty-list distinction `leg_premise_violations`
        # itself documents), isolating the cross-run signal.
        self.assertEqual(cfg["leg_premise_violations"], [])
        self.assertEqual(cfg["leg_premise_violations_second_run"], [])
        self.assertTrue(cfg["leg_premise_violations_cross_run"])
        self.assertTrue(any("seed" in v for v in cfg["leg_premise_violations_cross_run"]))
        self.assertTrue(any("seq" in v for v in cfg["leg_premise_violations_cross_run"]))
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX), cfg["verdict"])
        self.assertIn("cross-run leg premise mismatch", cfg["verdict"])
        self.assertEqual(rc, 1)

    def test_cross_run_premise_absent_when_second_run_absent(self):
        """The single-run shape: no second run at all -> the cross-run
        check has nothing to compare, `None`, never a spurious INVALID.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["leg_premise_violations_cross_run"])
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)

    # ---- {leg:<14} column separator -------------------------------------

    def test_second_run_row_has_a_separator_after_the_leg_name(self):
        """`jammi-fused-2`/`torch-sdpa-2` (13/12 characters) must never
        run directly into the `outcome` column with zero separating
        whitespace.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            write_second_run(raw_dir, "b8-s128-d0")
            rc, merged, table = self.run_merge(raw_dir)
        for line in table.splitlines():
            if line.startswith("b8-s128-d0") and ("jammi-fused-2" in line or "torch-sdpa-2" in line):
                # The leg name must be followed by at least one space
                # before the outcome column starts.
                self.assertRegex(
                    line, r"(jammi-fused-2|torch-sdpa-2)\s+(OK|FAIL|OOM|MISSING|DRY_RUN)",
                    f"no separator after the leg name in row: {line!r}",
                )


class TwoRunModeMissingThroughputRefusalTests(unittest.TestCase):
    """No silent single-pair PASS under the marker: under `two_run_mode`, an
    `OK`-outcome leg whose own report
    still carries a falsy/missing `triplets_per_s` must refuse the WHOLE
    config, never silently hand the verdict back to the OTHER (still
    valid) pair. Both directions are probed.
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        with open(os.path.join(out_dir, "finetune_ab_table.txt")) as fh:
            table = fh.read()
        return rc, merged, table

    def test_marker_present_zero_tps_on_first_run_torch_sdpa_refuses(self):
        """`torch-sdpa` (first run) reads `OK` but `triplets_per_s ==
        0.0` -- `ratio` (pair 1) is `None`; `pair2_ratio` (a clean second
        run) is a real float. `bar_ratio_classification` alone would hand
        back `pair2_ratio` as `bar_ratio`, and the config would silently
        PASS off exactly one of the two pairs the marker promised both of.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_two_run_marker(raw_dir)
            write_ok_config(
                raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES,
                torch_overrides={"triplets_per_s": {"value": 0.0, "unit": "triplets/s"}},
            )
            write_second_run(raw_dir, "b8-s128-d0")  # clean second run
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["ratio_jammi_fused_over_torch_sdpa"])
        self.assertIsNotNone(cfg["bar_pair_ratios"]["pair2_jammi_fused_2_over_torch_sdpa_2"])
        self.assertFalse(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertIn("no ratio: triplets_per_s missing on an OK leg", cfg["verdict"])
        self.assertIn("no ratio: triplets_per_s missing on an OK leg", table)
        self.assertEqual(rc, 0)  # record-don't-gate: this FAIL never gates exit code.

    def test_marker_present_zero_tps_on_second_run_torch_sdpa_2_refuses(self):
        """The MIRROR probe: `torch-sdpa` (first run) is clean, but
        `torch-sdpa-2` (second run) reads `OK` with `triplets_per_s ==
        0.0` -- `pair2_ratio` is `None`, `ratio` (pair 1) is a real float.
        `bar_ratio_classification` alone would hand back `ratio` as
        `bar_ratio` and the config would silently PASS off pair 1 alone.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_two_run_marker(raw_dir)
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # clean first run
            write_second_run(
                raw_dir, "b8-s128-d0",
                torch_overrides={"triplets_per_s": {"value": 0.0, "unit": "triplets/s"}},
            )
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNotNone(cfg["ratio_jammi_fused_over_torch_sdpa"])
        self.assertIsNone(cfg["bar_pair_ratios"]["pair2_jammi_fused_2_over_torch_sdpa_2"])
        self.assertFalse(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertIn("no ratio: triplets_per_s missing on an OK leg", cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_legacy_no_marker_mode_keeps_the_single_pair_fallback_unchanged(self):
        """WITHOUT the marker, a zero-tps first-run torch-sdpa with no second
        run at all keeps the single-pair "no ratio" classification -- the
        widened condition applies only under `two_run_mode`.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(
                raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES,
                torch_overrides={"triplets_per_s": {"value": 0.0, "unit": "triplets/s"}},
            )
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertFalse(merged["two_run_protocol"])
        self.assertIn("no ratio: triplets_per_s missing on an OK leg", cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_marker_present_both_pairs_clean_still_passes(self):
        """Positive control: the widened condition must not false-positive
        when both pairs genuinely produced a usable ratio.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_two_run_marker(raw_dir)
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            write_second_run(raw_dir, "b8-s128-d0")
            rc, merged, table = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)


class CrossRunPremiseTriStateTests(unittest.TestCase):
    """`leg_premise_violations_cross_run` must be able to state a POSITIVE
    "checked and clean" fact (`[]`), never collapse "checked, clean" and
    "never checked" onto the SAME `None` value
    (`ci/artifacts/finetune-ab-runs/2026-08-30-full-sweep-acce7b3d-a100-pcie/
    finetune_ab_report.json` reads `null` there on every config; see that
    artifact's own README).
    """

    def run_merge(self, raw_dir):
        out_dir = tempfile.mkdtemp()
        rc = ab_merge.main([raw_dir, out_dir, "20", "5", "0.9"])
        with open(os.path.join(out_dir, "finetune_ab_report.json")) as fh:
            merged = json.load(fh)
        return rc, merged

    def test_all_ok_two_run_config_reads_checked_clean_not_none(self):
        """Every bar leg OK, both runs, no drift -- the cross-run check
        RAN (both sub-comparisons had two OK legs to compare) and found
        nothing, so the field must read `[]` (checked, clean), never
        `None` (which would mean "never checked" -- FALSE here).
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            write_second_run(raw_dir, "b8-s128-d0")
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations_cross_run"], [])
        self.assertIsNotNone(cfg["leg_premise_violations_cross_run"])
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_legacy_single_run_config_still_reads_none(self):
        """No second run at all -- neither sub-comparison ever had two OK
        legs to compare, so the field must stay `None` (genuinely
        unchecked), never collapse to `[]` just because nothing went
        wrong elsewhere.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertIsNone(cfg["leg_premise_violations_cross_run"])
        self.assertTrue(cfg["verdict"].startswith("PASS"), cfg["verdict"])
        self.assertEqual(rc, 0)

    def test_one_side_checked_clean_other_side_unavailable_still_reads_checked(self):
        """Only the jammi-vs-jammi-2 sub-comparison has two OK legs (e.g.
        torch-sdpa-2 OOM'd) -- the field must still flip to `[]` (checked
        via that ONE sub-comparison), not stay `None` just because the
        OTHER sub-comparison never ran.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)
            write_leg(raw_dir, "b8-s128-d0", "jammi-fused-2", report=jammi_fs(_CLEAN_YES_DISPATCHES))
            write_leg(
                raw_dir, "b8-s128-d0", "torch-sdpa-2",
                exit_code=1, stderr="RuntimeError: CUDA error: out of memory",
            )
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertEqual(cfg["leg_premise_violations_cross_run"], [])

    def test_a_real_cross_run_violation_still_reports_the_drift(self):
        """Non-vacuity: the tri-state must not weaken the VIOLATION-reporting
        arm -- a genuine cross-run drift
        still populates the list with the actual violation.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_ok_config(raw_dir, "b8-s128-d0", _CLEAN_YES_DISPATCHES)  # run1: seed=42, seq=128
            write_leg(
                raw_dir, "b8-s128-d0", "jammi-fused-2",
                report=jammi_fs(_CLEAN_YES_DISPATCHES, seed=7, seq=1024),
            )
            write_leg(raw_dir, "b8-s128-d0", "torch-sdpa-2", report=torch_fs(seed=7, seq=1024))
            rc, merged = self.run_merge(raw_dir)
        cfg = merged["configs"]["b8-s128-d0"]
        self.assertTrue(cfg["leg_premise_violations_cross_run"])
        self.assertTrue(any("seed" in v for v in cfg["leg_premise_violations_cross_run"]))
        self.assertTrue(cfg["verdict"].startswith(ab_merge.FINETUNE_AB_VERDICT_INVALID_PREFIX), cfg["verdict"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
