#!/usr/bin/env python3
"""Hermetic tests for `profile_421_merge.py` (issue #421 P1-b; CONTRACT
`scratchpad/contract-421-profile.md` v2.3 §D4 items 1, 4 and 5).

Everything here is SYNTHETIC tier JSON written into a tempdir — no GPU, no
pod, no `nsys`, no committed baseline, no network. That is deliberate: the
merge step's whole job is to REFUSE, and a refusal path can only be trusted
if the inputs that trigger it can be constructed on demand. Each test builds
the minimal `$OUT_DIR` shape the driver emits, perturbs exactly ONE thing,
and asserts the verdict flips (or, for the sign-convention and text-front-end
cases, that it deliberately does NOT).

Numbers, not shapes: the per-step decomposition is asserted against an
INDEPENDENT oracle computed here (`_oracle_per_step`) — numpy-first with a
pure-Python fallback, this repo's standing "numpy-first oracle" convention
(`compare_grad_oracle.py`'s own module doc) — never against a value copied
out of a previous run of the tool under test.

One test drives the REAL `profile_421_legs.sh` in its `PROFILE_421_P2_BF16=1`
dry-run mode and feeds its actual output to the real merge script, so the
two halves of the P2 pre-flight are proven to fit as shipped, not as
imagined.

Run: `python3 ci/scripts/perf/test_profile_421_merge.py`
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
MERGE = PERF_DIR / "profile_421_merge.py"
LEGS_SH = PERF_DIR / "profile_421_legs.sh"

sys.path.insert(0, str(PERF_DIR))
import profile_421_merge as merge  # noqa: E402

try:  # numpy-first oracle; the pure-Python fallback below is exact for these
    import numpy as _np  # type: ignore

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised on numpy-less runners
    _np = None
    HAVE_NUMPY = False


def _oracle_per_step(wall_n, wall_m, front_n, front_m, busy_us, steps_n, steps_m):
    """`(wall, front, busy, residual)` per step, computed INDEPENDENTLY of
    the module under test.

    numpy-first (float64 array arithmetic) with a pure-Python fallback, so
    the oracle is a genuinely separate implementation on a numpy-bearing
    runner and still runs at all on one without it. `front_*` of `None`
    means the TEXT arm, where the front end is 0 by construction.
    """
    denom = steps_m - steps_n
    if HAVE_NUMPY:
        walls = _np.array([wall_n, wall_m], dtype=_np.float64)
        wall = float((walls[1] - walls[0]) / denom)
        if front_n is None or front_m is None:
            front = 0.0
        else:
            fronts = _np.array([front_n, front_m], dtype=_np.float64)
            front = float((fronts[1] - fronts[0]) / denom)
        busy = float(_np.float64(busy_us) / _np.float64(1e6))
    else:
        wall = (wall_m - wall_n) / denom
        front = 0.0 if (front_n is None or front_m is None) else (front_m - front_n) / denom
        busy = busy_us / 1e6
    return wall, front, busy, wall - front - busy


# A CLIP-tower-shaped witnessed census: `quick_gelu` has no fused seam, so
# `gelu_seam_calls_per_forward` is legitimately 0 and the leg must then read
# 0/0 dispatches -- a zero census is a CHECKABLE claim here, not a skip.
CENSUS_CLIP = {"lora_sites_wrapped": 48, "layer_norms": 25, "gelu_seam_calls_per_forward": 0}
# An HTSAT-shaped one: sum(depths) MLP dispatches plus the projection head.
CENSUS_HTSAT = {"lora_sites_wrapped": 77, "layer_norms": 30, "gelu_seam_calls_per_forward": 9}


def make_tier(
    *,
    steps: int,
    wall: float,
    front: float | None,
    census: dict | None,
    disabled: tuple[str, ...] = (),
    fused_overrides: dict | None = None,
    eager_overrides: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """A synthetic `tiers.finetune_run` that SATISFIES the positive-proof
    equation by construction: for every key, `fused = census x steps` and
    `eager = 0` on a fused arm, and the other way round for a key named in
    `disabled`. A test that wants a violation overrides exactly one counter.
    """
    tier: dict = {
        "steps_measured": steps,
        # The pinned measurement convention: `batches == steps_measured`
        # holds only here. Every leg the driver runs pins both.
        "grad_accum": 1,
        "epochs": 1,
        "train_run_wall_s": wall,
        "media_front_end_wall_s": front,
        "kernels_disabled_expected": sorted(disabled),
    }
    if census is not None:
        tier["fusible_site_census"] = dict(census)
    for key, (fused_field, eager_field, census_field) in merge.KEY_FIELDS.items():
        total = (census or {}).get(census_field, 0) * steps
        if key in disabled:
            tier[fused_field], tier[eager_field] = 0, total
        else:
            tier[fused_field], tier[eager_field] = total, 0
    for field, value in (fused_overrides or {}).items():
        tier[field] = value
    for field, value in (eager_overrides or {}).items():
        tier[field] = value
    tier.update(extra or {})
    return tier


def write_leg(
    out_dir: Path,
    leg_id: str,
    *,
    tower: str = "clip-text",
    task: str = "text_embedding",
    dtype: str = "f32",
    disabled: tuple[str, ...] = (),
    tier_n: dict,
    tier_m: dict,
    busy_us: float = 1200.0,
    status: str = "ok",
    reason: str = "",
    census_extra: dict | None = None,
    omit: tuple[str, ...] = (),
) -> Path:
    leg_dir = out_dir / leg_id
    leg_dir.mkdir(parents=True)
    manifest = {
        "leg_id": leg_id,
        "tower": tower,
        "task": task,
        "dtype": dtype,
        "kernels_disabled": list(disabled),
        "status": status,
        "reason": reason,
    }
    (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if "run_n" not in omit:
        (leg_dir / "run_n.json").write_text(
            json.dumps({"tiers": {"finetune_run": tier_n}}), encoding="utf-8"
        )
    if "run_m" not in omit:
        (leg_dir / "run_m.json").write_text(
            json.dumps({"tiers": {"finetune_run": tier_m}}), encoding="utf-8"
        )
    if "census" not in omit:
        census_file = {"gpu_kernel_us_per_step": busy_us}
        census_file.update(census_extra or {})
        (leg_dir / "census.json").write_text(json.dumps(census_file), encoding="utf-8")
    return leg_dir


def a_leg_pair(census=CENSUS_CLIP, *, front=None, wall_n=1.0, wall_m=6.0, steps=(100, 600)):
    steps_n, steps_m = steps
    front_n = None if front is None else front[0]
    front_m = None if front is None else front[1]
    return (
        make_tier(steps=steps_n, wall=wall_n, front=front_n, census=census),
        make_tier(steps=steps_m, wall=wall_m, front=front_m, census=census),
    )


def run_merge(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MERGE), *args], capture_output=True, text=True, timeout=300
    )


def merged(out_dir: Path, *extra: str) -> dict:
    result = run_merge("--legs-dir", str(out_dir), *extra)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return json.loads(result.stdout)


def only_leg(report: dict) -> dict:
    legs = report["legs"]
    assert len(legs) == 1, legs
    return legs[0]


def reasons_text(row: dict) -> str:
    return "\n".join(row["reasons"])


class PositiveProofTests(unittest.TestCase):
    def test_a_clean_a_leg_is_valid_and_reports_the_witnessed_equation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            self.assertEqual(row["arm"], "A")
            proof = row["positive_proof"]["run_m"]
            # The expected total is the WITNESSED census times the measured
            # step count -- recomputed here rather than transcribed.
            self.assertEqual(proof["lora_linear_fused"]["calls_per_forward"], 48)
            self.assertEqual(proof["lora_linear_fused"]["expected_total"], 48 * 600)
            self.assertEqual(proof["lora_linear_fused"]["fused"], 48 * 600)
            self.assertEqual(proof["layer_norm_fused"]["expected_total"], 25 * 600)
            self.assertTrue(all(entry["equation_ok"] for entry in proof.values()), proof)

    def test_an_off_by_one_dispatch_count_invalidates_the_leg(self):
        """The equation is the whole point: one extra LayerNorm dispatch
        means the census and the counters disagree about what ran."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["ln_fused_dispatches"] += 1
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("layer_norm_fused positive proof failed", reasons_text(row))
            self.assertFalse(row["positive_proof"]["run_m"]["layer_norm_fused"]["equation_ok"])

    def test_an_a_leg_that_never_dispatched_fused_is_invalid_even_when_the_equation_holds(self):
        """`fused == 0, eager == census x steps` satisfies the SUM equation
        exactly. Only the one-sided A-leg requirement catches it."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            for tier, steps in ((tier_n, 100), (tier_m, 600)):
                tier["lora_linear_fused_dispatches"] = 0
                tier["lora_linear_eager_dispatches"] = 48 * steps
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("lora_linear_fused_dispatches=0 on a leg that disabled nothing",
                          reasons_text(row))
            # The SUM equation itself is untouched -- proving the two checks
            # are genuinely independent.
            self.assertTrue(row["positive_proof"]["run_m"]["lora_linear_fused"]["equation_ok"])

    def test_a_zero_census_key_still_constrains_the_counters(self):
        """CLIP's `quick_gelu` has no seam, so the census is 0 -- and the
        leg must then read 0/0. A zero census is a claim, not a skip."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            self.assertEqual(tier_m["gelu_fused_dispatches"], 0)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            self.assertEqual(only_leg(merged(out))["verdict"], "VALID")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["gelu_fused_dispatches"] = 1
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("gelu_erf_fused positive proof failed", reasons_text(row))

    def test_an_htsat_leg_checks_the_gelu_seam_count_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair(CENSUS_HTSAT, front=(0.4, 2.4))
            write_leg(out, "htsat-A1", tower="htsat", task="audio_embedding",
                      tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            self.assertEqual(
                row["positive_proof"]["run_m"]["gelu_erf_fused"]["expected_total"], 9 * 600
            )

    def test_a_d_leg_must_show_zero_fused_for_every_disabled_key(self):
        disabled = ("layer_norm_fused", "lora_linear_fused")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=1.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            tier_m = make_tier(steps=600, wall=6.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            write_leg(out, "clip-text-D1", disabled=disabled, tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            self.assertEqual(row["arm"], "D")
            self.assertEqual(row["positive_proof"]["run_m"]["layer_norm_fused"]["eager"], 25 * 600)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=1.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            tier_m = make_tier(steps=600, wall=6.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            # One fused dispatch leaked through, with `eager` reduced to keep
            # the SUM equation satisfied -- so ONLY the disabled-key check
            # can catch it.
            tier_m["ln_fused_dispatches"] = 1
            tier_m["ln_eager_dispatches"] = 25 * 600 - 1
            write_leg(out, "clip-text-D1", disabled=disabled, tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("dispatched the fused kernel it claimed to have disabled",
                          reasons_text(row))
            self.assertTrue(row["positive_proof"]["run_m"]["layer_norm_fused"]["equation_ok"])

    def test_a_d2_leg_does_not_require_ln_to_be_eager(self):
        """D2 disables `lora_linear_fused` only; `layer_norm_fused` stays
        fused there by design, and the merge must not invent a requirement
        the contract never pre-registered."""
        disabled = ("lora_linear_fused",)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=1.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            tier_m = make_tier(steps=600, wall=6.0, front=None, census=CENSUS_CLIP,
                               disabled=disabled)
            write_leg(out, "clip-text-D2", disabled=disabled, tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            self.assertEqual(row["positive_proof"]["run_m"]["layer_norm_fused"]["fused"], 25 * 600)

    def test_a_multi_epoch_leg_is_refused_by_name_not_as_an_equation_failure(self):
        """`steps_measured` is the equation's `batches` term only at
        `--epochs 1`: this tier sums each resume-chained leg's own
        `global_step`, which a resumed leg carries forward, so a 2-epoch run
        over-counts training forwards. Reporting THAT as "the counters
        disagree with the census" would blame the kernels for a convention
        mismatch."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["epochs"] = 2
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            text = reasons_text(row)
            self.assertIn("epochs=2", text)
            self.assertIn("over-counts training forwards", text)
            self.assertNotIn("positive proof failed", text)
            # The equation was not even attempted for that run.
            self.assertNotIn("run_m", row["positive_proof"] or {})

    def test_a_grad_accum_above_one_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["grad_accum"] = 4
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            text = reasons_text(row)
            self.assertIn("grad_accum=4", text)
            self.assertNotIn("positive proof failed", text)

    def test_a_missing_site_census_invalidates_but_still_reports_the_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            del tier_n["fusible_site_census"]
            del tier_m["fusible_site_census"]
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("refuses to guess it", reasons_text(row))
            # The decomposition does not depend on the census struct, so a
            # missing census must not blank the table's timings too.
            self.assertIsNotNone(row["per_step"])
            self.assertGreater(row["per_step"]["wall_s_per_step"], 0.0)

    def test_a_pair_whose_two_runs_built_different_models_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=1.0, front=None, census=CENSUS_CLIP)
            tier_m = make_tier(steps=600, wall=6.0, front=None, census=CENSUS_HTSAT)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("did not build the same model", reasons_text(row))

    def test_a_report_that_contradicts_the_manifests_arm_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["kernels_disabled_expected"] = ["lora_linear_fused"]
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("does not match the manifest's declared arm", reasons_text(row))

    def test_the_drivers_own_invalid_verdict_is_honoured_not_relitigated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m,
                      status="invalid", reason="kernel_census.py refused (exit 4)")
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("kernel_census.py refused", reasons_text(row))

    def test_a_missing_run_report_is_a_reason_never_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m, omit=("run_m",))
            result = run_merge("--legs-dir", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            row = only_leg(json.loads(result.stdout))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("run_m.json is missing", reasons_text(row))
            self.assertNotIn("Traceback", result.stderr)


class PerStepDecompositionTests(unittest.TestCase):
    def test_the_decomposition_matches_an_independent_oracle(self):
        wall_n, wall_m = 12.5, 62.5
        front_n, front_m = 4.0, 20.0
        busy_us = 55_000.0
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=wall_n, front=front_n, census=CENSUS_HTSAT)
            tier_m = make_tier(steps=600, wall=wall_m, front=front_m, census=CENSUS_HTSAT)
            write_leg(out, "htsat-A1", tower="htsat", task="audio_embedding",
                      tier_n=tier_n, tier_m=tier_m, busy_us=busy_us)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            wall, front, busy, residual = _oracle_per_step(
                wall_n, wall_m, front_n, front_m, busy_us, 100, 600
            )
            per_step = row["per_step"]
            self.assertAlmostEqual(per_step["wall_s_per_step"], wall, places=12)
            self.assertAlmostEqual(per_step["front_s_per_step"], front, places=12)
            self.assertAlmostEqual(per_step["busy_s_per_step"], busy, places=12)
            self.assertAlmostEqual(per_step["residual_s_per_step"], residual, places=12)
            self.assertIsNone(per_step["overlap_s_per_step"])
            self.assertFalse(per_step["front_end_inside_residual"])
            self.assertAlmostEqual(per_step["front_share_of_wall"], front / wall, places=12)
            self.assertAlmostEqual(per_step["busy_share_of_wall"], busy / wall, places=12)

    def test_a_negative_residual_is_reported_as_overlap_and_never_invalidates(self):
        """`front` is CPU wall and `busy` is GPU device time on a different
        timeline, so `front + busy > wall` MEASURES overlap. Invalidating
        the leg for it would throw away a real reading."""
        wall_n, wall_m = 12.5, 62.5  # 0.1 s/step
        front_n, front_m = 4.0, 40.0  # 0.072 s/step
        busy_us = 60_000.0  # 0.06 s/step -> residual = -0.032
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=wall_n, front=front_n, census=CENSUS_HTSAT)
            tier_m = make_tier(steps=600, wall=wall_m, front=front_m, census=CENSUS_HTSAT)
            write_leg(out, "htsat-A1", tower="htsat", task="audio_embedding",
                      tier_n=tier_n, tier_m=tier_m, busy_us=busy_us)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            _w, _f, _b, residual = _oracle_per_step(
                wall_n, wall_m, front_n, front_m, busy_us, 100, 600
            )
            self.assertLess(residual, 0.0)
            self.assertAlmostEqual(row["per_step"]["residual_s_per_step"], residual, places=12)
            self.assertAlmostEqual(row["per_step"]["overlap_s_per_step"], -residual, places=12)

    def test_a_text_legs_null_front_end_is_zero_and_says_where_it_went(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair(wall_n=12.5, wall_m=62.5)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m, busy_us=55_000.0)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "VALID", reasons_text(row))
            per_step = row["per_step"]
            self.assertEqual(per_step["front_s_per_step"], 0.0)
            self.assertTrue(per_step["front_end_inside_residual"])
            wall, front, busy, residual = _oracle_per_step(
                12.5, 62.5, None, None, 55_000.0, 100, 600
            )
            self.assertAlmostEqual(per_step["residual_s_per_step"], residual, places=12)
            self.assertEqual(front, 0.0)

    def test_a_pair_that_mixes_a_text_run_with_a_media_run_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=12.5, front=None, census=CENSUS_HTSAT)
            tier_m = make_tier(steps=600, wall=62.5, front=20.0, census=CENSUS_HTSAT)
            write_leg(out, "htsat-A1", tower="htsat", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("a text run was paired with a media run", reasons_text(row))

    def test_a_front_end_that_got_cheaper_with_more_rows_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n = make_tier(steps=100, wall=12.5, front=20.0, census=CENSUS_HTSAT)
            tier_m = make_tier(steps=600, wall=62.5, front=4.0, census=CENSUS_HTSAT)
            write_leg(out, "htsat-A1", tower="htsat", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("front end got CHEAPER", reasons_text(row))

    def test_a_foreign_census_file_is_caught_by_the_wall_cross_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair(wall_n=12.5, wall_m=62.5)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m,
                      census_extra={"wall_s_per_step": 0.25})
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("was not built from this pair of reports", reasons_text(row))

    def test_a_matching_census_wall_passes_the_cross_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair(wall_n=12.5, wall_m=62.5)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m,
                      census_extra={"wall_s_per_step": (62.5 - 12.5) / 500})
            self.assertEqual(only_leg(merged(out))["verdict"], "VALID")

    def test_a_pair_whose_wall_did_not_grow_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair(wall_n=62.5, wall_m=12.5)
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            row = only_leg(merged(out))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("outside the differencing domain", reasons_text(row))


class NonFiniteControlTests(unittest.TestCase):
    """A negative control has to fail on EVERY way the bad path can fail --
    including the non-finite one. `NaN > 0`, `NaN < 0` and `NaN == 0` are ALL
    `False`, so a diverged run is invisible to any ordinary comparison; the
    reader has to test `math.isfinite` at the point of reading."""

    def test_the_comparison_that_would_have_missed_it(self):
        nan = float("nan")
        self.assertFalse(nan > 0.0)
        self.assertFalse(nan < 0.0)
        self.assertFalse(nan == 0.0)
        self.assertFalse(math.isfinite(nan))
        self.assertFalse(math.isfinite(float("inf")))

    def _leg_with(self, tmp, field, value, *, on_census=False):
        out = Path(tmp)
        tier_n, tier_m = a_leg_pair(CENSUS_HTSAT, front=(4.0, 20.0), wall_n=12.5, wall_m=62.5)
        extra = {}
        if on_census:
            extra["census_extra"] = {field: value}
        else:
            tier_m[field] = value
        write_leg(out, "htsat-A1", tower="htsat", tier_n=tier_n, tier_m=tier_m, **extra)
        return only_leg(merged(out))

    def test_a_nan_wall_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "train_run_wall_s", float("nan"))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("train_run_wall_s is not finite", reasons_text(row))
            self.assertIsNone(row["per_step"])

    def test_an_infinite_wall_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "train_run_wall_s", float("inf"))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("train_run_wall_s is not finite", reasons_text(row))

    def test_a_nan_front_end_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "media_front_end_wall_s", float("nan"))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("media_front_end_wall_s is not finite", reasons_text(row))

    def test_a_nan_gpu_busy_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "gpu_kernel_us_per_step", float("nan"), on_census=True)
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("gpu_kernel_us_per_step is not finite", reasons_text(row))

    def test_a_float_dispatch_counter_is_refused(self):
        """A count that arrived as a float is not the field this reader
        thinks it is -- `24000.0 == 24000` would otherwise pass silently."""
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "ln_fused_dispatches", float(30 * 600))
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("ln_fused_dispatches is not an integer", reasons_text(row))

    def test_a_negative_dispatch_counter_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._leg_with(tmp, "ln_fused_dispatches", -1)
            self.assertEqual(row["verdict"], "INVALID")
            self.assertIn("ln_fused_dispatches is negative", reasons_text(row))


class P2PreflightTests(unittest.TestCase):
    def _write_p2(self, root: Path, tower: str, *, tier_overrides=None, manifest_overrides=None):
        tower_dir = root / tower
        tower_dir.mkdir(parents=True)
        manifest = {
            "mode": "p2-bf16-preflight",
            "tower": tower,
            "exit": 0,
            "status": "ok",
            "reason": "",
        }
        manifest.update(manifest_overrides or {})
        tier = {
            "backbone_dtype": "bf16",
            "lora_init": "gaussian",
            "final_loss_diagnostic": 0.581,
            "train_probe_series": [0.60, 0.55],
            "lora_linear_fused_dispatches": 96,
            "ln_fused_dispatches": 50,
        }
        tier.update(tier_overrides or {})
        (tower_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tower_dir / "run.json").write_text(
            json.dumps({"tiers": {"finetune_run": tier}}), encoding="utf-8"
        )
        return tower_dir

    def _verdict(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_p2(root, "clip-text", **kwargs)
            result = run_merge("--p2-dir", str(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(result.stdout)["p2_bf16"]
            self.assertEqual(len(rows), 1, rows)
            return rows[0]

    def test_a_clean_preflight_passes(self):
        row = self._verdict()
        self.assertEqual(row["verdict"], "PASS", row["reasons"])
        self.assertEqual(row["train_probe_series_head"], [0.60, 0.55])

    def test_a_nonzero_exit_fails(self):
        row = self._verdict(manifest_overrides={"exit": 101, "status": "invalid",
                                                "reason": "dtype error"})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("exited 101", "\n".join(row["reasons"]))

    def test_an_f32_run_cannot_pass_as_the_bf16_preflight(self):
        row = self._verdict(tier_overrides={"backbone_dtype": "f32"})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("but the P2 pre-flight is bf16", "\n".join(row["reasons"]))

    def test_a_zeros_b_run_cannot_pass_because_its_probe_check_would_be_vacuous(self):
        row = self._verdict(tier_overrides={"lora_init": "zeros_b"})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("would be vacuous", "\n".join(row["reasons"]))

    def test_an_unmoved_probe_series_fails(self):
        row = self._verdict(tier_overrides={"train_probe_series": [0.6, 0.6]})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("did not move the loss at step 1", "\n".join(row["reasons"]))

    def test_a_nan_final_loss_fails(self):
        row = self._verdict(tier_overrides={"final_loss_diagnostic": float("nan")})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("final_loss_diagnostic is not finite", "\n".join(row["reasons"]))

    def test_a_nan_probe_fails(self):
        row = self._verdict(tier_overrides={"train_probe_series": [float("nan"), 0.55]})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("train_probe_series[0] is not finite", "\n".join(row["reasons"]))

    def test_a_short_probe_series_fails(self):
        row = self._verdict(tier_overrides={"train_probe_series": [0.6]})
        self.assertEqual(row["verdict"], "FAIL")
        self.assertIn("at least 2 entries", "\n".join(row["reasons"]))

    def test_zero_fused_counters_fail(self):
        for field in ("lora_linear_fused_dispatches", "ln_fused_dispatches"):
            with self.subTest(field=field):
                row = self._verdict(tier_overrides={field: 0})
                self.assertEqual(row["verdict"], "FAIL")
                self.assertIn(f"{field}=0", "\n".join(row["reasons"]))


class EndToEndWithTheRealDriverTests(unittest.TestCase):
    def test_the_real_drivers_p2_dry_run_output_merges_and_passes(self):
        """The two halves as SHIPPED: the real `profile_421_legs.sh` in its
        `PROFILE_421_P2_BF16=1` dry-run mode writes the P2 tree, and the real
        merge script reads it. A hand-written fixture could not catch a
        disagreement about the tree's own shape."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["PROFILE_421_LEGS_DRY_RUN"] = "1"
            env["PROFILE_421_P2_BF16"] = "1"
            env["OUT_DIR"] = tmp
            env["NSYS_BIN"] = "/nonexistent/nsys-DRY-RUN-PLACEHOLDER"
            env["BENCH_BIN"] = "/nonexistent/jammi-bench-DRY-RUN-PLACEHOLDER"
            env.pop("JAMMI_KERNELS_DISABLE", None)
            driver = subprocess.run(
                ["bash", str(LEGS_SH)], env=env, capture_output=True, text=True, timeout=600
            )
            self.assertEqual(driver.returncode, 0,
                             f"stdout={driver.stdout}\nstderr={driver.stderr}")
            result = run_merge("--p2-dir", os.path.join(tmp, "p2-bf16"))
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(result.stdout)["p2_bf16"]
            self.assertEqual(
                sorted(row["tower"] for row in rows), ["clip-text", "clip-vision", "htsat"]
            )
            for row in rows:
                self.assertEqual(row["verdict"], "PASS", (row["tower"], row["reasons"]))
                self.assertEqual(row["backbone_dtype"], "bf16")
                self.assertEqual(row["lora_init"], "gaussian")


class CliTests(unittest.TestCase):
    def test_no_directory_argument_is_a_usage_error(self):
        result = run_merge()
        self.assertEqual(result.returncode, 2)
        self.assertIn("at least one of --legs-dir/--p2-dir", result.stderr)

    def test_a_nonexistent_directory_is_a_tool_failure(self):
        result = run_merge("--legs-dir", "/nonexistent/profile-421-DOES-NOT-EXIST")
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not a directory", result.stderr)

    def test_an_invalid_leg_is_loud_on_stderr_but_exits_zero(self):
        """An INVALID leg is a recorded OUTCOME of the experiment, not a
        failure of this tool -- but it is never silent."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            tier_m["ln_fused_dispatches"] += 1
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            result = run_merge("--legs-dir", str(out))
            self.assertEqual(result.returncode, 0)
            self.assertIn("::warning::leg clip-text-A1: INVALID", result.stderr)
            self.assertIn("0/1 legs VALID", result.stderr)

    def test_the_out_flag_writes_the_table_to_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            legs = out / "legs"
            legs.mkdir()
            tier_n, tier_m = a_leg_pair()
            write_leg(legs, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            target = out / "merged.json"
            result = run_merge("--legs-dir", str(legs), "--out", str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["legs_valid"], 1)
            self.assertEqual(report["tool"], "profile_421_merge")

    def test_the_table_is_keyed_by_tower_dtype_and_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for leg_id, tower, dtype in (
                ("clip-text-A1", "clip-text", "f32"),
                ("clip-text-A2", "clip-text", "bf16"),
                ("htsat-A1", "htsat", "f32"),
            ):
                census = CENSUS_HTSAT if tower == "htsat" else CENSUS_CLIP
                front = (4.0, 20.0) if tower == "htsat" else None
                tier_n, tier_m = a_leg_pair(census, front=front)
                write_leg(out, leg_id, tower=tower, dtype=dtype,
                          tier_n=tier_n, tier_m=tier_m)
            report = merged(out)
            keyed = {(row["tower"], row["dtype"], row["leg_id"]) for row in report["legs"]}
            self.assertEqual(
                keyed,
                {
                    ("clip-text", "f32", "clip-text-A1"),
                    ("clip-text", "bf16", "clip-text-A2"),
                    ("htsat", "f32", "htsat-A1"),
                },
            )
            self.assertEqual(report["summary"]["legs_valid"], 3)

    def test_the_p2_subtree_is_not_mistaken_for_a_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tier_n, tier_m = a_leg_pair()
            write_leg(out, "clip-text-A1", tier_n=tier_n, tier_m=tier_m)
            p2 = out / "p2-bf16"
            p2.mkdir()
            (p2 / "manifest.json").write_text("{}", encoding="utf-8")
            report = merged(out)
            self.assertEqual([row["leg_id"] for row in report["legs"]], ["clip-text-A1"])


if __name__ == "__main__":
    unittest.main()
