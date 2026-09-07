#!/usr/bin/env python3
"""The issue #421 tower-profile MERGE step: turn `profile_421_legs.sh`'s raw
`$OUT_DIR` into the per-`(tower, dtype, leg)` table the artifact producer
embeds — and, in doing so, apply every check the CONTRACT
(`scratchpad/contract-421-profile.md` v2.3, "Positive proof" + §D4 items 1,
4 and 5) says must hold before a leg's numbers are a datum at all.

Why the checks live HERE and not in the driver: the driver RUNS legs and
records what came back; a producer that also judged its own output would be
two implementations of one rule, and the one that mattered would be
whichever ran last. This module is the single reader, and it is tested
hermetically against SYNTHETIC tier JSON (`test_profile_421_merge.py`) — no
GPU, no pod, no committed baseline — so every branch below (including every
REFUSAL branch) is exercised in CI.

## What a leg must satisfy (all of it, or the leg is INVALID)

1. **The positive-proof equation, per key, per run** (§D4 item 1). The
   number of `calls` per forward is WITNESSED, never guessed: it is read off
   the run's own `tiers.finetune_run.fusible_site_census`, which
   `jammi-bench` derives from the encoder it actually built. For each of the
   three fusible keys the towers can admit:

   | admit key          | census field                 | tier counters                                             |
   |--------------------|------------------------------|-----------------------------------------------------------|
   | `lora_linear_fused`| `lora_sites_wrapped`         | `lora_linear_fused_dispatches` / `lora_linear_eager_dispatches` |
   | `layer_norm_fused` | `layer_norms`                | `ln_fused_dispatches` / `ln_eager_dispatches`             |
   | `gelu_erf_fused`   | `gelu_seam_calls_per_forward`| `gelu_fused_dispatches` / `gelu_eager_dispatches`         |

   and the equation is `fused + eager == census × steps_measured` (exact
   integer equality). A census value of `0` is a real, checkable claim,
   not a skip: a CLIP leg must then read `0`/`0`, because `quick_gelu` has
   no seam. The eager SHARE is reported either way, never hidden.

   **The measurement convention is PINNED before anything is compared.**
   `batches == steps_measured` is true only at `--grad-accum 1` AND
   `--epochs 1`, and a leg that reports anything else is refused by NAME
   rather than reported as an equation failure. `--grad-accum 1` is the
   obvious half (one optimizer step is one training forward). `--epochs 1`
   is the half that is easy to get wrong: `finetune_run::run` drives
   `epochs` single-epoch, resume-chained `TrainingLoop::run` legs and SUMS
   each leg's `TrainingResult::total_steps`, but that field is the leg's
   own `global_step`, which a RESUMED leg carries forward from before the
   resume. A 2-epoch, 2-batch-per-epoch run therefore reports
   `steps_measured == 6` for `4` training forwards. Every #421 leg pins
   `--epochs 1`, where the two coincide exactly — proven on real CLI output
   by `crates/jammi-bench/tests/finetune_run_smoke.rs`'s
   `fusible_site_census_satisfies_the_positive_proof_equation_on_a_real_run`.

   EVAL forwards contribute nothing to EITHER side of any of the three
   pairs (the LoRA site early-returns in eval, the house LayerNorm's fused
   arm is under its training branch, and the GELU seam's eval arm is the
   plain `Tensor::gelu_erf`), so held-out evaluations and train probes do
   not enter the equation. It is UNDEFINED for a window that mixes forwards
   the tier does not count as steps.

2. **`fused > 0` for `lora_linear_fused` and `layer_norm_fused` on an A
   leg.** "A leg" is decided from the leg's OWN recorded arm (`kernels_
   disabled` empty), never from its id string.

3. **`fused == 0` for every key in `kernels_disabled_expected` on a D leg**
   — the forced-eager twin actually ran eager. `finetune-run
   --expect-kernels-disabled` already refuses in-process; this is the
   independent second read, off the emitted report.

4. **The two runs of a pair describe the SAME built model**: their
   `fusible_site_census` values must be equal. Two runs whose site counts
   differ are not an N/M pair of one workload, and differencing them is
   meaningless.

5. **Every number read is FINITE.** Explicitly, via `math.isfinite`, at the
   point of reading — never implicitly via a comparison. `NaN > 0` is
   `False` and `NaN < 0` is `False`, so a diverged run whose wall or loss
   came back `NaN` would sail through a naive threshold check and be
   reported as a clean measurement.

## The per-step decomposition (§D4 item 5)

Per `(N, M)` pair, with `d = steps_measured_M − steps_measured_N`:

    wall_per_step  = (train_run_wall_s_M − train_run_wall_s_N) / d
    front_per_step = (media_front_end_wall_s_M − ..._N) / d
    busy_per_step  = census["gpu_kernel_us_per_step"] / 1e6
    residual       = wall_per_step − front_per_step − busy_per_step

`media_front_end_wall_s` is `null` on a TEXT leg by construction (the
trainer reports `Duration::ZERO` there and the tier refuses to call that a
measurement), so a text leg's `front_per_step` is `0.0` carrying the label
`front_end_inside_residual: true` — tokenization cost stays inside the
residual and the table says so rather than implying the front end was timed
and found free.

SIGN CONVENTION, pinned: `front` is CPU wall and `busy` is GPU device time,
on two different timelines. A NEGATIVE residual therefore MEASURES OVERLAP
(the front end pipelined against the previous step's kernels), reported as
`overlap_s = −residual`. It NEVER invalidates the leg. `residual_s` keeps
the signed value regardless, so a consumer never has to reconstruct the
sign from two fields.

## P2 (`--p2-dir`, contract §D4 item 4)

The BF16 pre-flight's assertions, over `profile_421_legs.sh
PROFILE_421_P2_BF16=1`'s output: exit 0, the run really was
`bf16`+`gaussian`, `final_loss_diagnostic` finite,
`train_probe_series[0] != train_probe_series[1]` (the Gaussian adapter moves
the loss at step 1 — under the wire-default ZerosB it could not, so this
assertion is only meaningful BECAUSE the pre-flight pins `gaussian`), and
`lora_linear_fused`/`layer_norm_fused` fused counters non-zero.

## Exit status

`0` when the merge itself completed and a table was written — a leg
verdicted INVALID or a P2 tower verdicted FAIL is a RECORDED OUTCOME, not a
tool failure, and is announced loudly on stderr. `1` is a tool-level
failure (an unreadable input directory, an unwritable output). `2` is a
usage error.

Run:
  python3 ci/scripts/perf/profile_421_merge.py --legs-dir .profile-421-legs/<ts> \\
      --out .profile-421-legs/<ts>/merged.json
  python3 ci/scripts/perf/profile_421_merge.py --p2-dir .profile-421-legs/<ts>/p2-bf16 \\
      --out .../p2_verdicts.json
Hermetic self-tests: `python3 ci/scripts/perf/test_profile_421_merge.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# The admit-key -> (fused field, eager field, census field) mapping, spelled
# out ONCE. Every entry is a pair of names that must agree across three
# independent surfaces (`jammi_kernels::admission`'s op key, the
# `FinetuneRunTier` counter field, and the `FusibleSiteCensus` field), which
# is exactly why it is a table here rather than three scattered lookups.
KEY_FIELDS: dict[str, tuple[str, str, str]] = {
    "lora_linear_fused": (
        "lora_linear_fused_dispatches",
        "lora_linear_eager_dispatches",
        "lora_sites_wrapped",
    ),
    "layer_norm_fused": (
        "ln_fused_dispatches",
        "ln_eager_dispatches",
        "layer_norms",
    ),
    "gelu_erf_fused": (
        "gelu_fused_dispatches",
        "gelu_eager_dispatches",
        "gelu_seam_calls_per_forward",
    ),
}

# The keys an A (fused) leg must show a NON-ZERO fused count for. `gelu_erf_
# fused` is deliberately absent: its census is legitimately 0 on both CLIP
# towers (`quick_gelu` has no seam), so requiring `fused > 0` there would
# fail a correct leg.
A_LEG_FUSED_REQUIRED = ("lora_linear_fused", "layer_norm_fused")

VERDICT_VALID = "VALID"
VERDICT_INVALID = "INVALID"
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# `wall_s_per_step` cross-check tolerance. Both sides divide the SAME two
# `train_run_wall_s` floats by the same integer denominator, so they agree
# bit-for-bit in practice; the tolerance exists so a future census that
# rounds its persisted value cannot turn a correct leg red.
WALL_CROSS_CHECK_REL_TOL = 1e-9


class LegReadError(Exception):
    """A leg's inputs could not be read/parsed. Carries the message that
    becomes this leg's INVALID reason — never a traceback."""


def _load_json(path: Path, what: str) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
    except FileNotFoundError as exc:
        raise LegReadError(f"{what} is missing ({path})") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LegReadError(f"{what} ({path}) could not be read as JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise LegReadError(f"{what} ({path}) is not a JSON object")
    return loaded


def _tier(report: dict, what: str) -> dict:
    tiers = report.get("tiers")
    tier = tiers.get("finetune_run") if isinstance(tiers, dict) else None
    if not isinstance(tier, dict):
        raise LegReadError(f"{what} carries no `tiers.finetune_run` object")
    return tier


def finite_float(value: object, label: str, reasons: list[str]) -> float | None:
    """A float read that REFUSES non-finite input at the point of reading.

    `None` is returned (and a reason appended) for a missing field, a
    non-numeric field, and — the case this helper exists for — a `NaN` or
    `±inf`. A caller that merely compared the raw value against a threshold
    would silently pass a diverged run: `NaN > 0`, `NaN < 0` and `NaN == 0`
    are ALL `False`, so no ordinary comparison can reject one.
    """
    if value is None:
        reasons.append(f"{label} is absent")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reasons.append(f"{label} is not a number ({value!r})")
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        reasons.append(f"{label} is not finite ({value!r})")
        return None
    return as_float


def nonneg_int(value: object, label: str, reasons: list[str]) -> int | None:
    """An integer read that refuses a non-integer, a bool, or a negative.

    Dispatch counters and site counts are COUNTS: a float here (even a
    whole-valued one) means the field was not what this reader thinks it is,
    and a negative one means a delta was taken the wrong way round.
    """
    if value is None:
        reasons.append(f"{label} is absent")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        reasons.append(f"{label} is not an integer ({value!r})")
        return None
    if value < 0:
        reasons.append(f"{label} is negative ({value})")
        return None
    return value


def check_batches_convention(tier: dict, run_label: str, reasons: list[str]) -> bool:
    """`steps_measured` is the equation's `batches` term only under the
    convention the legs pin. Refuse by NAME when it is not met.

    Reporting an equation FAILURE for a leg that simply was not run under
    the convention the equation is defined for would be the "restored
    coverage that dissolves when the rule is aligned" mistake in miniature:
    the number would move for a reason that has nothing to do with what
    dispatched.
    """
    ok = True
    grad_accum = tier.get("grad_accum")
    if grad_accum != 1:
        reasons.append(
            f"{run_label}: grad_accum={grad_accum!r}, but the positive-proof equation's "
            "`batches` term is `steps_measured` only at --grad-accum 1 (one optimizer step "
            "is one training forward)"
        )
        ok = False
    epochs = tier.get("epochs")
    if epochs != 1:
        reasons.append(
            f"{run_label}: epochs={epochs!r}, but the equation's `batches` term is "
            "`steps_measured` only at --epochs 1: this tier drives one resume-chained "
            "TrainingLoop leg per epoch and sums each leg's own `global_step`, which a "
            "RESUMED leg carries forward from before the resume, so steps_measured "
            "over-counts training forwards for any multi-epoch run"
        )
        ok = False
    return ok


def read_census(tier: dict, run_label: str, reasons: list[str]) -> dict[str, int] | None:
    """The WITNESSED per-forward call counts for this run's built encoder.

    Absent entirely => the leg is INVALID, never "assume the obvious value":
    guessing `calls` is precisely the failure §D4 item 1 exists to close, and
    a build that does not emit the census cannot support the equation at all.
    """
    census = tier.get("fusible_site_census")
    if census is None:
        reasons.append(
            f"{run_label}: no `fusible_site_census` on the tier — the positive-proof "
            "equation reads `calls` off the built encoder's own census and refuses to "
            "guess it; re-run on a build that emits the field"
        )
        return None
    if not isinstance(census, dict):
        reasons.append(f"{run_label}: `fusible_site_census` is not an object ({census!r})")
        return None
    out: dict[str, int] = {}
    ok = True
    for _key, (_fused, _eager, census_field) in KEY_FIELDS.items():
        value = nonneg_int(
            census.get(census_field), f"{run_label}: fusible_site_census.{census_field}", reasons
        )
        if value is None:
            ok = False
        else:
            out[census_field] = value
    return out if ok else None


def check_positive_proof(
    tier: dict,
    run_label: str,
    census: dict[str, int],
    steps_measured: int,
    disabled_expected: list[str],
    reasons: list[str],
) -> dict[str, dict]:
    """The per-key `fused + eager == census × steps_measured` equation, plus
    the arm-specific one-sided requirements. Returns the per-key REPORT
    (populated whether or not the checks passed — the split is reported
    either way, per the contract's "never hidden"); failures land in
    `reasons`, which is what makes the leg INVALID."""
    is_a_leg = not disabled_expected
    per_key: dict[str, dict] = {}
    for key, (fused_field, eager_field, census_field) in KEY_FIELDS.items():
        calls = census[census_field]
        fused = nonneg_int(tier.get(fused_field), f"{run_label}: {fused_field}", reasons)
        eager = nonneg_int(tier.get(eager_field), f"{run_label}: {eager_field}", reasons)
        entry: dict[str, object] = {
            "census_field": census_field,
            "calls_per_forward": calls,
            "steps_measured": steps_measured,
            "expected_total": calls * steps_measured,
            "fused": fused,
            "eager": eager,
        }
        per_key[key] = entry
        if fused is None or eager is None:
            entry["equation_ok"] = False
            continue
        expected = calls * steps_measured
        equation_ok = fused + eager == expected
        entry["equation_ok"] = equation_ok
        entry["eager_share"] = (eager / (fused + eager)) if (fused + eager) else None
        if not equation_ok:
            reasons.append(
                f"{run_label}: {key} positive proof failed — {fused_field}={fused} + "
                f"{eager_field}={eager} = {fused + eager}, but the witnessed census says "
                f"{census_field}={calls} × steps_measured={steps_measured} = {expected}"
            )
        if is_a_leg and key in A_LEG_FUSED_REQUIRED and fused == 0:
            reasons.append(
                f"{run_label}: {key} has {fused_field}=0 on a leg that disabled nothing — "
                f"an A leg must show the fused arm actually dispatched "
                f"(witnessed {census_field}={calls})"
            )
        if key in disabled_expected and fused != 0:
            reasons.append(
                f"{run_label}: {key} is in kernels_disabled_expected but {fused_field}={fused} — "
                "the forced-eager twin dispatched the fused kernel it claimed to have disabled"
            )
    return per_key


def merge_leg(leg_dir: Path) -> dict:
    """One leg's row. NEVER raises for a leg-level problem: every failure
    mode becomes this row's `verdict` + `reasons`, so one bad leg cannot
    discard the other eleven."""
    leg_id = leg_dir.name
    row: dict[str, object] = {
        "leg_id": leg_id,
        "tower": None,
        "task": None,
        "dtype": None,
        "arm": None,
        "kernels_disabled": None,
        "verdict": VERDICT_INVALID,
        "reasons": [],
        "positive_proof": None,
        "per_step": None,
        # Set only when BOTH runs of this leg's own N/M pair agree (unit-467
        # finding R3) — `None` otherwise, which is what makes these safe to
        # feed into `_check_cross_tower_identity`'s per-tower comparison
        # without that pass having to re-derive "did this leg even agree
        # with itself" on every read.
        "checkpoint_weights_sha256": None,
        "fusible_site_census": None,
    }
    reasons: list[str] = []
    try:
        manifest = _load_json(leg_dir / "manifest.json", f"{leg_id}: manifest.json")
    except LegReadError as exc:
        row["reasons"] = [str(exc)]
        return row

    row["tower"] = manifest.get("tower")
    row["task"] = manifest.get("task")
    row["dtype"] = manifest.get("dtype")
    disabled_manifest = manifest.get("kernels_disabled")
    if not isinstance(disabled_manifest, list):
        row["reasons"] = [f"{leg_id}: manifest.kernels_disabled is not a list"]
        return row
    disabled_manifest = sorted(str(k) for k in disabled_manifest)
    row["kernels_disabled"] = disabled_manifest
    row["arm"] = "A" if not disabled_manifest else "D"

    # The driver's OWN verdict is honoured, never re-litigated: it saw the
    # stderr, the census exit code and the corpus provisioning this reader
    # cannot.
    if manifest.get("status") != "ok":
        row["reasons"] = [
            f"{leg_id}: the driver recorded status={manifest.get('status')!r} — "
            f"{manifest.get('reason') or 'no reason recorded'}"
        ]
        return row

    try:
        run_n = _tier(_load_json(leg_dir / "run_n.json", f"{leg_id}: run_n.json"), "run_n.json")
        run_m = _tier(_load_json(leg_dir / "run_m.json", f"{leg_id}: run_m.json"), "run_m.json")
        census_file = _load_json(leg_dir / "census.json", f"{leg_id}: census.json")
    except LegReadError as exc:
        row["reasons"] = [str(exc)]
        return row

    # The leg's declared arm must match what BOTH runs recorded, or this
    # row's numbers came from a different invocation than the one the
    # manifest describes.
    proof: dict[str, dict] = {}
    steps: dict[str, int] = {}
    censuses: dict[str, dict[str, int]] = {}
    checkpoint_shas: dict[str, str] = {}
    for run_label, tier in (("run_n", run_n), ("run_m", run_m)):
        # `steps_measured` is read FIRST and independently of everything
        # else: the wall/front/busy decomposition needs only it, and a leg
        # whose site census is missing must still report its per-step
        # timings (labelled INVALID for the census reason) rather than come
        # back with an empty table that reads as "nothing was measured".
        steps_measured = nonneg_int(
            tier.get("steps_measured"), f"{run_label}: steps_measured", reasons
        )
        if steps_measured == 0:
            reasons.append(f"{run_label}: steps_measured is 0 — nothing was measured")
        elif steps_measured is not None:
            steps[run_label] = steps_measured

        # Checkpoint identity (unit-467 finding R3): read UNCONDITIONALLY,
        # independent of every other check below — a wrong-checkpoint leg
        # must be catchable even if its arm/census/equation all otherwise
        # look fine.
        checkpoint_sha = tier.get("checkpoint_weights_sha256")
        if not isinstance(checkpoint_sha, str) or not checkpoint_sha:
            reasons.append(
                f"{run_label}: checkpoint_weights_sha256 is absent or not a non-empty string"
            )
        else:
            checkpoint_shas[run_label] = checkpoint_sha

        # Finding F1 (unit-467 adversarial audit): an A leg (this leg's own
        # `manifest.kernels_disabled` is empty) makes a POSITIVE claim that
        # nothing was disabled. `kernels_disabled_expected` alone cannot
        # catch a contaminated A leg (it stays `[]` on an unclaimed leg
        # regardless of the real env), so this reads the PROCESS-RESOLVED
        # `kernels_disabled_requested` — the same field the binary's own
        # `--arm fused` refusal reads — and refuses BY NAME when it is
        # non-empty on a leg that declared itself an A leg. `finetune-run`'s
        # own start-of-run check (this PR's companion fix) should make this
        # branch unreachable for a NEW run, but a leg produced by an OLDER
        # binary build (or a manifest hand-edited after the fact) must still
        # be caught here, independently.
        requested = tier.get("kernels_disabled_requested")
        if not isinstance(requested, list):
            reasons.append(f"{run_label}: kernels_disabled_requested is absent or not a list")
            continue
        requested_sorted = sorted(str(k) for k in requested)
        if not disabled_manifest and requested_sorted:
            reasons.append(
                f"{run_label}: this leg declared itself an A leg (kernels_disabled=[]) but "
                f"kernels_disabled_requested={requested_sorted} — an ambient JAMMI_KERNELS_DISABLE "
                "contaminated this leg (finding F1); the run is INVALID, not a datum"
            )
            continue

        expected = tier.get("kernels_disabled_expected")
        if not isinstance(expected, list):
            reasons.append(f"{run_label}: kernels_disabled_expected is absent or not a list")
            continue
        expected_sorted = sorted(str(k) for k in expected)
        if expected_sorted != disabled_manifest:
            reasons.append(
                f"{run_label}: kernels_disabled_expected={expected_sorted} does not match the "
                f"manifest's declared arm {disabled_manifest}"
            )
            continue
        # The convention gate comes BEFORE the equation, so a leg run
        # outside it is refused by name rather than reported as a counter
        # mismatch it never had.
        if not check_batches_convention(tier, run_label, reasons):
            continue
        census = read_census(tier, run_label, reasons)
        if census is None or run_label not in steps:
            continue
        censuses[run_label] = census
        proof[run_label] = check_positive_proof(
            tier, run_label, census, steps[run_label], expected_sorted, reasons
        )

    if len(checkpoint_shas) == 2 and checkpoint_shas["run_n"] != checkpoint_shas["run_m"]:
        reasons.append(
            f"the pair's checkpoint_weights_sha256 differ (run_n={checkpoint_shas['run_n']}, "
            f"run_m={checkpoint_shas['run_m']}) — the two runs did not load the same checkpoint "
            "bytes, so differencing them is not a measurement of one workload"
        )
    elif len(checkpoint_shas) == 2:
        row["checkpoint_weights_sha256"] = checkpoint_shas["run_n"]

    if len(censuses) == 2 and censuses["run_n"] != censuses["run_m"]:
        reasons.append(
            f"the pair's witnessed site censuses differ (run_n={censuses['run_n']}, "
            f"run_m={censuses['run_m']}) — the two runs did not build the same model, so "
            "differencing them is not a measurement of one workload"
        )
    elif len(censuses) == 2:
        row["fusible_site_census"] = censuses["run_n"]

    row["positive_proof"] = proof or None
    per_step, per_step_reasons = decompose_per_step(run_n, run_m, census_file, steps)
    reasons.extend(per_step_reasons)
    row["per_step"] = per_step
    row["reasons"] = reasons
    row["verdict"] = VERDICT_INVALID if reasons else VERDICT_VALID
    return row


def decompose_per_step(
    run_n: dict, run_m: dict, census_file: dict, steps: dict[str, int]
) -> tuple[dict | None, list[str]]:
    """`(per_step_row_or_None, reasons)` — the wall/front/busy/residual
    decomposition of §D4 item 5. A negative residual is an OVERLAP
    measurement and produces NO reason; every other anomaly does."""
    reasons: list[str] = []
    if "run_n" not in steps or "run_m" not in steps:
        return None, reasons
    steps_n, steps_m = steps["run_n"], steps["run_m"]
    if steps_m <= steps_n:
        reasons.append(
            f"steps_measured did not increase across the pair (N={steps_n}, M={steps_m}) — "
            "the differencing denominator would be zero or negative"
        )
        return None, reasons
    denom = steps_m - steps_n

    wall_n = finite_float(run_n.get("train_run_wall_s"), "run_n: train_run_wall_s", reasons)
    wall_m = finite_float(run_m.get("train_run_wall_s"), "run_m: train_run_wall_s", reasons)
    if wall_n is None or wall_m is None:
        return None, reasons
    if not wall_m > wall_n > 0:
        reasons.append(
            f"the pair's walls are outside the differencing domain (wall_n={wall_n}, "
            f"wall_m={wall_m}) — `wall_m > wall_n > 0` must hold for the same workload at "
            "M > N steps"
        )
        return None, reasons
    wall_per_step = (wall_m - wall_n) / denom

    # The TEXT arm: `media_front_end_wall_s` is null by construction, so the
    # front end is 0 and the label says the tokenization cost lives in the
    # residual. Both-null and both-present are the only coherent shapes; one
    # of each means the pair mixed a text run with a media run.
    raw_front_n = run_n.get("media_front_end_wall_s")
    raw_front_m = run_m.get("media_front_end_wall_s")
    front_inside_residual = raw_front_n is None and raw_front_m is None
    if front_inside_residual:
        front_per_step = 0.0
    elif raw_front_n is None or raw_front_m is None:
        reasons.append(
            "media_front_end_wall_s is null on exactly one run of the pair "
            f"(n={raw_front_n!r}, m={raw_front_m!r}) — a text run was paired with a media run"
        )
        return None, reasons
    else:
        front_n = finite_float(raw_front_n, "run_n: media_front_end_wall_s", reasons)
        front_m = finite_float(raw_front_m, "run_m: media_front_end_wall_s", reasons)
        if front_n is None or front_m is None:
            return None, reasons
        if front_n < 0 or front_m < 0:
            reasons.append(
                f"media_front_end_wall_s is negative (n={front_n}, m={front_m}) — "
                "a wall-clock accumulator cannot run backwards"
            )
            return None, reasons
        front_per_step = (front_m - front_n) / denom
        if front_per_step < 0:
            reasons.append(
                f"the front end got CHEAPER with more rows (front_per_step={front_per_step}) — "
                "the M run decoded more media than the N run, so this difference is incoherent"
            )
            return None, reasons

    busy_us = finite_float(
        census_file.get("gpu_kernel_us_per_step"), "census: gpu_kernel_us_per_step", reasons
    )
    if busy_us is None:
        return None, reasons
    if busy_us < 0:
        reasons.append(f"census gpu_kernel_us_per_step is negative ({busy_us})")
        return None, reasons
    busy_per_step = busy_us / 1e6

    # The census computed the SAME wall difference from the SAME two floats
    # the driver handed it. A disagreement means this `census.json` was not
    # produced from this pair of reports.
    census_wall = census_file.get("wall_s_per_step")
    if census_wall is not None:
        census_wall_f = finite_float(census_wall, "census: wall_s_per_step", reasons)
        if census_wall_f is None:
            return None, reasons
        if not math.isclose(
            census_wall_f, wall_per_step, rel_tol=WALL_CROSS_CHECK_REL_TOL, abs_tol=0.0
        ):
            reasons.append(
                f"census wall_s_per_step={census_wall_f} disagrees with the reports' own "
                f"(wall_m - wall_n)/(M - N)={wall_per_step} — this census.json was not built "
                "from this pair of reports"
            )
            return None, reasons

    residual = wall_per_step - front_per_step - busy_per_step
    return (
        {
            "steps_measured_n": steps_n,
            "steps_measured_m": steps_m,
            "steps_diff": denom,
            "wall_s_per_step": wall_per_step,
            "front_s_per_step": front_per_step,
            "busy_s_per_step": busy_per_step,
            # SIGNED, always. Negative == overlap; see this module's doc.
            "residual_s_per_step": residual,
            "overlap_s_per_step": (-residual) if residual < 0 else None,
            "front_end_inside_residual": front_inside_residual,
            "front_share_of_wall": (front_per_step / wall_per_step) if wall_per_step else None,
            "busy_share_of_wall": (busy_per_step / wall_per_step) if wall_per_step else None,
        },
        reasons,
    )


def merge_p2(tower_dir: Path) -> dict:
    """One tower's BF16 pre-flight verdict (contract §D4 item 4)."""
    tower_id = tower_dir.name
    row: dict[str, object] = {
        "tower": tower_id,
        "verdict": VERDICT_FAIL,
        "reasons": [],
        "exit": None,
        "backbone_dtype": None,
        "lora_init": None,
        "final_loss_diagnostic": None,
        "train_probe_series_head": None,
        "lora_linear_fused_dispatches": None,
        "ln_fused_dispatches": None,
        # Unit-467 finding R3: fed into `_check_cross_tower_identity`
        # alongside this tower's own A1/A2/D1/D2 legs, exactly like the
        # matching fields on a leg row.
        "checkpoint_weights_sha256": None,
        "fusible_site_census": None,
    }
    reasons: list[str] = []
    try:
        manifest = _load_json(tower_dir / "manifest.json", f"{tower_id}: manifest.json")
    except LegReadError as exc:
        row["reasons"] = [str(exc)]
        return row
    row["tower"] = manifest.get("tower", tower_id)

    exit_code = manifest.get("exit")
    row["exit"] = exit_code
    if exit_code != 0:
        reasons.append(
            f"{tower_id}: the pre-flight run exited {exit_code!r}, not 0 — "
            f"{manifest.get('reason') or 'no reason recorded'}"
        )

    try:
        tier = _tier(_load_json(tower_dir / "run.json", f"{tower_id}: run.json"), "run.json")
    except LegReadError as exc:
        reasons.append(str(exc))
        row["reasons"] = reasons
        return row

    # The run must BE the pre-flight configuration. Without this, a
    # mis-invoked f32/zeros_b run would satisfy every assertion below
    # trivially and be recorded as a passing bf16 pre-flight.
    dtype = tier.get("backbone_dtype")
    lora_init = tier.get("lora_init")
    row["backbone_dtype"] = dtype
    row["lora_init"] = lora_init

    # Checkpoint identity (unit-467 finding R3), read the same way a leg's
    # own per-run read is — the P2 pre-flight is a SINGLE untraced run, so
    # there is no N/M pair to cross-check within, only this tower's OTHER
    # rows (its four legs) via `_check_cross_tower_identity`.
    checkpoint_sha = tier.get("checkpoint_weights_sha256")
    if not isinstance(checkpoint_sha, str) or not checkpoint_sha:
        reasons.append(f"{tower_id}: checkpoint_weights_sha256 is absent or not a non-empty string")
    else:
        row["checkpoint_weights_sha256"] = checkpoint_sha
    p2_census_reasons: list[str] = []
    census = read_census(tier, f"{tower_id}: p2", p2_census_reasons)
    if census is not None:
        row["fusible_site_census"] = census
    # Deliberately NOT folded into `reasons`/this row's own verdict: a
    # missing census here would already make `read_census`'s caller-visible
    # failure mode fire wherever THIS tower's real legs need it, and P2's
    # own pass/fail is about the bf16 dtype path, not the census shape —
    # `_check_cross_tower_identity` is the only consumer of this field and
    # already reports its own reason when it finds a mismatch.

    if dtype != "bf16":
        reasons.append(f"{tower_id}: backbone_dtype={dtype!r}, but the P2 pre-flight is bf16")
    if lora_init != "gaussian":
        reasons.append(
            f"{tower_id}: lora_init={lora_init!r}, but the P2 pre-flight pins gaussian — "
            "under zeros_b the two probes could not differ, so the probe assertion below "
            "would be vacuous"
        )

    loss = finite_float(
        tier.get("final_loss_diagnostic"), f"{tower_id}: final_loss_diagnostic", reasons
    )
    row["final_loss_diagnostic"] = loss

    series = tier.get("train_probe_series")
    if not isinstance(series, list) or len(series) < 2:
        reasons.append(
            f"{tower_id}: train_probe_series is not a list of at least 2 entries ({series!r}) — "
            "index 0 is the untrained init probe and index 1 the post-epoch one"
        )
    else:
        row["train_probe_series_head"] = series[:2]
        probe_0 = finite_float(series[0], f"{tower_id}: train_probe_series[0]", reasons)
        probe_1 = finite_float(series[1], f"{tower_id}: train_probe_series[1]", reasons)
        if probe_0 is not None and probe_1 is not None and probe_0 == probe_1:
            reasons.append(
                f"{tower_id}: train_probe_series[0] == train_probe_series[1] == {probe_0} — "
                "the Gaussian adapter did not move the loss at step 1, so this run proves "
                "nothing about whether the bf16 backbone trains"
            )

    for field in ("lora_linear_fused_dispatches", "ln_fused_dispatches"):
        count = nonneg_int(tier.get(field), f"{tower_id}: {field}", reasons)
        row[field] = count
        if count is not None and count == 0:
            reasons.append(
                f"{tower_id}: {field}=0 — the fused arm never dispatched, so the pre-flight "
                "did not exercise the path the legs measure"
            )

    row["reasons"] = reasons
    row["verdict"] = VERDICT_FAIL if reasons else VERDICT_PASS
    return row


def _leg_dirs(legs_dir: Path) -> list[Path]:
    """Every leg subdirectory, sorted. `p2-bf16` is the driver's OWN
    pre-flight output living under the same `$OUT_DIR` and is not a leg."""
    return sorted(
        d
        for d in legs_dir.iterdir()
        if d.is_dir() and d.name != "p2-bf16" and (d / "manifest.json").is_file()
    )


# The two fields `_check_cross_tower_identity` compares across a tower's
# rows, paired with a human-readable label for the reason string.
_CROSS_TOWER_FIELDS = (
    ("checkpoint_weights_sha256", "checkpoint_weights_sha256"),
    ("fusible_site_census", "fusible_site_census"),
)


def _check_cross_tower_identity(legs: list[dict], p2_rows: list[dict]) -> None:
    """Unit-467 finding R3: every row of the SAME tower — its A1/A2/D1/D2
    legs AND its P2 bf16 pre-flight row, when both are present in this merge
    — must agree on `checkpoint_weights_sha256` and `fusible_site_census`.
    Two rows of one tower that measured DIFFERENT checkpoint bytes or built
    a DIFFERENT encoder are not comparable at all, whatever their own
    within-row checks already found.

    Mutates `legs`/`p2_rows` IN PLACE: a disagreeing row's verdict is
    downgraded (`VALID` -> `INVALID` for a leg, `PASS` -> `FAIL` for a P2
    row) and a reason naming the mismatch — by tower, by field, and by
    row-id — is appended, even when every other check on that row passed.
    A row whose own value for a field is `None` (it already failed some
    OTHER check that field depends on) is left out of the comparison for
    that field entirely: a row that never produced a witnessed value cannot
    be blamed for disagreeing with one that did, and is already INVALID/FAIL
    for the reason that made the value `None` in the first place.
    """
    by_tower: dict[str, list[tuple[str, dict]]] = {}
    for row in legs:
        tower = row.get("tower")
        if isinstance(tower, str):
            by_tower.setdefault(tower, []).append(("leg", row))
    for row in p2_rows:
        tower = row.get("tower")
        if isinstance(tower, str):
            by_tower.setdefault(tower, []).append(("p2", row))

    for tower, kind_rows in by_tower.items():
        for field, label in _CROSS_TOWER_FIELDS:
            present = [(kind, row) for kind, row in kind_rows if row.get(field) is not None]
            if len(present) < 2:
                continue
            # Canonicalized (`fusible_site_census` is a dict) so equality is
            # a genuine structural comparison, not an identity check on the
            # unhashable dict itself.
            distinct = {json.dumps(row[field], sort_keys=True) for _kind, row in present}
            if len(distinct) <= 1:
                continue
            summary = {
                (row.get("leg_id") if kind == "leg" else f"{row.get('tower')} (P2)"): row[field]
                for kind, row in present
            }
            for kind, row in present:
                row_id = row.get("leg_id") if kind == "leg" else f"{row.get('tower')} (P2)"
                reason = (
                    f"{row_id}: tower {tower!r}'s rows disagree on {label} — {summary!r} — "
                    "these rows did not measure the same checkpoint/build and are not "
                    "comparable, whatever their own within-row checks found"
                )
                if kind == "leg":
                    row["reasons"].append(reason)
                    row["verdict"] = VERDICT_INVALID
                else:
                    row["reasons"].append(reason)
                    row["verdict"] = VERDICT_FAIL


def build_report(legs_dir: Path | None, p2_dir: Path | None) -> dict:
    report: dict[str, object] = {"tool": "profile_421_merge", "schema": 1}
    legs: list[dict] = []
    p2_rows: list[dict] = []
    if legs_dir is not None:
        report["legs_dir"] = str(legs_dir)
        legs = [merge_leg(d) for d in _leg_dirs(legs_dir)]
        report["legs"] = legs
    if p2_dir is not None:
        report["p2_dir"] = str(p2_dir)
        p2_rows = [
            merge_p2(d)
            for d in sorted(
                d for d in p2_dir.iterdir() if d.is_dir() and (d / "manifest.json").is_file()
            )
        ]
        report["p2_bf16"] = p2_rows
    # Unit-467 finding R3: a per-tower CROSS-leg (and cross-P2) identity
    # check, run AFTER every leg/P2 row has its own within-row verdict — a
    # row already INVALID/FAIL for its own reason can still surface a
    # cross-tower mismatch reason too (both are true), and a row that was
    # otherwise VALID/PASS can be downgraded by this pass alone.
    _check_cross_tower_identity(legs, p2_rows)
    report["summary"] = {
        "legs_total": len(legs),
        "legs_valid": sum(1 for row in legs if row["verdict"] == VERDICT_VALID),
        "legs_invalid": sum(1 for row in legs if row["verdict"] == VERDICT_INVALID),
        "p2_total": len(p2_rows),
        "p2_pass": sum(1 for row in p2_rows if row["verdict"] == VERDICT_PASS),
        "p2_fail": sum(1 for row in p2_rows if row["verdict"] == VERDICT_FAIL),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_merge.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", help="a profile_421_legs.sh $OUT_DIR to merge the 12 legs from")
    ap.add_argument("--p2-dir", help="the PROFILE_421_P2_BF16=1 output dir ($OUT_DIR/p2-bf16)")
    ap.add_argument("--out", help="write the merged table here (default: stdout)")
    args = ap.parse_args(argv)

    if not args.legs_dir and not args.p2_dir:
        print("::error::profile_421_merge: at least one of --legs-dir/--p2-dir is required", file=sys.stderr)
        return 2

    dirs: list[Path | None] = []
    for raw in (args.legs_dir, args.p2_dir):
        if raw is None:
            dirs.append(None)
            continue
        path = Path(raw)
        if not path.is_dir():
            print(f"::error::profile_421_merge: {path} is not a directory", file=sys.stderr)
            return 1
        dirs.append(path)
    legs_dir, p2_dir = dirs

    report = build_report(legs_dir, p2_dir)

    payload = json.dumps(report, indent=1, sort_keys=False)
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::profile_421_merge: could not write {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)

    # Loud on stderr, exit 0: an INVALID leg is a RECORDED OUTCOME of the
    # experiment, not a failure of this tool. Silence would be the bug.
    for row in report.get("legs", []):
        if row["verdict"] != VERDICT_VALID:
            print(f"::warning::leg {row['leg_id']}: {row['verdict']}", file=sys.stderr)
            for reason in row["reasons"]:
                print(f"  - {reason}", file=sys.stderr)
    for row in report.get("p2_bf16", []):
        if row["verdict"] != VERDICT_PASS:
            print(f"::warning::P2 {row['tower']}: {row['verdict']}", file=sys.stderr)
            for reason in row["reasons"]:
                print(f"  - {reason}", file=sys.stderr)
    summary = report["summary"]
    print(
        "profile_421_merge: "
        f"{summary['legs_valid']}/{summary['legs_total']} legs VALID, "
        f"{summary['p2_pass']}/{summary['p2_total']} P2 towers PASS",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
