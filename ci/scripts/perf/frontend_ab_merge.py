#!/usr/bin/env python3
"""Merge + bar-decision stage for `ci/scripts/perf/frontend_ab.sh`'s #421
follow-on ("media front-end parallelization") A/B driver.

Extracted out of that script's own inline heredoc for the same reason
`ab_merge.py`'s own module doc gives for ITS extraction (B3): an inline
heredoc has ZERO automated coverage of its own report-reading code path
beyond whatever its DRY_RUN stub happens to fabricate. That gap is exactly
how the real A/B rehearsal on pod p421b found this module's own defect --
every one of the eight `finetune-run` raw legs was actually complete, but
the old inline heredoc read `report["steps_measured"]` /
`report["media_front_end_wall_s"]` / `report["train_run_wall_s"]` off the
report's TOP LEVEL, when the real `jammi-bench finetune-run` report shape
(like every other tier this repo reads -- `report.rs`'s `FinetuneRunTier`)
nests them under `tiers.finetune_run`. Every leg therefore read back
`FAIL "report missing 'steps_measured'"` although nothing had actually
failed. The prior DRY_RUN stub emitted a FLAT shape too, so no hermetic
test could catch this before a real pod run did (esc-088's class: a stub
that does not mirror the real envelope hides the exact bug a real run then
hits).

`_tier()` / `LegReadError` / `finite_float` / `nonneg_int` are imported
from `profile_421_merge.py` (this same directory) rather than redeclared
here -- one tier-accessor idiom, one non-finite-refusing float reader,
shared by every #421-family report reader in this directory.

`test_frontend_ab_merge.py` in this same directory drives this module's
real entry point (`main`, exactly what `frontend_ab.sh` invokes) against
fixture directories shaped like `run_leg`'s own `.exit`/`.json` pairs,
including a REAL, committed, envelope-trimmed cut of a `finetune-run`
report from that same pod-p421b rehearsal
(`fixtures/frontend_ab_rehearsal/htsat_tip_r1_envelope.json`) -- never a
hand-rolled flat dict standing in for what a real report looks like.

Never imported by any Cargo crate, never a jammi-bench dependency -- a
CI-adjacent script `frontend_ab.sh` alone invokes, same footing that
script itself already has.

`ratio_lo`/`ratio_hi` (the bar decision's own interval, propagated from
BOTH arms' observed repeats -- never a `spread(base) / mean(base)` term
added directly onto `ratio` as an error bar, which was a UNITS error: that
fraction is relative to `front_base`, not to `ratio`, so adding it onto
`ratio` mixed two different reference scales and made almost any real
result read UNRESOLVED. `build_report`'s own doc has the corrected rule).

Invocation (positional, mirrors `ab_merge.py`'s own convention):
    frontend_ab_merge.py RAW_DIR OUT_PATH SERIAL_TAIL_RATIO N_ITEMS_PER_STEP \
        TIP_SHA BASE_SHA BOX DRY_RUN [REPEATS]

`REPEATS` (optional, default 2) is the number of interleaved base/tip
PAIRS per tower `frontend_ab.sh`'s own `$FRONTEND_AB_REPEATS` ran --
leg files are named `..._r1.json` .. `..._rN.json`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_421_merge import LegReadError, _tier, finite_float, nonneg_int  # noqa: E402

TOWERS = ("htsat", "clip-vision")
ROLES = ("base", "tip")
DEFAULT_REPEATS = 2
# Kept for callers/tests that still want the contract's own default
# base/tip/base/tip pair labels without threading a repeat count through.
REPEATS = ("r1", "r2")


def repeat_labels(repeats: int) -> tuple[str, ...]:
    """`r1` .. `r<repeats>` -- `frontend_ab.sh`'s own leg-file naming
    convention, generalized from the contract's default 2 pairs to
    `$FRONTEND_AB_REPEATS` pairs."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    return tuple(f"r{i}" for i in range(1, repeats + 1))


def load_leg(raw_dir: Path, tower: str, role: str, repeat: str) -> dict:
    """Read one `run_leg`-written raw leg (`<raw_dir>/<tower>__<role>__<repeat>.{exit,json}`)
    back into a merge-ready dict.

    Mirrors `profile_421_merge.py`'s own `_tier()` idiom: the report's
    `tiers.finetune_run` object -- never its top level -- carries
    `steps_measured` / `media_front_end_wall_s` / `train_run_wall_s` /
    `rayon_pool_threads`, exactly the fields `report.rs`'s
    `FinetuneRunTier` serializes them under.
    """
    leg_id = f"{tower}/{role}/{repeat}"
    out_file = Path(raw_dir) / f"{tower}__{role}__{repeat}.json"
    exit_file = Path(raw_dir) / f"{tower}__{role}__{repeat}.exit"
    exit_code = int(exit_file.read_text().strip()) if exit_file.exists() else None
    if exit_code != 0:
        return {"outcome": "FAIL", "exit_code": exit_code}
    try:
        report = json.loads(out_file.read_text())
    except (OSError, ValueError) as e:
        return {
            "outcome": "FAIL",
            "exit_code": exit_code,
            "reason": f"{leg_id}: unparseable report: {e}",
        }
    if not isinstance(report, dict):
        return {"outcome": "FAIL", "reason": f"{leg_id}: report is not a JSON object"}
    try:
        tier = _tier(report, f"{leg_id} report")
    except LegReadError as exc:
        return {"outcome": "FAIL", "reason": str(exc)}

    reasons: list[str] = []
    steps = nonneg_int(tier.get("steps_measured"), f"{leg_id}: steps_measured", reasons)
    if steps == 0:
        reasons.append(f"{leg_id}: steps_measured is 0 -- nothing was measured")
        steps = None
    front_wall = finite_float(
        tier.get("media_front_end_wall_s"), f"{leg_id}: media_front_end_wall_s", reasons
    )
    train_wall = finite_float(
        tier.get("train_run_wall_s"), f"{leg_id}: train_run_wall_s", reasons
    )
    if steps is None or front_wall is None or train_wall is None:
        return {"outcome": "FAIL", "reason": "; ".join(reasons)}

    return {
        "outcome": "OK",
        "front_per_step": front_wall / steps,
        "train_per_step": train_wall / steps,
        "steps_measured": steps,
        "rayon_pool_threads": tier.get("rayon_pool_threads"),
    }


def _front_values(towers: dict, tower: str, role: str, repeats: tuple[str, ...]) -> list[float]:
    return [towers[tower][role][r]["front_per_step"] for r in repeats]


def mean_front(towers: dict, tower: str, role: str, repeats: tuple[str, ...]) -> float:
    vals = _front_values(towers, tower, role, repeats)
    return sum(vals) / len(vals)


def min_front(towers: dict, tower: str, role: str, repeats: tuple[str, ...]) -> float:
    return min(_front_values(towers, tower, role, repeats))


def max_front(towers: dict, tower: str, role: str, repeats: tuple[str, ...]) -> float:
    return max(_front_values(towers, tower, role, repeats))


def build_report(
    raw_dir: Path,
    r: float,
    n: int,
    tip_sha: str,
    base_sha: str,
    box: str,
    dry_run: bool,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """Load every leg under `raw_dir` and compute the merged report +
    HTSAT bar decision -- the exact arithmetic `frontend_ab.sh`'s own
    module doc spells out. `dry_run` only ever affects the `dry_run` field
    of the written report; every other computation is identical on a real
    run and a dry run.

    The bar's interval, `ratio_lo`/`ratio_hi`, is propagated WITHOUT
    assuming a distribution: `ratio_lo = min(tip legs) / max(base legs)`
    and `ratio_hi = max(tip legs) / min(base legs)` -- every tip leg's own
    observed value against every base leg's own observed value, taking the
    extremes. This replaces an earlier, INCORRECT "add
    `spread(base legs) / mean(base)` onto `ratio` as an error bar" rule --
    a units error (`spread / front_base_mean` is a fraction of
    `front_base`, not of `ratio`; adding it directly onto `ratio` mixed two
    different reference scales and made almost any real result read
    UNRESOLVED). PASS iff the WHOLE interval clears the bar
    (`ratio_hi <= upper_bound` and `ratio_lo >= lower_bound`); FAIL iff the
    whole interval is too slow (`ratio_lo > upper_bound`);
    INVALID_BEATS_IDEAL iff the whole interval beats the machine model's
    own ideal (`ratio_hi < lower_bound`); UNRESOLVED otherwise (a bound
    falls strictly inside `[ratio_lo, ratio_hi]`).
    """
    repeat_ids = repeat_labels(repeats)
    towers = {
        tower: {
            role: {repeat: load_leg(raw_dir, tower, role, repeat) for repeat in repeat_ids}
            for role in ROLES
        }
        for tower in TOWERS
    }

    status = "GREEN"
    for tower in TOWERS:
        for role in ROLES:
            for repeat in repeat_ids:
                if towers[tower][role][repeat]["outcome"] != "OK":
                    status = "INVALID"

    report = {
        "tool": "frontend_ab.sh",
        "dry_run": dry_run,
        "base_sha": base_sha,
        "tip_sha": tip_sha,
        "box": box,
        "serial_tail_ratio": r,
        "n_items_per_step": n,
        "repeats": repeats,
        "status": status,
        "towers": towers,
    }

    htsat_bar = None
    if status == "GREEN":
        tip_p_values = {
            towers["htsat"]["tip"][repeat]["rayon_pool_threads"] for repeat in repeat_ids
        }
        if len(tip_p_values) != 1 or None in tip_p_values:
            report["status"] = "INVALID"
            report["invalid_reason"] = (
                f"htsat tip legs report inconsistent/missing rayon_pool_threads: {tip_p_values!r}"
            )
        else:
            p = next(iter(tip_p_values))
            ideal = n / math.ceil(n / p)
            upper_bound = r + (1 - r) / (0.5 * ideal)
            lower_bound = r + (1 - r) / ideal

            front_tip = mean_front(towers, "htsat", "tip", repeat_ids)
            front_base = mean_front(towers, "htsat", "base", repeat_ids)
            ratio = front_tip / front_base
            ratio_lo = min_front(towers, "htsat", "tip", repeat_ids) / max_front(
                towers, "htsat", "base", repeat_ids
            )
            ratio_hi = max_front(towers, "htsat", "tip", repeat_ids) / min_front(
                towers, "htsat", "base", repeat_ids
            )

            if ratio_hi <= upper_bound and ratio_lo >= lower_bound:
                verdict = "PASS"
            elif ratio_lo > upper_bound:
                verdict = "FAIL"
            elif ratio_hi < lower_bound:
                verdict = "INVALID_BEATS_IDEAL"
            else:
                verdict = "UNRESOLVED"

            htsat_bar = {
                "p": p,
                "n": n,
                "ideal": ideal,
                "r": r,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "front_tip_mean_s": front_tip,
                "front_base_mean_s": front_base,
                "ratio": ratio,
                "ratio_lo": ratio_lo,
                "ratio_hi": ratio_hi,
                "verdict": verdict,
            }
    report["htsat_bar"] = htsat_bar

    # NOTE: gated on the local `status` (computed before the tip
    # rayon_pool_threads consistency check above), matching this driver's
    # pre-existing behaviour -- the clip-vision report-only block is not
    # itself gated on the htsat-only rayon_pool_threads consistency check.
    if status == "GREEN" and all(
        towers["clip-vision"][role][repeat]["outcome"] == "OK"
        for role in ROLES
        for repeat in repeat_ids
    ):
        report["clip_vision_report_only"] = {
            "front_tip_mean_s": mean_front(towers, "clip-vision", "tip", repeat_ids),
            "front_base_mean_s": mean_front(towers, "clip-vision", "base", repeat_ids),
            "ratio": mean_front(towers, "clip-vision", "tip", repeat_ids)
            / mean_front(towers, "clip-vision", "base", repeat_ids),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (8, 9):
        print(
            "usage: frontend_ab_merge.py RAW_DIR OUT_PATH SERIAL_TAIL_RATIO "
            "N_ITEMS_PER_STEP TIP_SHA BASE_SHA BOX DRY_RUN [REPEATS]",
            file=sys.stderr,
        )
        return 2
    raw_dir, out_path, r_str, n_str, tip_sha, base_sha, box, dry_run_str = argv[:8]
    repeats = int(argv[8]) if len(argv) == 9 else DEFAULT_REPEATS
    report = build_report(
        Path(raw_dir),
        float(r_str),
        int(n_str),
        tip_sha,
        base_sha,
        box,
        dry_run_str == "1",
        repeats,
    )
    Path(out_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
