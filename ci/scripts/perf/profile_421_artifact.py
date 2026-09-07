#!/usr/bin/env python3
"""Issue #421 tower-profile ARTIFACT step: turn `profile_421_merge.py`'s and
`profile_421_attribute.py`'s own reports into the single committed
`crates/jammi-kernels/artifacts/cuda-runs/*.json` `check_cuda_run_artifacts.py`
schema-checks (contract `scratchpad/contract-421-profile.md` v2.5
`## Artifacts`).

## Why a third script, not hand-assembled JSON

The merge and attribution reports already carry every NUMBER this artifact
needs (positive-proof witnesses, per-step wall/front/busy/residual, chain
shares, `UNATTRIBUTED`, realized gains, the candidate-port verdict strings).
Hand-typing the artifact JSON from those reports would be exactly the
"restored coverage 0.867 -> 0.895" transcription failure mode the house
principles name: a value could drift from its source, or never have come
from a real run at all, and the artifact would read as equally authoritative
either way. This module is the single reader that turns
`--merge-json` + `--attribution-json` + a small, non-numeric identity
sidecar (`--identity`; device strings, checkpoint repo/file names, prose —
see `profile_421_run2_identity.json`'s own header comment for exactly what
it is and is not allowed to carry) into the artifact, and is tested
hermetically against SYNTHETIC merge/attribution/identity/legs-dir fixtures
(`test_profile_421_artifact.py`) — no GPU, no pod, no committed baseline.

## What comes from where

- `git_sha` / `box`: read off every leg's (and, if `--p2-dir` is given,
  every P2 tower's) own `manifest.json` under `--legs-dir` — WITNESSED per
  run, never declared in the identity sidecar — and cross-checked to agree
  on every single one. A run whose legs disagree on which build or which
  physical box produced them is not one measurement session; this module
  refuses rather than pick one arbitrarily.
- `checkpoint_weights_sha256` (per tower family): read off `--merge-json`'s
  own per-leg `checkpoint_weights_sha256` (already cross-checked equal
  within a tower by `profile_421_merge.py`'s own `_check_cross_tower_
  identity`) — cross-checked AGAIN here across every leg of a family before
  it is trusted as "the" sha256 for that family's `notes.checkpoints` entry.
- `launches_per_step` (used only by the CLIP launch-bound finding below):
  read off each named leg's own `census.json` under `--legs-dir` — the one
  number neither the merge nor the attribution report carries.
- Every per-leg wall/front/busy/residual number, every chain share, every
  realized-gain delta, every candidate-port verdict string: copied VERBATIM
  from `--merge-json` / `--attribution-json` — the verdict strings in
  `candidate_decisions` are copied character-for-character, never
  reformatted or re-derived, per the contract's own "the verdict strings
  written once (never edited)" line.
- Everything else in `notes` (`what`, `gpu`, `driver`, `cpu`, `nsys`,
  checkpoint repo/file names, `recorded_deviations` prose, the producer
  `invocation` string): read from `--identity`, which carries ONLY
  provenance with no numeric guarantee attached (a device-model string is
  not a measurement an oracle could re-derive) — see that file's own header
  comment.

## Findings: computed, not hand-typed

`findings` (the four items the contract's own close-out names: HTSAT
front-end share, CLIP-vision front-end share, CLIP launch-bound + the BF16
busy/wall trade at batch 8, and the out-of-tier `C-ATTN-HTSAT` number) are
built by `compute_findings()` below from the SAME merge/attribution numbers
already read for `legs`/`attribution` — the prose sentence is an f-string
formatted around values this module computed, not a string a human typed a
percentage into. Each finding also carries its own `evidence` block (the
raw leg ids and the exact numeric inputs the sentence was built from), so a
downstream reader (the docs-ci guide table, an auditor) can re-derive the
rounded number in the sentence from the unrounded evidence without trusting
the sentence's own English.

Run: `python3 ci/scripts/perf/profile_421_artifact.py --legs-dir <dir>
[--p2-dir <dir>] --merge-json <path> --attribution-json <path>
--identity <path> [--out <path>]`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = 1


class ArtifactBuildError(Exception):
    """Uncomputable or inconsistent input — fails closed, never guesses."""


def _finite(value: object, label: str) -> float:
    """A float read that refuses non-finite input AT THE POINT OF READING —
    `profile_421_merge.py`'s own `finite_float` doctrine, restated here
    because every finding below does ARITHMETIC (subtraction, division,
    min/max) on values read out of `--merge-json`/`--attribution-json`/
    `census.json`. An ordinary threshold or `min`/`max` call would let a
    diverged (`NaN`) upstream measurement sail through silently: `NaN > 0`,
    `NaN < 0` and `NaN == 0` are ALL `False`, and `min(nan, 1.0)` is
    order-dependent, not a refusal. Every value this module ever compares or
    divides is routed through here first."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactBuildError(f"{label} is not a number ({value!r})")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ArtifactBuildError(f"{label} is not finite ({value!r})")
    return as_float


def _load_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {what} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactBuildError(f"{what} at {path} is not valid JSON: {exc}") from exc


def _leg_dirs(legs_dir: Path) -> list[Path]:
    if not legs_dir.is_dir():
        raise ArtifactBuildError(f"--legs-dir {legs_dir} is not a directory")
    return sorted((p for p in legs_dir.iterdir() if p.is_dir()), key=lambda p: p.name)


# --------------------------------------------------------------------------- #
# git_sha / box: witnessed off every manifest.json, cross-checked to agree
# --------------------------------------------------------------------------- #
def collect_identity(legs_dir: Path, p2_dir: Path | None) -> tuple[str, str]:
    """Reads `git_sha` + `box` off every leg's (and, if given, every P2
    tower's) own `manifest.json`, and refuses (`ArtifactBuildError`) unless
    every single one agrees — a mixed-build or mixed-box session is not one
    measurement, and this module never silently picks the first value it
    saw."""
    seen: dict[str, tuple[str, str]] = {}
    for leg_dir in _leg_dirs(legs_dir):
        manifest = _load_json(leg_dir / "manifest.json", f"{leg_dir.name}: manifest.json")
        git_sha = manifest.get("git_sha")
        box = manifest.get("box")
        if not isinstance(git_sha, str) or not git_sha:
            raise ArtifactBuildError(f"{leg_dir.name}: manifest.json has no string git_sha")
        if not isinstance(box, str) or not box:
            raise ArtifactBuildError(f"{leg_dir.name}: manifest.json has no string box")
        seen[f"leg:{leg_dir.name}"] = (git_sha, box)
    if p2_dir is not None:
        if not p2_dir.is_dir():
            raise ArtifactBuildError(f"--p2-dir {p2_dir} is not a directory")
        for tower_dir in sorted((p for p in p2_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            manifest_path = tower_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = _load_json(manifest_path, f"p2/{tower_dir.name}: manifest.json")
            git_sha = manifest.get("git_sha")
            box = manifest.get("box")
            if not isinstance(git_sha, str) or not git_sha:
                raise ArtifactBuildError(f"p2/{tower_dir.name}: manifest.json has no string git_sha")
            if not isinstance(box, str) or not box:
                raise ArtifactBuildError(f"p2/{tower_dir.name}: manifest.json has no string box")
            seen[f"p2:{tower_dir.name}"] = (git_sha, box)
    if not seen:
        raise ArtifactBuildError("no manifest.json found under --legs-dir (or --p2-dir) to witness git_sha/box from")
    distinct = set(seen.values())
    if len(distinct) != 1:
        detail = "; ".join(f"{k}={v}" for k, v in sorted(seen.items()))
        raise ArtifactBuildError(f"manifests disagree on (git_sha, box) — not one measurement session: {detail}")
    (git_sha, box) = next(iter(distinct))
    return git_sha, box


# --------------------------------------------------------------------------- #
# checkpoint sha256 per tower family — cross-checked (again) across legs
# --------------------------------------------------------------------------- #
_TOWER_FAMILY = {
    "clip-text": "clip",
    "clip-vision": "clip",
    "htsat": "clap",
}


def collect_checkpoint_sha256(merge_legs: list[dict]) -> dict[str, str]:
    by_family: dict[str, set[str]] = {}
    for leg in merge_legs:
        tower = leg.get("tower")
        family = _TOWER_FAMILY.get(tower)
        sha = leg.get("checkpoint_weights_sha256")
        if family is None or not isinstance(sha, str) or not sha:
            continue
        by_family.setdefault(family, set()).add(sha)
    out: dict[str, str] = {}
    for family, shas in by_family.items():
        if len(shas) != 1:
            raise ArtifactBuildError(
                f"checkpoint family {family!r} has disagreeing checkpoint_weights_sha256 across legs: {sorted(shas)}"
            )
        out[family] = next(iter(shas))
    return out


# --------------------------------------------------------------------------- #
# legs — merge + attribution + census launches_per_step, combined per leg id
# --------------------------------------------------------------------------- #
LEG_FIELDS_FROM_MERGE = (
    "tower",
    "task",
    "dtype",
    "arm",
    "kernels_disabled",
    "positive_proof",
    "per_step",
    "checkpoint_weights_sha256",
    "fusible_site_census",
)
LEG_FIELDS_FROM_ATTRIBUTION = (
    "signatures",
    "chains",
    "unknown_kernels",
    "decision_grade",
    "decision_grade_reason",
    "outside_signature_plausibly_attention",
)


def _index_by_leg_id(rows: list[dict]) -> dict[str, dict]:
    return {row["leg_id"]: row for row in rows if isinstance(row, dict) and "leg_id" in row}


def build_legs(merge_report: dict, attribution_report: dict, legs_dir: Path) -> list[dict]:
    merge_by_id = _index_by_leg_id(merge_report.get("legs", []))
    attr_by_id = _index_by_leg_id(attribution_report.get("legs", []))
    legs_dir_ids = {p.name for p in _leg_dirs(legs_dir)}

    merge_ids, attr_ids = set(merge_by_id), set(attr_by_id)
    if merge_ids != legs_dir_ids:
        raise ArtifactBuildError(
            "--merge-json's leg set does not match --legs-dir's leg set: "
            f"only in merge={sorted(merge_ids - legs_dir_ids)}, only in legs-dir={sorted(legs_dir_ids - merge_ids)}"
        )
    if attr_ids != legs_dir_ids:
        raise ArtifactBuildError(
            "--attribution-json's leg set does not match --legs-dir's leg set: "
            f"only in attribution={sorted(attr_ids - legs_dir_ids)}, only in legs-dir={sorted(legs_dir_ids - attr_ids)}"
        )

    legs: list[dict] = []
    for leg_dir in _leg_dirs(legs_dir):
        leg_id = leg_dir.name
        merge_leg = merge_by_id[leg_id]
        attr_leg = attr_by_id[leg_id]
        census = _load_json(leg_dir / "census.json", f"{leg_id}: census.json")
        launches = _finite(census.get("launches_per_step"), f"{leg_id}: census.json launches_per_step")

        row: dict[str, object] = {"leg_id": leg_id}
        for field in LEG_FIELDS_FROM_MERGE:
            row[field] = merge_leg.get(field)
        row["merge_verdict"] = merge_leg.get("verdict")
        row["merge_reasons"] = merge_leg.get("reasons")
        for field in LEG_FIELDS_FROM_ATTRIBUTION:
            row[field] = attr_leg.get(field)
        row["attribution_verdict"] = attr_leg.get("verdict")
        row["attribution_reasons"] = attr_leg.get("reasons")
        row["launches_per_step"] = launches
        legs.append(row)
    return legs


# --------------------------------------------------------------------------- #
# findings — computed sentences, each carrying the raw evidence it was built
# from (contract close-out: HTSAT front-end, CLIP-vision front-end, CLIP
# launch-bound + BF16 busy/wall trade, C-ATTN-HTSAT out-of-tier number)
# --------------------------------------------------------------------------- #
def _per_step(merge_legs: list[dict], leg_id: str) -> dict:
    row = _index_by_leg_id(merge_legs).get(leg_id)
    if row is None or not isinstance(row.get("per_step"), dict):
        raise ArtifactBuildError(f"{leg_id}: no per_step block in --merge-json to build a finding from")
    return row["per_step"]


def _per_step_finite(merge_legs: list[dict], leg_id: str, field: str) -> float:
    return _finite(_per_step(merge_legs, leg_id).get(field), f"{leg_id}: per_step.{field}")


def _chain(attribution_legs: list[dict], leg_id: str, chain: str) -> dict:
    row = _index_by_leg_id(attribution_legs).get(leg_id)
    if row is None or chain not in row.get("chains", {}):
        raise ArtifactBuildError(f"{leg_id}: no {chain!r} chain in --attribution-json to build a finding from")
    return row["chains"][chain]


def compute_findings(merge_report: dict, attribution_report: dict, legs_dir: Path) -> list[dict]:
    merge_legs = merge_report.get("legs", [])
    attribution_legs = attribution_report.get("legs", [])
    findings: list[dict] = []

    # 1) HTSAT front-end share of wall (A1 f32, A2 bf16 decision legs).
    htsat_shares = {
        leg_id: _per_step_finite(merge_legs, leg_id, "front_share_of_wall") * 100.0 for leg_id in ("htsat-A1", "htsat-A2")
    }
    lo, hi = min(htsat_shares.values()), max(htsat_shares.values())
    findings.append(
        {
            "id": "htsat-front-end-bound",
            "text": (
                f"The HTSAT training step is CPU front-end-bound: front-end share of wall is "
                f"{lo:.0f}-{hi:.0f}% across the F32/BF16 decision legs (audio decode/resample/STFT/mel "
                "dominating wall time), dtype- and arm-invariant."
            ),
            "evidence": {"front_share_of_wall_pct": htsat_shares},
        }
    )

    # 2) CLIP-vision front-end share of wall (A1 f32, A2 bf16 decision legs).
    vision_shares = {
        leg_id: _per_step_finite(merge_legs, leg_id, "front_share_of_wall") * 100.0
        for leg_id in ("clip-vision-A1", "clip-vision-A2")
    }
    lo, hi = min(vision_shares.values()), max(vision_shares.values())
    findings.append(
        {
            "id": "clip-vision-front-end-share",
            "text": (
                f"CLIP-vision's image decode/preprocess front end is {lo:.0f}-{hi:.0f}% of wall on the "
                "F32/BF16 decision legs."
            ),
            "evidence": {"front_share_of_wall_pct": vision_shares},
        }
    )

    # 3) CLIP launch-bound at batch 8, + the BF16 busy/wall trade.
    launch_legs = ("clip-text-A1", "clip-text-A2", "clip-vision-A1", "clip-vision-A2")
    legs_by_id = _index_by_leg_id([{"leg_id": p.name} for p in _leg_dirs(legs_dir)])
    launches: dict[str, float] = {}
    for leg_id in launch_legs:
        if leg_id not in legs_by_id:
            raise ArtifactBuildError(f"{leg_id}: not present under --legs-dir; cannot build the launch-bound finding")
        census = _load_json(legs_dir / leg_id / "census.json", f"{leg_id}: census.json")
        launches[leg_id] = _finite(census.get("launches_per_step"), f"{leg_id}: census.json launches_per_step")
    busy_deltas_pct: dict[str, float] = {}
    wall_deltas_pct: dict[str, float] = {}
    for tower, a1, a2 in (("clip-text", "clip-text-A1", "clip-text-A2"), ("clip-vision", "clip-vision-A1", "clip-vision-A2")):
        busy_a1 = _per_step_finite(merge_legs, a1, "busy_s_per_step")
        busy_a2 = _per_step_finite(merge_legs, a2, "busy_s_per_step")
        wall_a1 = _per_step_finite(merge_legs, a1, "wall_s_per_step")
        wall_a2 = _per_step_finite(merge_legs, a2, "wall_s_per_step")
        busy_deltas_pct[tower] = (busy_a2 - busy_a1) / busy_a1 * 100.0
        wall_deltas_pct[tower] = (wall_a2 - wall_a1) / wall_a1 * 100.0
    launch_lo, launch_hi = min(launches.values()), max(launches.values())
    # Both deltas are negative (BF16 is cheaper): order the range by
    # ascending MAGNITUDE ("-32...-41%", smallest shrink first), not by
    # ascending signed value, so the sentence reads as a growing effect —
    # `sorted(..., key=abs)` rather than a plain `min`/`max` pick.
    busy_by_magnitude = sorted(busy_deltas_pct.values(), key=abs)
    wall_by_magnitude = sorted(wall_deltas_pct.values(), key=abs)
    findings.append(
        {
            "id": "clip-launch-bound-batch8",
            "text": (
                f"At batch 8 the CLIP training steps are launch-bound: {launch_lo:.0f}-{launch_hi:.0f} "
                "launches/step across the four F32/BF16 A-arm CLIP legs (text and vision); switching to "
                f"BF16 cuts GPU busy {busy_by_magnitude[0]:.0f}...{busy_by_magnitude[-1]:.0f}% per tower "
                f"while wall drops only {wall_by_magnitude[0]:.0f}...{wall_by_magnitude[-1]:.0f}%."
            ),
            "evidence": {
                "launches_per_step": launches,
                "busy_delta_pct_bf16_vs_f32": busy_deltas_pct,
                "wall_delta_pct_bf16_vs_f32": wall_deltas_pct,
            },
        }
    )

    # 4) C-ATTN-HTSAT: measured, out-of-tier (declared out of scope for a
    #    port decision under this contract — a NUMBER, never a verdict).
    c_attn_htsat = _chain(attribution_legs, "htsat-A1", "C-ATTN-htsat")
    share_gpu_busy = _finite(c_attn_htsat.get("share_gpu_busy"), "htsat-A1: chains['C-ATTN-htsat'].share_gpu_busy")
    share_wall = _finite(c_attn_htsat.get("share_wall"), "htsat-A1: chains['C-ATTN-htsat'].share_wall")
    busy_pct = share_gpu_busy * 100.0
    wall_pct = share_wall * 100.0
    findings.append(
        {
            "id": "c-attn-htsat-out-of-tier",
            "text": (
                f"C-ATTN-HTSAT is measured, not a candidate port: {busy_pct:.0f}% of GPU busy "
                f"(~{wall_pct:.0f}% of wall) on the F32 decision leg (htsat-A1). HTSAT attention "
                "(head_dim 24 at every stage) sits OUTSIDE the fixed-head-dim port tier by the "
                "contract's own declaration — this stays a measured, OPEN number, never folded into "
                "UNATTRIBUTED and never decided under this issue."
            ),
            "evidence": {"leg_id": "htsat-A1", "share_gpu_busy": share_gpu_busy, "share_wall": share_wall},
        }
    )

    return findings


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #
def build_report(
    legs_dir: Path,
    p2_dir: Path | None,
    merge_report: dict,
    attribution_report: dict,
    identity: dict,
) -> dict:
    git_sha, box = collect_identity(legs_dir, p2_dir)
    legs = build_legs(merge_report, attribution_report, legs_dir)
    merge_legs = merge_report.get("legs", [])
    checkpoint_sha256 = collect_checkpoint_sha256(merge_legs)

    checkpoints_notes: dict[str, dict] = {}
    for family, block in identity.get("checkpoints", {}).items():
        entry = dict(block)
        if family in checkpoint_sha256:
            entry["sha256"] = checkpoint_sha256[family]
        checkpoints_notes[family] = entry

    findings = compute_findings(merge_report, attribution_report, legs_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "box": box,
        "producer": {
            "path": "ci/scripts/perf/profile_421_legs.sh",
            "kind": "script",
            "invocation": identity.get("producer_invocation"),
            "gating": "none",
        },
        "status": identity.get("status", "GREEN"),
        "notes": {
            "what": identity.get("what"),
            "gpu": identity.get("gpu"),
            "driver": identity.get("driver"),
            "cpu": identity.get("cpu"),
            "nsys": identity.get("nsys_human"),
            "checkpoints": checkpoints_notes,
            "recorded_deviations": identity.get("recorded_deviations", []),
        },
        "legs": legs,
        "p2": merge_report.get("p2_bf16", []),
        "attribution": attribution_report.get("legs", []),
        "realized_gains": attribution_report.get("realized_gains", []),
        "candidate_decisions": attribution_report.get("candidate_decisions", []),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_artifact.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", required=True, help="a profile_421_legs.sh $OUT_DIR/legs to read manifests/census from")
    ap.add_argument("--p2-dir", help="the PROFILE_421_P2_BF16=1 output dir ($OUT_DIR/p2-bf16), for the git_sha/box cross-check")
    ap.add_argument("--merge-json", required=True, help="profile_421_merge.py's own output over the SAME --legs-dir")
    ap.add_argument("--attribution-json", required=True, help="profile_421_attribute.py's own output over the SAME --legs-dir")
    ap.add_argument("--identity", required=True, help="the non-numeric identity/notes sidecar (see profile_421_run2_identity.json)")
    ap.add_argument("--out", help="write the artifact here (default: stdout)")
    args = ap.parse_args(argv)

    legs_dir = Path(args.legs_dir)
    p2_dir = Path(args.p2_dir) if args.p2_dir else None
    try:
        merge_report = _load_json(Path(args.merge_json), "--merge-json")
        attribution_report = _load_json(Path(args.attribution_json), "--attribution-json")
        identity = _load_json(Path(args.identity), "--identity")
        report = build_report(legs_dir, p2_dir, merge_report, attribution_report, identity)
    except ArtifactBuildError as exc:
        print(f"::error::profile_421_artifact: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(report, indent=1, sort_keys=False)
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::profile_421_artifact: could not write {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)

    print(
        f"profile_421_artifact: git_sha={report['git_sha']} box={report['box']} "
        f"legs={len(report['legs'])} p2={len(report['p2'])} findings={len(report['findings'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
