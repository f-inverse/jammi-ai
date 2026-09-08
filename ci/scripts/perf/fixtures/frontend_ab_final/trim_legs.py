#!/usr/bin/env python3
"""Trims a FULL `jammi-bench finetune-run` raw-leg directory (the pod's own
`--work-dir` reports, one per `<tower>__<role>__<repeat>` leg) down to the
envelope `frontend_ab_artifact.py`'s own reader (`_read_raw_leg`) actually
reads, for the committed `fixtures/frontend_ab_final/` close-out fixture.

Committed so a future regeneration of this fixture is a RUN of this
script against the pod's own untrimmed sources, never a hand-rolled dict
or a one-off ad hoc script that leaves no trace of how the committed bytes
were produced.

## Kept fields (see this directory's own PROVENANCE.md)

`host.logical_cpus`, `provenance.build_sha`, and
`tiers.finetune_run.{steps_measured, media_front_end_wall_s,
train_run_wall_s, task, device_name, rayon_pool_threads}` (the last key
present only on `tip` legs -- a real pre-#421-follow-on `base` binary
never emits it). `task`/`device_name` are kept (not just the four
timing/thread fields the original cut carried) so
`frontend_ab_artifact.py`'s own per-leg cross-checks -- tower-vs-task,
box-vs-device_name -- have something real to check against, rather than
trusting the filename alone.

A REAL `jammi-bench finetune-run` report carries ~90 OTHER
`tiers.finetune_run` fields (checkpoint identity hashes, target modules,
per-dispatch-site fused/eager counters, LoRA hyperparameters, etc.) this
reader has no business asserting on -- `jammi-bench`'s own generic report
shape, not a specific consumer's data shape, is what this driver's own bar
arithmetic is contracted against.

## `--check`

Re-trims from an UNTRIMMED source raw-leg directory (never committed to
this repo -- it carries the full report) and asserts BYTE-IDENTITY,
`.json` and `.exit` both, against this fixture's own committed `raw/`
files, WITHOUT writing anything. This mode is runnable only where the
untrimmed sources still exist (a pod's own scratch directory) -- CI never
has them, which is exactly why `test_frontend_ab_artifact.py`'s
`RealFixtureRegressionTests` instead covers the COMMITTED bytes end to
end, the only thing CI ever sees.

Run (regenerate the committed fixture from a fresh untrimmed pod dir):
    python3 trim_legs.py --src-raw <untrimmed raw dir> --dst raw

Run (verify the committed fixture still matches its own untrimmed source):
    python3 trim_legs.py --src-raw <untrimmed raw dir> --dst raw --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOWERS = ("htsat", "clip-vision")
ROLES = ("base", "tip")

FIXTURE_DIR = Path(__file__).resolve().parent


class TrimError(Exception):
    """A source leg is missing an expected field -- fails closed, never
    guesses a value the untrimmed report did not actually carry."""


def _trim_leg(src_json: Path, src_exit: Path) -> tuple[bytes, bytes]:
    full = json.loads(src_json.read_text(encoding="utf-8"))
    try:
        host = full["host"]
        provenance = full["provenance"]
        tier = full["tiers"]["finetune_run"]
        trimmed_tier = {
            "steps_measured": tier["steps_measured"],
            "media_front_end_wall_s": tier["media_front_end_wall_s"],
            "train_run_wall_s": tier["train_run_wall_s"],
            "task": tier["task"],
            "device_name": tier["device_name"],
        }
        if "rayon_pool_threads" in tier:
            trimmed_tier["rayon_pool_threads"] = tier["rayon_pool_threads"]
        trimmed = {
            "host": {"logical_cpus": host["logical_cpus"]},
            "provenance": {"build_sha": provenance["build_sha"]},
            "tiers": {"finetune_run": trimmed_tier},
        }
    except KeyError as exc:
        raise TrimError(f"{src_json}: untrimmed source is missing expected key {exc}") from exc

    json_bytes = (json.dumps(trimmed, indent=2, sort_keys=False) + "\n").encode("utf-8")
    exit_bytes = (src_exit.read_text(encoding="utf-8").strip() + "\n").encode("utf-8")
    return json_bytes, exit_bytes


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="trim_legs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-raw", required=True, help="the UNTRIMMED jammi-bench finetune-run raw-leg directory")
    ap.add_argument("--dst", default=str(FIXTURE_DIR / "raw"), help="destination raw dir (default: this fixture's own committed raw/)")
    ap.add_argument("--repeats", type=int, default=3, help="repeat count (default: 3, this fixture's own FRONTEND_AB_REPEATS)")
    ap.add_argument(
        "--check", action="store_true",
        help="re-trim into memory and assert byte-identity against --dst's existing files; never writes",
    )
    args = ap.parse_args(argv)

    if args.repeats < 1:
        print(f"::error::trim_legs: --repeats must be a positive int, got {args.repeats!r}", file=sys.stderr)
        return 1

    src_raw = Path(args.src_raw)
    dst = Path(args.dst)
    repeats = [f"r{i}" for i in range(1, args.repeats + 1)]

    mismatches: list[str] = []
    written = 0
    for tower in TOWERS:
        for role in ROLES:
            for repeat in repeats:
                stem = f"{tower}__{role}__{repeat}"
                src_json = src_raw / f"{stem}.json"
                src_exit = src_raw / f"{stem}.exit"
                if not src_json.is_file() or not src_exit.is_file():
                    print(f"::error::trim_legs: missing source leg {stem!r} under {src_raw}", file=sys.stderr)
                    return 1

                try:
                    json_bytes, exit_bytes = _trim_leg(src_json, src_exit)
                except TrimError as exc:
                    print(f"::error::trim_legs: {exc}", file=sys.stderr)
                    return 1

                dst_json = dst / f"{stem}.json"
                dst_exit = dst / f"{stem}.exit"
                if args.check:
                    for path, expected, src in ((dst_json, json_bytes, src_json), (dst_exit, exit_bytes, src_exit)):
                        if not path.is_file():
                            mismatches.append(f"{path}: not present (a fresh trim of {src} would write it)")
                            continue
                        actual = path.read_bytes()
                        if actual != expected:
                            mismatches.append(f"{path}: committed bytes differ from a fresh trim of {src}")
                else:
                    dst.mkdir(parents=True, exist_ok=True)
                    dst_json.write_bytes(json_bytes)
                    dst_exit.write_bytes(exit_bytes)
                    written += 1

    if args.check:
        if mismatches:
            for mismatch in mismatches:
                print(f"::error::trim_legs --check: {mismatch}", file=sys.stderr)
            return 1
        print(
            f"trim_legs --check: {len(TOWERS) * len(ROLES) * len(repeats)} legs byte-identical to {src_raw}",
            file=sys.stderr,
        )
        return 0

    print(f"trim_legs: wrote {written} legs to {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
