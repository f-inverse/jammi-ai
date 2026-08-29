#!/usr/bin/env python3
"""GREEN-but-nonzero cause namer for `runpod_gpu_howwell.sh` (unit-63 round-10
audit advisory (d) / round-13 audit F1) -- extracted out of that script's own
inline python heredoc into a real, testable module, mirroring this repo's own
`check_X.py` + `test_X.py` gate-suite convention (see `test_check_kernel_
oracles.py`'s own doc for the shape).

## Why this exists

A GREEN primary A/B decision does NOT itself force `ab_merge.py`'s own exit
code to 0 -- the mutant dose ladder (an INVALID dose column, a negative-eps
`dose_anomaly`, a `sensitivity_error`, or -- unit-63 round-13 audit F1 -- an
undischarged RED-proof column, `red_proof_verdict` starting with
`"NOT_PROVEN"`, `ab_merge.py`'s own exit fold at its `main()`'s dose-ladder
branch) can still fail the merge while the primary decision itself reads
GREEN. `runpod_gpu_howwell.sh` names the actual cause BY NAME on exactly this
shape, so a GREEN-but-nonzero run is legible outside the collapsed log group
instead of looking like the unexplained-contradiction this namer exists to
prevent.

`dose_ladder_cause` enumerates ALL FOUR of `ab_merge.py`'s own dose-ladder
exit-code causes (`sensitivity_error`, `dose_anomalies`, an `INVALID` dose
column, `red_proof_verdict` NOT_PROVEN) -- never a subset -- and the fallback
"unknown" text names every one it checked, so an operator reading the
fallback text can see exactly what was ruled out rather than a bare "unknown"
that looks like this namer forgot a cause.
"""

from __future__ import annotations

import json
import sys

# Kept as one literal, named list -- `dose_ladder_cause`'s own fallback text
# is built FROM this list (never hand-duplicated) so the two can never drift
# apart when a fifth cause is added to `ab_merge.py`'s own exit fold.
_ALL_CAUSE_NAMES = ["dose_anomalies", "sensitivity_error", "invalid dose column", "red_proof_verdict"]


def dose_ladder_cause(report: dict) -> str:
    """Names every `ab_merge.py` dose-ladder exit-code cause present in
    `report` (an already-parsed `finetune_run_ab_report.json`), comma-joined
    in the SAME order `ab_merge.py`'s own `main()` folds them into
    `exit_code` (`sensitivity_error`, then INVALID doses, then
    `dose_anomalies`, then `red_proof_verdict`) -- see that function's own
    dose-ladder branch. When none of the four causes is present, returns a
    fallback string that enumerates all four checked classes by name (never
    a bare "unknown"), so a truly unexplained GREEN-but-nonzero contradiction
    is still legible as "none of these four" rather than silently opaque.
    """
    ladder = report.get("mutant_dose_ladder") or {}
    causes = []
    if ladder.get("sensitivity_error"):
        causes.append("sensitivity_error")
    invalid_doses = [c.get("dose_label") for c in ladder.get("doses", []) if c.get("detected") == "INVALID"]
    if invalid_doses:
        causes.append("invalid_doses=" + ",".join(invalid_doses))
    if ladder.get("dose_anomalies"):
        causes.append("dose_anomalies")
    red_proof_verdict = ladder.get("red_proof_verdict")
    if isinstance(red_proof_verdict, str) and red_proof_verdict.startswith("NOT_PROVEN"):
        causes.append(f"red_proof_verdict={red_proof_verdict}")
    if causes:
        return ",".join(causes)
    checked = "/".join(_ALL_CAUSE_NAMES)
    return f"unknown (no {checked} found)"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: howwell_dose_ladder_cause.py REPORT_JSON_PATH", file=sys.stderr)
        return 2
    with open(argv[0]) as fh:
        report = json.load(fh)
    print(dose_ladder_cause(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
