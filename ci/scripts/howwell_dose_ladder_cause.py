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

## Binding to `ab_merge.py`'s own exit fold (unit-63 round-14 audit F6)

`ab_merge.py`'s `main()` no longer hand-maintains four independent `if`
blocks for its `finetune-run` dose-ladder exit code -- it folds a DATA list
of `(cause_name, triggered, message)` tuples, asserted at runtime to name
exactly `ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES`. `_ALL_CAUSE_NAMES` below is
that SAME constant, imported directly (never a hand-duplicated literal) --
`DoseLadderCauseNamesBoundToAbMergeExitFoldTests`
(`test_howwell_dose_ladder_cause.py`) imports both this module and
`ab_merge` and asserts `set(namer._ALL_CAUSE_NAMES) ==
set(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES)` as an explicit, executable
cross-module pin. What this GUARANTEES: a fifth cause added to `ab_merge.
py`'s own exit fold without a matching entry in `DOSE_LADDER_EXIT_CAUSE_
NAMES` fails `ab_merge.py`'s own internal assertion (`main()`'s dose-ladder
branch) the first time that code path runs, AND (since this module imports
that same constant) this namer's own fallback text grows to match
automatically -- there is no third, independently-drifting copy of the
cause-name set left anywhere in this pairing. What it does NOT guarantee: a
literal fifth `if`/branch written OUTSIDE that data-driven fold (bypassing
`dose_ladder_causes` entirely) would still need its own review to be
caught -- this binding covers the one fold both sides already commit to,
never arbitrary future code shape.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf"))
import ab_merge  # noqa: E402

# Unit-63 round-14 audit F6: imported directly from `ab_merge.py`, never a
# hand-duplicated literal -- see this module's own "Binding to ab_merge.py's
# own exit fold" doc above for exactly what this import makes impossible
# (a fifth cause silently added to one side alone) versus what it does not
# (an exit-fold branch written outside the shared data structure).
_ALL_CAUSE_NAMES = list(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES)


def _inspect_doses(ladder: dict) -> tuple[list, str | None]:
    """Unit-63 round-14 audit A4: `ladder["doses"]` is a producer/merger
    artifact field, never assumed well-shaped by this namer -- a corrupted
    or hand-edited `finetune_run_ab_report.json` (`"doses": null`, `"doses"`
    not a list at all, or a list carrying a `null`/non-dict element) must
    degrade to a NAMED cause-inspection failure here, never an uncaught
    `TypeError`/`AttributeError` that propagates out of `dose_ladder_cause`
    -- `runpod_gpu_howwell.sh` catches exactly that crash shape with
    `2>/dev/null || echo "unknown (could not inspect ...)"`, which would
    silently swallow the SPECIFIC malformation into the same opaque
    "unknown" text a truly-no-cause-found run also produces, defeating this
    namer's own "never a bare unknown" purpose one layer up.

    Returns `(usable_doses, malformation_cause)`: `usable_doses` is the
    (possibly empty, possibly entry-filtered) list `dose_ladder_cause` can
    safely scan for `INVALID` entries; `malformation_cause` is `None` when
    `doses` was well-shaped (including simply absent/empty, never itself a
    malformation), or a short, named string identifying WHICH shape defect
    fired.
    """
    if "doses" not in ladder:
        return [], None
    raw = ladder.get("doses")
    if raw is None:
        return [], "doses_field_is_null"
    if not isinstance(raw, list):
        return [], f"doses_field_is_not_a_list(type={type(raw).__name__})"
    usable = []
    malformed_count = 0
    for entry in raw:
        if isinstance(entry, dict):
            usable.append(entry)
        else:
            malformed_count += 1
    if malformed_count:
        plural = "y" if malformed_count == 1 else "ies"
        return usable, f"doses_field_has_{malformed_count}_malformed_entr{plural}"
    return usable, None


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

    Unit-63 round-14 audit A4: `ladder["doses"]` itself is inspected via
    `_inspect_doses` before being scanned for `INVALID` entries -- a
    malformed `doses` field (`null`, not a list, or carrying a `null`/
    non-dict element) is named as its own cause rather than crashing this
    function outright (see that helper's own doc for why a crash here is
    strictly worse than a named "unknown").
    """
    ladder = report.get("mutant_dose_ladder") or {}
    causes = []
    if ladder.get("sensitivity_error"):
        causes.append("sensitivity_error")
    doses, doses_malformation = _inspect_doses(ladder)
    if doses_malformation is not None:
        causes.append(doses_malformation)
    invalid_doses = [c.get("dose_label") for c in doses if c.get("detected") == "INVALID"]
    if invalid_doses:
        causes.append("invalid_doses=" + ",".join(str(d) for d in invalid_doses))
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
