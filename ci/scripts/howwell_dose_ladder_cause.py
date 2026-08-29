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
of `(cause_name, triggered, message)` tuples, checked at runtime (an
explicit `if`/`raise AssertionError` -- unit-63 round-15 audit advisory 4:
never a bare `assert`, which `python -O` strips entirely) to name exactly
`ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES`. `_ALL_CAUSE_NAMES` below is that
SAME constant, imported directly (never a hand-duplicated literal) --
`DoseLadderCauseNamesBoundToAbMergeExitFoldTests`
(`test_howwell_dose_ladder_cause.py`) imports both this module and
`ab_merge` and asserts `set(namer._ALL_CAUSE_NAMES) ==
set(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES)` as an explicit, executable
cross-module pin -- this test-side binding is the PRIMARY enforcement (it
runs on every commit regardless of `-O`); `ab_merge.py`'s own runtime check
is defense-in-depth for the one process that actually folds
`dose_ladder_causes` at merge time. What this GUARANTEES: a fifth cause
added to `ab_merge.py`'s own exit fold without a matching entry in
`DOSE_LADDER_EXIT_CAUSE_NAMES` fails `ab_merge.py`'s own internal runtime
check (`main()`'s dose-ladder branch) the first time that code path runs
(under `-O` or not), AND (since this module imports that same constant)
this namer's own fallback text grows to match automatically -- there is no
third, independently-drifting copy of the cause-name set left anywhere in
this pairing. What it does NOT guarantee: a literal fifth `if`/branch
written OUTSIDE that data-driven fold (bypassing `dose_ladder_causes`
entirely) would still need its own review to be caught -- this binding
covers the one fold both sides already commit to, never arbitrary future
code shape.
"""

from __future__ import annotations

import json
import os
import sys

# Unit-63 round-15 audit advisory 3: this insert+import is itself a crash
# surface upstream of `_inspect_doses`'s own A4 hardening -- an import-time
# failure here (a syntax error introduced into `perf/ab_merge.py`, a missing
# `perf/` directory, or -- the shadowing risk -- some OTHER `ab_merge` module
# earlier on `sys.path` that this `insert(0, ...)` does NOT protect against
# if this file is ever copied/executed from a location where `os.path.
# dirname(__file__)` no longer resolves to `ci/scripts`) must degrade to a
# NAMED cause here, never an uncaught `ImportError`/`SyntaxError` that
# propagates out of module load. Left uncaught, `runpod_gpu_howwell.sh`'s own
# `2>/dev/null || echo "unknown (could not inspect ...)"` wrapper (this
# module's own doc above) swallows the SPECIFIC failure into the same opaque
# "unknown" text a truly-no-cause-found run also produces -- the exact
# unexplained-contradiction shape `_inspect_doses` exists to prevent one
# layer down, now recurring one layer up. Note on the shadowing risk named
# above: `sys.path.insert(0, ...)` puts THIS `perf/` directory first, so it
# always wins over anything a caller's `PYTHONPATH` places earlier in
# `sys.path` -- the risk is the opposite direction, an `ab_merge` module a
# caller intended to be picked up from elsewhere on `sys.path` being silently
# shadowed by this repo's own `perf/ab_merge.py`, never the reverse.
_PERF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf")
sys.path.insert(0, _PERF_DIR)
try:
    import ab_merge  # noqa: E402

    _AB_MERGE_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - exercised via subprocess in test_howwell_dose_ladder_cause.py
    ab_merge = None  # type: ignore[assignment]
    _AB_MERGE_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# Unit-63 round-14 audit F6: imported directly from `ab_merge.py`, never a
# hand-duplicated literal -- see this module's own "Binding to ab_merge.py's
# own exit fold" doc above for exactly what this import makes impossible
# (a fifth cause silently added to one side alone) versus what it does not
# (an exit-fold branch written outside the shared data structure). When the
# import above failed, there is no `ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES` to
# read -- `_ALL_CAUSE_NAMES` is left empty; `dose_ladder_cause` below never
# reaches the code that would consult it in that state (the import-failure
# cause short-circuits first).
_ALL_CAUSE_NAMES = list(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES) if ab_merge is not None else []

# Unit-63 round-16 audit (identity-completeness, sibling class): the dose
# column `detected` vocabulary and the `red_proof_verdict` NOT_PROVEN prefix
# are `ab_merge.py`'s own producer-side constants, imported directly here --
# never a re-typed literal -- exactly as `DOSE_LADDER_EXIT_CAUSE_NAMES` is
# above. When the import above failed, there is nothing to read; the
# import-failure cause short-circuits `dose_ladder_cause` before either
# constant is ever consulted (same discipline as `_ALL_CAUSE_NAMES` above).
_MUTANT_DOSE_DETECTED_INVALID = ab_merge.MUTANT_DOSE_DETECTED_INVALID if ab_merge is not None else None
_RED_PROOF_VERDICT_NOT_PROVEN_PREFIX = (
    ab_merge.RED_PROOF_VERDICT_NOT_PROVEN_PREFIX if ab_merge is not None else None
)


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

    Unit-63 round-15 audit advisory 3: if the module-level `import ab_merge`
    itself failed, `_AB_MERGE_IMPORT_ERROR` is non-`None` and this function
    returns that failure as its own named cause immediately, before touching
    the ALREADY-PARSED `report` dict passed in here at all -- same discipline
    as `_inspect_doses`, degrading a crash surface into legible text rather
    than letting it propagate to an uncaught exception that
    `runpod_gpu_howwell.sh`'s own wrapper would collapse into the opaque
    "unknown (could not inspect ...)" text. This function's own contract
    starts AFTER `report` has already been read and `json.loads`-parsed by
    the caller -- reading/parsing the report FILE is `main()`'s own job (see
    that function's own doc, unit-63 round-16 audit advisory 3, for the
    file-read hardening this function does not itself provide).
    """
    if _AB_MERGE_IMPORT_ERROR is not None:
        return f"ab_merge_import_failed({_AB_MERGE_IMPORT_ERROR})"
    ladder = report.get("mutant_dose_ladder") or {}
    causes = []
    if ladder.get("sensitivity_error"):
        causes.append("sensitivity_error")
    doses, doses_malformation = _inspect_doses(ladder)
    if doses_malformation is not None:
        causes.append(doses_malformation)
    invalid_doses = [c.get("dose_label") for c in doses if c.get("detected") == _MUTANT_DOSE_DETECTED_INVALID]
    if invalid_doses:
        causes.append("invalid_doses=" + ",".join(str(d) for d in invalid_doses))
    if ladder.get("dose_anomalies"):
        causes.append("dose_anomalies")
    red_proof_verdict = ladder.get("red_proof_verdict")
    if isinstance(red_proof_verdict, str) and red_proof_verdict.startswith(_RED_PROOF_VERDICT_NOT_PROVEN_PREFIX):
        causes.append(f"red_proof_verdict={red_proof_verdict}")
    if causes:
        return ",".join(causes)
    checked = "/".join(_ALL_CAUSE_NAMES)
    return f"unknown (no {checked} found)"


def main(argv=None) -> int:
    """Unit-63 round-16 audit advisory 3 (correcting `dose_ladder_cause`'s
    own docstring, which was read as claiming MORE than it does): opening
    and `json.loads`-parsing `REPORT_JSON_PATH` is THIS function's own job,
    not `dose_ladder_cause`'s -- and it used to happen entirely OUTSIDE the
    named-degradation discipline that function provides, an unreadable file
    (missing, permission-denied, a directory, ...) or malformed JSON crashed
    straight into `runpod_gpu_howwell.sh`'s own
    `2>/dev/null || echo "unknown (could not inspect ...)"` wrapper -- the
    exact opaque-collapse shape `_AB_MERGE_IMPORT_ERROR`/`_inspect_doses`
    exist to prevent one layer down, recurring one layer up, and reachable
    even when `ab_merge` itself imported cleanly (a broken `ab_merge` and an
    unreadable report are independent failure axes; this hardening covers
    the report-read axis regardless of the other). The read+parse is now
    INSIDE the same discipline: a read failure or a JSON parse failure
    degrades to its own NAMED cause on stdout, exit 0 -- never an uncaught
    exception, and never silently folded into the generic
    "unknown (could not inspect ...)" text a genuinely-no-cause-found run
    also produces.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: howwell_dose_ladder_cause.py REPORT_JSON_PATH", file=sys.stderr)
        return 2
    report_path = argv[0]
    try:
        with open(report_path) as fh:
            report_text = fh.read()
    except OSError as exc:
        print(f"report_unreadable({type(exc).__name__}: {exc})")
        return 0
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        print(f"report_malformed_json({exc})")
        return 0
    print(dose_ladder_cause(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
