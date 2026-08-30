#!/usr/bin/env python3
"""Mechanically re-derives `ci/scripts/perf/gpu_inference_ab.py`'s own
`PRE_REGISTERED_ADVISORY_BAND` from the COMMITTED `ci/artifacts/gpu-perf-aa-null/`
evidence and asserts equality — hermetic, static, no build, no GPU, modeled
on `check_pod_build_timings.py`'s own shape (a real-tree run plus a
`--self-test` synthetic-fixture leg, wired into `ci.yml`'s Guard matrix
alongside that sibling gate).

## The class this closes (round-4 delta-audit F5)

`ci/scripts/perf/gpu_inference_ab.py`'s own module doc and
`ci/artifacts/gpu-perf-aa-null/README.md`'s own "Band derivation" section
both ASSERT, in prose, that `PRE_REGISTERED_ADVISORY_BAND = (0.75, 1.33)`
is what the committed empirical-null campaign's PRIMARY evidence derives to
— but nothing mechanically re-checked that claim before this gate. A PR
that edits either the committed artifacts, `manifest.json`'s own
primary/aux classification, or the band constant itself, without updating
the OTHER two in the same diff, would previously have gone GREEN on every
existing test suite (none of them read the committed `ci/artifacts/`
evidence at all) — this gate is what makes "never hand-tunes the two
numbers" (`gpu_inference_ab.sh`'s own doc) an ENFORCED property, not merely
an asserted one.

## What this gate checks

1. **Manifest presence + shape**: `ci/artifacts/gpu-perf-aa-null/manifest.json`
   exists, is a JSON object with a `runs` list, each entry carrying a
   `file` (string) and a `role` (`"primary"` or `"aux"`, nothing else).
2. **Per-file minimal schema**: every manifest-listed file exists under
   `ci/artifacts/gpu-perf-aa-null/`, parses as JSON, and carries `legs`,
   `adjacent_pair_ratios`, `mode`, and `recorded_order` top-level keys, with
   `mode == "aa-null"` — the shape `gpu_inference_ab.py::build_report`'s own
   `--aa-null` producer mode writes, never re-implementing that module's own
   parser, just checking the four keys a re-derivation needs are present
   and the record honestly claims to be a null-campaign run.
3. **Re-derivation**: over every `"primary"`-role file's own
   `adjacent_pair_ratios` values (both pairs, every primary file), computes
   the worst `|ln(ratio)|`, feeds it through
   `gpu_inference_ab.derive_advisory_band` (imported, NEVER
   re-implemented — the SAME rounding rule the constant itself was derived
   with), and asserts the result equals
   `gpu_inference_ab.PRE_REGISTERED_ADVISORY_BAND` exactly.

4. **Directory completeness** (round-2 delta-audit B4): every `*.json`
   under `ci/artifacts/gpu-perf-aa-null/` except `manifest.json` itself
   must be LISTED in the manifest — an unlisted report is invisible to the
   re-derivation, which is exactly the escape hatch this check closes
   (drop a report from the manifest and the gate would otherwise stay
   green no matter what its ratios say).
5. **Primary start-window disjointness** (round-2 delta-audit B6, the
   mechanized form of the very evidence that demoted `pcie-p1`/`p2`):
   every `"primary"`-role run's `recorded_order` start-window
   (`[min, max]` over its four leg-start timestamps) must not overlap ANY
   other listed run's window. Overlapping start-windows are POSITIVE
   evidence of concurrent invocations (the `p1`/`p2` signature); disjoint
   start-windows are NECESSARY but not SUFFICIENT for isolation (a run's
   final leg extends past its start timestamp, and two clean runs on two
   different pods also have disjoint windows) — which is why the
   manifest's own `reason` strings claim only what this check plus the
   operator's session records can actually establish, never a bare
   "clean" assertion.

`"aux"`-role files are loaded and schema-checked (so a bit-rotted aux
artifact still fails loudly) but their `adjacent_pair_ratios` are EXCLUDED
from the re-derivation — the whole point of the primary/aux split. Two
aux-role runs MAY overlap each other (that is precisely why `pcie-p1`/`p2`
are aux); only a PRIMARY overlapping anything is a finding.

Run: `python3 ci/scripts/check_aa_null_band.py` (real tree) or
`python3 ci/scripts/check_aa_null_band.py --self-test` (synthetic fixtures,
proving the re-derivation/exclusion/schema-violation logic itself, never
touching the real committed artifacts).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
assert (REPO_ROOT / "Cargo.toml").is_file(), (
    f"REPO_ROOT resolved to {REPO_ROOT}, which has no Cargo.toml -- this script's own path-depth "
    f"assumption (ci/scripts/<file>.py -> parents[2] == repo root) is wrong; see esc-063's own class "
    f"(check_producer_provenance_gates.py's identical off-by-one) for why this assertion exists"
)
AA_NULL_DIR = REPO_ROOT / "ci" / "artifacts" / "gpu-perf-aa-null"
MANIFEST_PATH = AA_NULL_DIR / "manifest.json"

PERF_DIR = REPO_ROOT / "ci" / "scripts" / "perf"
sys.path.insert(0, str(PERF_DIR))
import gpu_inference_ab  # noqa: E402 -- the REAL derive_advisory_band/PRE_REGISTERED_ADVISORY_BAND, never re-implemented.

REQUIRED_REPORT_KEYS = ("legs", "adjacent_pair_ratios", "mode", "recorded_order")
VALID_ROLES = ("primary", "aux")


class BandGateError(Exception):
    """A schema or manifest-shape defect — a FINDING, distinct from "the
    re-derived band does not match" (which `main`/`self_test` report
    separately, as a mismatch rather than an exception, since it is the
    ONE finding this whole gate exists to surface loudly)."""


def load_manifest(aa_null_dir: Path) -> list[dict]:
    """Loads `<aa_null_dir>/manifest.json`'s own `runs` list, validating its
    OWN shape (never trusting a malformed manifest silently) — raises
    `BandGateError` naming exactly what is wrong, never a bare
    `KeyError`/`TypeError` traceback.
    """
    manifest_path = aa_null_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BandGateError(f"no manifest.json found at {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BandGateError(f"{manifest_path} does not parse as JSON: {exc}") from exc
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise BandGateError(f"{manifest_path}'s own 'runs' field must be a non-empty list, got {runs!r}")
    for entry in runs:
        if not isinstance(entry, dict) or "file" not in entry or "role" not in entry:
            raise BandGateError(f"{manifest_path}: every 'runs' entry needs 'file' and 'role', got {entry!r}")
        if entry["role"] not in VALID_ROLES:
            raise BandGateError(
                f"{manifest_path}: entry {entry['file']!r} has role {entry['role']!r}, not one of {VALID_ROLES}"
            )
    return runs


def load_and_validate_report(aa_null_dir: Path, filename: str) -> dict:
    """Loads `<aa_null_dir>/<filename>`, validates the minimal schema this
    gate's own re-derivation needs (`REQUIRED_REPORT_KEYS` present, `mode
    == "aa-null"`) — raises `BandGateError` naming exactly what is missing,
    never silently treating a malformed/foreign report as having zero
    pairs (which would make it silently NOT contribute to the worst
    deviation instead of failing loudly).
    """
    path = aa_null_dir / filename
    if not path.is_file():
        raise BandGateError(f"manifest names {filename!r}, but {path} does not exist")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BandGateError(f"{path} does not parse as JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise BandGateError(f"{path} is not a JSON object")
    missing = [k for k in REQUIRED_REPORT_KEYS if k not in report]
    if missing:
        raise BandGateError(f"{path} is missing required key(s) {missing} -- not a well-formed aa-null report")
    if report["mode"] != "aa-null":
        raise BandGateError(f"{path}'s own mode is {report['mode']!r}, not 'aa-null' -- this gate only derives from empirical-null campaign reports")
    pairs = report["adjacent_pair_ratios"]
    if not isinstance(pairs, dict) or not pairs:
        raise BandGateError(f"{path}'s own 'adjacent_pair_ratios' must be a non-empty object, got {pairs!r}")
    for key, ratio in pairs.items():
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise BandGateError(
                f"{path}'s own adjacent_pair_ratios[{key!r}] = {ratio!r} is not a positive finite "
                f"number -- |ln(ratio)| is undefined for it, and silently skipping it would "
                f"under-report the worst deviation"
            )
    return report


def run_start_window(aa_null_dir: Path, filename: str, report: dict) -> tuple[int, int]:
    """`(min, max)` over the report's `recorded_order` leg-start timestamps
    (nanosecond epoch) — the run's start-window, the same committed fields
    the `pcie-p1`/`p2` concurrency finding was computed from. Raises
    `BandGateError` when no usable timestamp exists (a committed aa-null
    report with no recorded starts cannot be cleared against, or checked
    for, concurrent siblings — fail loudly, never silently exempt it from
    the disjointness check).
    """
    order = report.get("recorded_order")
    if not isinstance(order, dict):
        raise BandGateError(f"{aa_null_dir / filename}'s own 'recorded_order' must be an object, got {order!r}")
    values = []
    for leg, entry in order.items():
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        values.append(value)
    if not values:
        raise BandGateError(
            f"{aa_null_dir / filename} carries no usable recorded_order timestamps -- its start-window "
            f"cannot be checked for concurrent-sibling overlap (the pcie-p1/p2 contamination signature)"
        )
    return min(values), max(values)


def check_directory_completeness(aa_null_dir: Path, runs: list[dict]) -> None:
    """Every `*.json` in the directory except `manifest.json` must be listed
    in the manifest (round-2 delta-audit B4): the re-derivation iterates
    manifest entries, so an unlisted report would be INVISIBLE to it —
    omission would otherwise be a silent escape hatch from the very
    evidence-completeness this gate exists to enforce.
    """
    on_disk = {p.name for p in aa_null_dir.glob("*.json")} - {"manifest.json"}
    listed = {entry["file"] for entry in runs}
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise BandGateError(
            f"{aa_null_dir} contains report file(s) not listed in manifest.json: {unlisted} -- "
            f"every committed report must be classified primary or aux; an unlisted report is "
            f"invisible to the re-derivation"
        )


def worst_abs_log_over_primary_pairs(aa_null_dir: Path) -> tuple[float, list[str]]:
    """Loads the manifest + every listed report under `aa_null_dir`
    (validating EVERY entry's schema, `"aux"`-role included, so a
    bit-rotted auxiliary artifact still fails loudly), then returns
    `(worst_abs_log_deviation, primary_filenames)` computed ONLY over
    `"primary"`-role files' own `adjacent_pair_ratios` values — `"aux"`-role
    pairs are loaded/validated but EXCLUDED from the derivation itself,
    the whole point of the primary/aux split
    (`ci/artifacts/gpu-perf-aa-null/README.md`'s own "Disclosure" section).
    Raises `BandGateError` if no primary pairs exist at all (an
    uncomputable input, never a silent worst-deviation of `0.0`).
    """
    runs = load_manifest(aa_null_dir)
    check_directory_completeness(aa_null_dir, runs)
    worst = None
    primary_files: list[str] = []
    windows: dict[str, tuple[int, int]] = {}
    roles: dict[str, str] = {}
    for entry in runs:
        report = load_and_validate_report(aa_null_dir, entry["file"])
        windows[entry["file"]] = run_start_window(aa_null_dir, entry["file"], report)
        roles[entry["file"]] = entry["role"]
        if entry["role"] != "primary":
            continue
        primary_files.append(entry["file"])
        for ratio in report["adjacent_pair_ratios"].values():
            abs_log = abs(math.log(ratio))
            if worst is None or abs_log > worst:
                worst = abs_log
    if worst is None:
        raise BandGateError("no 'primary'-role runs found in the manifest -- cannot derive a band from zero pairs")
    # Round-2 delta-audit B6, mechanized: a PRIMARY run's start-window
    # overlapping ANY other listed run's is the pcie-p1/p2 concurrency
    # signature -- positive contamination evidence, so that run cannot be
    # band evidence. Aux runs may overlap each other (that is why p1/p2
    # are aux); only a primary overlapping anything is a finding.
    for pf in primary_files:
        p_lo, p_hi = windows[pf]
        for other, (o_lo, o_hi) in windows.items():
            if other == pf:
                continue
            if p_lo <= o_hi and o_lo <= p_hi:
                raise BandGateError(
                    f"primary run {pf!r} (start-window [{p_lo}, {p_hi}]) overlaps {other!r} "
                    f"(role {roles[other]!r}, start-window [{o_lo}, {o_hi}]) -- the same concurrent-invocation "
                    f"signature that demoted pcie-p1/p2; a primary run with an overlapping sibling is "
                    f"contamination-suspect and cannot back the band"
                )
    return worst, primary_files


def check_band(aa_null_dir: Path):
    """Returns `(ok, message)`: `ok` is `True` iff the re-derived band
    (via [`worst_abs_log_over_primary_pairs`] +
    `gpu_inference_ab.derive_advisory_band`) equals
    `gpu_inference_ab.PRE_REGISTERED_ADVISORY_BAND` exactly. `message` is a
    human-readable summary either way (never silent on a PASS -- a reader
    should see the worst deviation and file count that produced it).
    """
    worst, primary_files = worst_abs_log_over_primary_pairs(aa_null_dir)
    derived = gpu_inference_ab.derive_advisory_band(worst)
    expected = gpu_inference_ab.PRE_REGISTERED_ADVISORY_BAND
    ok = derived == expected
    message = (
        f"worst |log deviation| over {len(primary_files)} primary file(s) {sorted(primary_files)} = {worst!r}; "
        f"re-derived band = {derived!r}; PRE_REGISTERED_ADVISORY_BAND = {expected!r}"
    )
    return ok, message


def main() -> int:
    try:
        ok, message = check_band(AA_NULL_DIR)
    except BandGateError as exc:
        print(f"check-aa-null-band: FAIL (uncomputable) -- {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(f"check-aa-null-band: FAIL -- {message}", file=sys.stderr)
        return 1
    print(f"check-aa-null-band: PASS -- {message}")
    return 0


# --------------------------------------------------------------------------- #
# --self-test: synthetic fixtures, never touching the real committed
# artifacts -- proves the re-derivation/exclusion/schema-violation logic
# itself, the same "prove the mechanism, not just today's committed data"
# discipline check_pod_build_timings.py's own --self-test already follows.
# --------------------------------------------------------------------------- #
def _write_fixture_report(path: Path, *, mode="aa-null", pair_a=1.0, pair_b=1.0, extra_keys=True, t0=1_000_000):
    """`t0` spaces each fixture's recorded_order start-window (four leg
    starts at `t0 .. t0+3`) — call sites give sibling fixtures disjoint
    `t0`s except when a test deliberately constructs the overlap the
    disjointness check must catch."""
    report = {
        "mode": mode,
        "legs": {"a1": {}, "b1": {}, "b2": {}, "a2": {}} if extra_keys else {},
        "adjacent_pair_ratios": {"a1/b1": pair_a, "b2/a2": pair_b},
        "recorded_order": {
            leg: {"value": t0 + i, "unavailable_reason": None}
            for i, leg in enumerate(("a1", "b1", "b2", "a2"))
        },
    }
    if not extra_keys:
        del report["legs"]
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_manifest(dirpath: Path, entries):
    (dirpath / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "runs": entries}), encoding="utf-8"
    )


def self_test() -> int:
    failures = []

    # (1) Re-derivation correctness on a KNOWN synthetic worst pair: proves
    # `worst_abs_log_over_primary_pairs` + `derive_advisory_band` recover
    # the SAME band an independent, by-hand computation over the SAME
    # fixture numbers gives -- `check_band`'s own PASS/FAIL is judged
    # against the REAL committed `PRE_REGISTERED_ADVISORY_BAND` constant
    # (unrelated to a synthetic fixture's own numbers), so this leg
    # deliberately does NOT call `check_band` at all; it drives the
    # re-derivation machinery directly and compares against an
    # independently-computed expectation, non-vacuously (a wrong worst
    # value would produce a DIFFERENT band, which the equality check below
    # would then catch).
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        worst_ratio = 0.9  # |ln(0.9)| ~= 0.10536
        _write_fixture_report(d / "p1.json", pair_a=worst_ratio, pair_b=1.0, t0=1_000_000)
        _write_fixture_report(d / "p2.json", pair_a=1.0, pair_b=1.0, t0=2_000_000)
        _write_manifest(d, [
            {"file": "p1.json", "role": "primary", "reason": "clean"},
            {"file": "p2.json", "role": "primary", "reason": "clean"},
        ])
        expected = gpu_inference_ab.derive_advisory_band(abs(math.log(worst_ratio)))
        worst_recomputed, primary_files = worst_abs_log_over_primary_pairs(d)
        if abs(worst_recomputed - abs(math.log(worst_ratio))) > 1e-12:
            failures.append(f"self-test (1): expected worst={abs(math.log(worst_ratio))!r}, got {worst_recomputed!r}")
        if sorted(primary_files) != ["p1.json", "p2.json"]:
            failures.append(f"self-test (1): expected both files counted as primary, got {primary_files!r}")
        if gpu_inference_ab.derive_advisory_band(worst_recomputed) != expected:
            failures.append("self-test (1): derive_advisory_band did not reproduce the independently-computed expectation")

        # (1b) RED-fixture non-vacuity: `check_band` itself must actually be
        # ABLE to FAIL -- this synthetic fixture's own worst deviation
        # derives (0.85, 1.17), which does not equal the REAL committed
        # PRE_REGISTERED_ADVISORY_BAND (0.75, 1.33); `check_band(d)` must
        # report `ok=False` here, proving the mismatch-detection path fires
        # on a genuine mismatch, never only ever returning True.
        ok, message = check_band(d)
        if ok:
            failures.append(f"self-test (1b): check_band should have reported a mismatch on this synthetic fixture, got ok=True: {message}")

    # (2) Aux exclusion actually matters: an aux-role file with an EXTREME
    # ratio must NOT move the worst deviation -- proven by comparing
    # against the SAME fixture with that file re-labeled primary.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "primary.json", pair_a=0.95, pair_b=1.0, t0=1_000_000)
        _write_fixture_report(d / "extreme.json", pair_a=0.1, pair_b=0.1, t0=2_000_000)  # would dominate if counted
        _write_manifest(d, [
            {"file": "primary.json", "role": "primary", "reason": "clean"},
            {"file": "extreme.json", "role": "aux", "reason": "contention"},
        ])
        worst_excluded, _ = worst_abs_log_over_primary_pairs(d)
        _write_manifest(d, [
            {"file": "primary.json", "role": "primary", "reason": "clean"},
            {"file": "extreme.json", "role": "primary", "reason": "clean"},
        ])
        worst_included, _ = worst_abs_log_over_primary_pairs(d)
        if worst_excluded == worst_included:
            failures.append("self-test (2): aux-role exclusion did not change the worst-deviation computation -- exclusion is not actually wired")
        if abs(worst_excluded - abs(math.log(0.95))) > 1e-12:
            failures.append(f"self-test (2): expected worst_excluded to be the primary-only pair, got {worst_excluded!r}")

    # (3) Schema violation (missing key) fails loudly, never silently
    # treated as zero pairs.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "broken.json").write_text(json.dumps({"mode": "aa-null"}), encoding="utf-8")  # no legs/pairs/recorded_order
        _write_manifest(d, [{"file": "broken.json", "role": "primary", "reason": "clean"}])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (3): a report missing required keys should have raised BandGateError")
        except BandGateError:
            pass

    # (4) Wrong mode fails loudly.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "wrongmode.json", mode="ab")
        _write_manifest(d, [{"file": "wrongmode.json", "role": "primary", "reason": "clean"}])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (4): a report with mode != 'aa-null' should have raised BandGateError")
        except BandGateError:
            pass

    # (5) No primary runs at all fails loudly (never a silent worst=0.0).
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "auxonly.json")
        _write_manifest(d, [{"file": "auxonly.json", "role": "aux", "reason": "clean"}])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (5): zero primary runs should have raised BandGateError")
        except BandGateError:
            pass

    # (6) Missing manifest fails loudly.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        try:
            load_manifest(d)
            failures.append("self-test (6): an absent manifest.json should have raised BandGateError")
        except BandGateError:
            pass

    # (8) Directory completeness: an on-disk report ABSENT from the
    # manifest fails loudly (round-2 delta-audit B4's own demonstrated
    # escape: an unlisted extreme-ratio report was invisible to the gate).
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "listed.json", pair_a=0.95, pair_b=1.0, t0=1_000_000)
        _write_fixture_report(d / "unlisted.json", pair_a=0.2, pair_b=0.2, t0=2_000_000)
        _write_manifest(d, [{"file": "listed.json", "role": "primary", "reason": "clean"}])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (8): an on-disk report missing from the manifest should have raised BandGateError")
        except BandGateError:
            pass

    # (9) Primary start-window overlap fails loudly (round-2 delta-audit
    # B6, mechanized: the pcie-p1/p2 concurrency signature must be a gate
    # finding when a PRIMARY run carries it, whatever the sibling's role).
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "prim.json", pair_a=0.95, pair_b=1.0, t0=1_000_000)
        _write_fixture_report(d / "nested.json", pair_a=1.0, pair_b=1.0, t0=1_000_001)  # nested inside prim's window
        _write_manifest(d, [
            {"file": "prim.json", "role": "primary", "reason": "clean"},
            {"file": "nested.json", "role": "aux", "reason": "contention"},
        ])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (9): a primary whose start-window overlaps a sibling should have raised BandGateError")
        except BandGateError:
            pass
        # ...while two AUX runs overlapping each other stays legal (that is
        # exactly the committed pcie-p1/p2 shape).
        _write_fixture_report(d / "solo.json", pair_a=0.95, pair_b=1.0, t0=9_000_000)
        _write_manifest(d, [
            {"file": "prim.json", "role": "aux", "reason": "contention"},
            {"file": "nested.json", "role": "aux", "reason": "contention"},
            {"file": "solo.json", "role": "primary", "reason": "clean"},
        ])
        try:
            worst_abs_log_over_primary_pairs(d)
        except BandGateError as exc:
            failures.append(f"self-test (9): two overlapping AUX runs beside a disjoint primary should be legal, got: {exc}")

    # (10) A non-positive / non-finite pair ratio fails loudly with the
    # gate's own named error, never a bare math-domain traceback.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_fixture_report(d / "zero.json", pair_a=0.0, pair_b=1.0, t0=1_000_000)
        _write_manifest(d, [{"file": "zero.json", "role": "primary", "reason": "clean"}])
        try:
            worst_abs_log_over_primary_pairs(d)
            failures.append("self-test (10): a zero pair ratio should have raised BandGateError")
        except BandGateError:
            pass
        except ValueError as exc:
            failures.append(f"self-test (10): a zero pair ratio leaked a bare {type(exc).__name__} instead of BandGateError: {exc}")

    # (7) Non-vacuousness on the REAL tree: the real gate must find a
    # nonzero number of primary files under the real committed directory
    # (a --self-test that only ever exercises synthetic fixtures, and never
    # confirms the real tree has ANY primary evidence at all, could stay
    # green even if AA_NULL_DIR's own manifest silently emptied out).
    try:
        _, real_primary_files = worst_abs_log_over_primary_pairs(AA_NULL_DIR)
        if not real_primary_files:
            failures.append("self-test (7): the REAL ci/artifacts/gpu-perf-aa-null/ tree has zero primary files")
    except BandGateError as exc:
        failures.append(f"self-test (7): the REAL tree failed to load at all: {exc}")

    if failures:
        print("check-aa-null-band --self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("check-aa-null-band --self-test: PASS -- 12 fixture checks (re-derivation, mismatch detection, "
          "aux-exclusion, 3 schema-violation shapes, absent-manifest, directory-completeness, "
          "primary-window-overlap + legal-aux-overlap, bad-ratio-value, real-tree non-vacuity), all as expected.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
