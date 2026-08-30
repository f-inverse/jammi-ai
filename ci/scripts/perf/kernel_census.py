#!/usr/bin/env python3
"""Per-`(kernel, grid, block)` census over an nsys sqlite export, and the
(M-N)-step DIFFERENCE of two exports (same declared workload, N vs M
measured training steps -- CONTRACT `scratchpad/contract-356-profile.md`
v3, `## Method` / `### Census`) -- isolates exactly `(M-N)` optimizer
steps' worth of kernels without any landmark segmentation, because every
`## Declared workload` flag is pinned identical across the pair
(`--validation-fraction 0 --early-stopping-metric train_loss --epochs 1
--grad-accum 1`, fixed eval cadence) so validation/probe/checkpoint work
is IDENTICAL across the two runs and cancels in the subtraction.

Promoted from the throwaway `scratchpad/pod/kernel_census.py` ancestor
(P4, contract's precondition table): same schema query (CUPTI_ACTIVITY_
KIND_KERNEL joined to StringIds on `shortName`; memcpy/memset totals; GPU
wall span), hardened per contract. Every guard below is a DOMAIN check on
whether the two sqlite exports actually describe the declared
same-workload (N, M) pair -- none of them are generic exception handling
for its own sake:

  - refuses (exit nonzero, no report) if EITHER export lacks a
    `CUPTI_ACTIVITY_KIND_KERNEL` table -- the contract's own "Per-leg
    check: export lacking the kernel table => leg INVALID" (`### Instrument`).
    A missing kernel table means the export never records the kernel-level
    trace this whole census depends on; there is no partial/degraded
    report to emit, only a loud refusal.
  - refuses (exit nonzero, no report) if EITHER export's kernel table is
    PRESENT but EMPTY (zero rows) -- a present-but-empty table is the SAME
    instrument failure as a missing table (no kernel-level trace was
    captured), not a legitimate "zero kernels dispatched" measurement; left
    unchecked this would exit 0 with every share silently zero, which a
    downstream reader could mistake for a genuine DECLINE-licensing
    negative result rather than an unusable capture.
  - refuses (exit nonzero) unless `steps_b > steps_a` (M > N) -- a
    same-workload pair is defined by "measure M steps, measure N steps,
    subtract"; M<=N is not a valid differencing pair at all (a zero or
    negative step delta divides by a non-positive quantity below).
  - refuses (exit nonzero, no report) if any per-key raw count/time delta
    is NEGATIVE beyond a tiny tolerance -- for a genuine same-workload N<M
    pair, every kernel/memcpy/memset a training step launches should only
    ACCUMULATE going from the N-step export to the M-step export (the
    contract's pinned flags make the two runs' non-training-step work
    identical, so it cancels to exactly zero, never negative); a
    meaningfully negative delta is the mechanical signature of "the two
    exports were not actually the declared same-workload (N, M) pair" --
    e.g. a stale/mismatched sqlite file, a build/config drift between the
    two captures, or an export path bug -- and must not be silently folded
    into a report a reader could mistake for a clean measurement.
    EXCEPTION, a per-key bucket whose LAUNCH COUNT delta is <= 0 AND within
    `--launch-tolerance` of zero (this includes the EXACT dn==0 case) is
    FIXED-COST, not part of the (M-N)-step differencing at all (round-4
    pod-run fix, first real 14-leg census): the init probe / held-out eval
    launch the SAME kernels the SAME number of times in the N-step export
    as in the M-step export (the contract's fixed eval cadence means this
    non-training-step work is IDENTICAL across the pair, so its LAUNCH
    COUNT cancels, exactly or within the caller's own declared count
    noise), but its SUMMED time is a nanosecond-scale accumulator that
    jitters either sign run to run (scheduler noise, clock granularity) --
    a bucket that never grew its count is not "the two exports diverged",
    it is "this bucket did no differencing-relevant work in either run",
    so it is excluded from the per-step report rows entirely (dividing a
    non-positive count delta by `steps_b - steps_a` would either report a
    fixed-cost kernel as if it scaled with the differenced step count,
    which it does not, or emit a NEGATIVE `launches_per_step` into the
    headline, which round-4's phase-4 audit reproduced as a live finding).
    A fixed-cost bucket's time delta is NEVER checked against the flat
    `--time-tolerance-us` the real added-work buckets use -- but it is NOT
    unconditionally waved through either (phase-4 audit BLOCK 2 round-1,
    tightened again round-2 re-audit BLOCK 1): the justifying premise is
    "nanosecond-scale jitter", so this is ENFORCED as a HYBRID bound --
    `|time delta| > max(--fixed-cost-jitter-floor-ns,
    --fixed-cost-jitter-rel-tolerance * max(time_a, time_b))` refuses,
    otherwise it is jitter. Never a RELATIVE bound alone: a relative-only
    bound has zero margin at any single calibrated value (a same-workload
    pair's own "must never refuse" fixture can measure exactly AT a
    relative bound with no headroom) and no floor at all lets a tiny
    absolute duration (a single-launch eval-probe kernel a few hundred
    nanoseconds long) trip a large RELATIVE jitter off a few-hundred-
    nanosecond absolute swing that is obviously still noise -- see
    `DEFAULT_FIXED_COST_JITTER_FLOOR_NS`/
    `DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE`'s own comment for the full
    calibration, including the round-4 pod-run LIVE FINDING's honest
    relative jitter (0.9998, not "many multiples of 100%" -- a ratio of an
    absolute-value numerator to a same-sign-or-larger denominator cannot
    exceed 1.0). A fixed-cost bucket whose jitter exceeds the hybrid bound
    refuses exactly like every other per-key violation above (folded into
    the same `NonComparablePairError`). Every fixed-cost bucket (whether
    it passed or would have failed the bound -- the informational tally
    always runs before the bound is checked) is summarized as
    `fixed_cost_buckets` (a count), `fixed_cost_time_us` (the sum of their
    `|time delta|`), and `fixed_cost_jitter_max_rel` (the largest single
    bucket's relative jitter observed) so the cancellation -- and how much
    headroom it used -- stays visible rather than silently vanishing. A
    bucket whose count delta is negative BEYOND `--launch-tolerance`, or
    positive with a negative time delta, is NOT fixed-cost -- both remain
    refused exactly as before this round. The SAME classification and
    hybrid bound apply to the memcpy/memset aggregate deltas too (phase-4
    audit round-2 re-audit advisory 2): a cancelling (dn<=0 within
    `--launch-tolerance`) memcpy/memset delta is fixed-cost, checked
    against the hybrid bound, never the flat `--time-tolerance-us`
    (whose default of 0.0 would otherwise refuse on a single nanosecond
    of negative memcpy/memset jitter -- the driver passes neither flag).

    `--launch-tolerance`, `--time-tolerance-us`, and
    `--fixed-cost-jitter-rel-tolerance` are themselves validated
    (phase-4 audit round-2 re-audit BLOCK 2): `launch_tolerance` and
    `time_tolerance_us` must be finite and >= 0 (a negative
    `launch_tolerance` would silently EMPTY the fixed-cost range
    `[-launch_tolerance, 0]` and invert the negative-count-delta
    refusal); `fixed_cost_jitter_rel_tolerance` must be finite and in
    `[0, 1)` (NaN/inf/>=1.0 would fail OPEN -- silently never refusing a
    fixed-cost bucket's jitter, reopening the exact gap round-1's fix
    closed -- and a negative value inverts the guard the same way a
    negative `launch_tolerance` does). A caller passing an out-of-domain
    value for any of the three gets a usage-error refusal (`ValueError`,
    `main()` exit 2) naming the knob and its valid domain, never a
    silently-degraded or silently-inverted guard.

    A SECOND, independent guard closes the "every bucket happened to be
    fixed-cost" degenerate case (phase-4 audit BLOCK 1): if ZERO buckets
    carry a POSITIVE launch-count delta at all (e.g. the M-step run
    silently executed only N steps, or the same export was diffed against
    itself), the classification above alone would still produce a
    STRUCTURALLY EMPTY but otherwise "clean" report (`gpu_kernel_us_per_step
    == 0`, `by_kernel_name == []`) that a caller could mistake for a
    genuine all-cancels measurement rather than "this was never a real
    M>N same-workload pair at all". This is checked AFTER every per-key
    violation above (a more specific bound violation is reported first)
    and refuses with `EmptyDifferencedCensusError` -- a report is never
    silently returned with zero kernel-level differencing signal in it.
  - refuses (exit nonzero, no report) unless `wall_b > wall_a > 0` whenever
    `--wall-a`/`--wall-b` are both given -- the SAME non-negativity/
    ordering domain check the per-key kernel/memcpy/memset deltas already
    get above, applied to the one delta this module cannot derive from the
    sqlite exports themselves (a caller-supplied `train_run_wall_s` pair
    that is non-positive, equal, or inverted is not a valid same-workload
    M>N wall-clock pair either, and dividing by `steps_b - steps_a` would
    otherwise silently emit a negative or infinite `wall_s_per_step`).
  - cross-checks the CALLER-declared `steps_a`/`steps_b` against the
    report's own MEASURED `steps_measured` (`FinetuneRunTier`), when the
    caller supplies `--steps-measured-a`/`--steps-measured-b` -- refuses
    if either measured value disagrees with the declared one, since a
    leg whose run did not actually execute the declared step count is not
    the pair this differencing was supposed to isolate (a truncated run,
    an early-stopping firing despite the never-stops idiom, or a caller
    passing the wrong (N, M) pair to this tool by mistake). Optional
    (omitted when the caller has no measured value to check against, e.g.
    a build predating `train_run_wall_s`/`steps_measured` or a DRY_RUN
    smoke stub that chooses not to populate it).
  - a corrupt/truncated sqlite file (`sqlite3.DatabaseError`, e.g. a
    0-byte-but-nonzero-length garbage file, or a file truncated mid-write)
    is caught and re-raised as a NAMED, declared exit code -- never an
    unhandled Python traceback, which is indistinguishable from a bug in
    this tool itself to anything scraping this producer's exit status.

Output JSON: the ancestor's shape (`steps_diff`, `gpu_kernel_us_per_step`,
`launches_per_step`, `memcpy_per_step`, `memset_per_step`,
`by_kernel_name`, `by_kernel_and_grid`) plus `nsys_sqlite_schema_ok`,
`steps_a`, `steps_b` (CONTRACT P4's own field list), plus
`fixed_cost_buckets`/`fixed_cost_time_us`/`fixed_cost_jitter_max_rel`
(round-4 pod-run fix -- see the fixed-cost exception in the guard list
above).

Wall denominator (round-3 pressure-test, contract v4 -- `### Wall
denominator`): `--wall-a`/`--wall-b` (seconds, each run's OWN
`train_run_wall_s` -- `FinetuneRunTier`) are OPTIONAL; when BOTH are
given (and pass the `wall_b > wall_a > 0` domain check above), the report
also carries `wall_s_per_step = (wall_b - wall_a) / (steps_b - steps_a)`
-- the same (M-N)-step differencing this module already applies to
kernel/memcpy/memset counts, applied to the wall clock too, so
`share_wall(C) = time(C)/wall_p50` (`### Wall denominator`) has a
same-footing per-step wall figure to divide into. Omitted (both flags
absent) leaves `wall_s_per_step` out of the report entirely -- never a
fabricated 0.0 -- since this differencer has no way to independently
measure wall-clock time itself; a caller unable to supply two real,
correctly-ordered `train_run_wall_s` values (whatever the reason -- a
build that does not emit the field, a report shape it could not parse,
or simply choosing not to wire it) must not have a wall denominator
silently fabricated on its behalf.

Chain-attribution exclusion (round-3 pressure-test, contract v4): E1 (the
ecological, variable-width leg) is EXCLUDED from signature-based chain
attribution (`### Attribution`'s element-count/shape signatures assume a
FIXED width; E1's `BatchLongest` batching fans width out per batch, which
would fan the same kernel out across multiple grid/block buckets and make
the by-shape signatures uncomparable to the fixed-width legs). Passing
`--excluded-from-chain-attribution` stamps `excluded_from_chain_attribution:
true` on the report so a downstream merger/reader never mistakes E1's
by-name/by-grid rows for chain-attributable evidence -- E1's own role is
the ecological wall anchor, the LoRA counter check, and the width report
(`fixture_width_report.py`), never `### Attribution`'s per-chain shares.
Omitted (the default) stamps `false`.

Usage: kernel_census.py A.sqlite B.sqlite STEPS_A STEPS_B out.json
       [--launch-tolerance N] [--time-tolerance-us F]
       [--fixed-cost-jitter-floor-ns N] [--fixed-cost-jitter-rel-tolerance F]
       [--wall-a SECONDS --wall-b SECONDS] [--excluded-from-chain-attribution]
       [--steps-measured-a N --steps-measured-b M]

Hermetic: reads only the two sqlite files named on the command line (plus
stdlib `sqlite3`); no network, no build, no GPU. Exit codes: 0 = report
written; 2 = usage error (including M<=N, a one-sided --wall-*, or any of
`--launch-tolerance`/`--time-tolerance-us`/
`--fixed-cost-jitter-rel-tolerance` outside its validated domain -- see
the module doc's "EXCEPTION" paragraph above); 3 = a kernel table is missing on one or both
exports (leg INVALID -- instrument failure); 4 = a per-key delta is
negative beyond tolerance, OR a fixed-cost bucket's (kernel or
memcpy/memset) time jitter exceeds the HYBRID
`--fixed-cost-jitter-floor-ns`/`--fixed-cost-jitter-rel-tolerance` bound
(leg INVALID -- non-comparable pair, not a genuine negative measurement);
5 = declared steps disagree with the report's own steps_measured (leg
INVALID); 6 = a kernel table is present but EMPTY on one or both exports
(leg INVALID -- same instrument failure as exit 3); 7 = the wall pair
fails `wall_b > wall_a > 0` (leg INVALID); 8 = a sqlite export is
corrupt/unreadable (`sqlite3.DatabaseError`); 9 = the differenced census
is EMPTY -- zero kernel buckets carried a positive launch-count delta
(leg INVALID -- not a genuine declared M>N same-workload pair).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sqlite3
import sys

# A "tiny tolerance" (contract's own phrasing): raw (pre-division) delta
# floors below which a negative per-key delta is treated as capture/export
# noise rather than a sign the pair is not comparable. Deliberately small --
# these are exact integer/nanosecond accumulators for a same-workload pair
# under pinned flags, so a real divergence should read as a large negative
# value, not a borderline one; a caller with evidence for a wider noise
# floor on a particular box/nsys version overrides via the CLI flags.
DEFAULT_LAUNCH_TOLERANCE = 0
DEFAULT_TIME_TOLERANCE_US = 0.0

# The HYBRID bound a FIXED-COST bucket's (see module doc) time delta is
# checked against (round-2 re-audit BLOCK 1 on the round-1 fix's own
# single relative bound): a fixed-cost bucket refuses only if
# `|time delta| > max(FIXED_COST_JITTER_FLOOR_NS, rel_tolerance *
# max(time_a, time_b))` -- the LARGER of an ABSOLUTE floor and a RELATIVE
# bound, never a relative bound alone. A relative-only bound has two
# failure modes a pod-run adversarial pass actually found: (1) ZERO
# margin at any calibrated rel value (the round-4 fix's own "must never
# refuse" in-tree fixture measured rel == 0.100000 EXACTLY against a
# 0.10 bound -- a single extra nanosecond of jitter per launch flips it
# to a false refusal), and (2) no floor at all means a bucket with a TINY
# absolute duration (a single-launch eval-probe kernel a few hundred
# nanoseconds long) can trip a large RELATIVE jitter from a few-hundred-
# nanosecond absolute swing that is obviously still noise, and because
# ANY single bucket's violation is leg-fatal, this false-refusal rate
# compounds over the K buckets a real census differences.
#
# `FIXED_COST_JITTER_FLOOR_NS` = 1,000,000ns (1ms): comfortably above any
# real per-bucket scheduler/clock-granularity jitter (which lands in the
# tens-to-hundreds of nanoseconds, not milliseconds), and three orders of
# magnitude below the round-4 pod-run LIVE FINDING this whole guard exists
# to catch -- that finding's own numbers are a fixed-cost bucket whose
# time delta was ~49.99ms against a bucket whose own max(time_a, time_b)
# was ~50.01ms, i.e. a RELATIVE jitter of 0.9998 (49_990_000 / 50_010_000)
# -- NOT "many multiples of 100%", which is arithmetically impossible for
# a ratio of an absolute-value numerator to a same-sign-or-larger
# denominator (phase-4 audit advisory 1, correcting an earlier overstated
# claim in this same comment).
#
# `DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE` = 0.5 (50%): the auditor's own
# note is that ANY bound in [0.3, 0.9] catches the live finding's 0.9998
# with margin; 0.5 sits in the middle of that range, comfortably clear of
# both the live finding (0.9998) and genuine calibration fixtures well
# under it, while still being loose enough that legitimate cross-run
# scheduling variance on a shared GPU (which can plausibly swing a bucket
# by tens of percent, not just single-digit percent) does not false-
# refuse. A caller with box/nsys-version-specific evidence for a
# different pair overrides via `--fixed-cost-jitter-floor-ns`/
# `--fixed-cost-jitter-rel-tolerance`.
DEFAULT_FIXED_COST_JITTER_FLOOR_NS = 1_000_000
DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE = 0.5


class KernelTableMissingError(RuntimeError):
    """Named exception for the leg-INVALID "no CUPTI_ACTIVITY_KIND_KERNEL
    table" condition -- lets `main()` report a distinguishable exit code
    and a message that names which export failed, never conflated with a
    generic sqlite/parse error."""


class KernelTableEmptyError(RuntimeError):
    """Named exception for the leg-INVALID "CUPTI_ACTIVITY_KIND_KERNEL table
    is PRESENT but has zero rows" condition -- distinct from
    `KernelTableMissingError` (the table exists, so the schema check alone
    would pass) but the SAME underlying failure: no kernel-level trace was
    actually captured. See module doc."""


class NonComparablePairError(RuntimeError):
    """Named exception for the leg-INVALID "a per-key delta went negative
    beyond tolerance" condition -- see module doc."""


class WallPairInvalidError(RuntimeError):
    """Named exception for the leg-INVALID "`--wall-a`/`--wall-b` do not
    satisfy `wall_b > wall_a > 0`" condition -- see module doc."""


class StepsMismatchError(RuntimeError):
    """Named exception for the leg-INVALID "declared steps_a/steps_b
    disagree with the report's own measured steps_measured" condition --
    see module doc."""


class EmptyDifferencedCensusError(RuntimeError):
    """Named exception for the leg-INVALID "zero buckets carried a
    positive launch-count delta" condition -- every bucket classified as
    FIXED-COST (or, before this round's fix, silently produced an
    all-zero report) is not the same failure as a genuine per-key
    violation (`NonComparablePairError`): it means the pair never showed
    ANY differencing-relevant kernel work at all. See module doc."""


class CensusDatabaseError(RuntimeError):
    """Named exception wrapping a `sqlite3.DatabaseError` (corrupt/
    truncated export) -- lets `main()` return a declared, distinguishable
    exit code instead of an unhandled traceback. See module doc."""


def _has_kernel_table(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
    ).fetchone()
    return row is not None


def census(path: str) -> tuple[dict, dict, tuple]:
    """Returns (per-(kernel,grid,block) {(name,gx,gy,gz,bx,by,bz): (n, ns)},
    per-kind memcpy/memset {(count, ns)}, (min_start, max_end) wall span).

    Raises `KernelTableMissingError` if `path` has no
    `CUPTI_ACTIVITY_KIND_KERNEL` table, `KernelTableEmptyError` if that
    table exists but has zero rows (checked BEFORE any further query runs
    against it, so neither condition is ever silently read as "zero
    kernels dispatched"), or `CensusDatabaseError` if `path` cannot be read
    as a sqlite database at all (`sqlite3.DatabaseError` -- a corrupt or
    truncated export).
    """
    con = sqlite3.connect(path)
    try:
        try:
            table_present = _has_kernel_table(con)
        except sqlite3.DatabaseError as e:
            raise CensusDatabaseError(
                f"{path}: not a readable sqlite database ({e}) -- leg INVALID"
            ) from e
        if not table_present:
            raise KernelTableMissingError(
                f"{path}: no CUPTI_ACTIVITY_KIND_KERNEL table in this nsys sqlite export -- "
                "leg INVALID (instrument failure, not a genuine measurement; contract "
                "`### Instrument`'s per-leg check)"
            )
        cur = con.cursor()
        try:
            (row_count,) = cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        except sqlite3.DatabaseError as e:
            raise CensusDatabaseError(
                f"{path}: CUPTI_ACTIVITY_KIND_KERNEL exists but could not be read ({e}) -- "
                "leg INVALID"
            ) from e
        if row_count == 0:
            raise KernelTableEmptyError(
                f"{path}: CUPTI_ACTIVITY_KIND_KERNEL table is PRESENT but has zero rows -- "
                "leg INVALID (the same instrument failure as a missing table: no kernel-level "
                "trace was actually captured; a genuine all-zero census cannot be distinguished "
                "from a broken capture, so this refuses rather than emitting an all-zero report)"
            )
        # nsys 2025.x schema: CUPTI_ACTIVITY_KIND_KERNEL joined to StringIds for names.
        q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                      COUNT(*), SUM(k.end - k.start)
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
               GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ"""
        out: dict[tuple, tuple[int, int]] = {}
        for name, gx, gy, gz, bx, by, bz, n, ns in cur.execute(q):
            out[(name, gx, gy, gz, bx, by, bz)] = (n, ns or 0)

        # memcpy / memset totals too (host<->device traffic is a
        # capture-relevant signal) -- a missing table here (older/newer
        # schema variance) degrades to (0, 0), NOT a leg-INVALID condition:
        # unlike the kernel table, memcpy/memset presence is not what this
        # census's own validity depends on.
        mem: dict[str, tuple[int, int]] = {}
        for kind, tbl in (
            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
        ):
            try:
                n, ns = cur.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(end-start),0) FROM {tbl}"
                ).fetchone()
                mem[kind] = (n, ns)
            except sqlite3.OperationalError:
                mem[kind] = (0, 0)

        lo, hi = cur.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    finally:
        con.close()
    return out, mem, (lo, hi)


def _check_non_negative(
    label: str, delta: float, tolerance: float, violations: list[str]
) -> None:
    if delta < -tolerance:
        violations.append(f"{label}: raw delta {delta} is negative beyond tolerance {tolerance}")


def _classify_delta(dn: float, launch_tolerance: float) -> str:
    """Classifies a bucket's launch-count delta into exactly one of three
    buckets this module's guards treat differently (see module doc's
    "EXCEPTION" paragraph): `"fixed_cost"` (`dn` in
    `[-launch_tolerance, 0]` -- includes the exact `dn==0` case),
    `"positive"` (`dn > 0`, real added work), or `"negative_regression"`
    (`dn < -launch_tolerance`, always refused). Shared by the per-key
    kernel-bucket loop and the memcpy/memset aggregate deltas (phase-4
    audit round-2 re-audit advisory 2) so both apply the IDENTICAL
    classification, not two hand-maintained copies of the same three-way
    split."""
    if -launch_tolerance <= dn <= 0:
        return "fixed_cost"
    if dn > 0:
        return "positive"
    return "negative_regression"


def _fixed_cost_violation(
    label: str,
    dn: float,
    dns: float,
    ns_a: float,
    ns_b: float,
    jitter_floor_ns: float,
    jitter_rel_tolerance: float,
) -> tuple[float, str | None]:
    """For a bucket `_classify_delta` already called `"fixed_cost"`,
    returns `(relative jitter observed, violation message or None)` --
    the HYBRID bound (phase-4 audit round-2 re-audit BLOCK 1, see
    `DEFAULT_FIXED_COST_JITTER_FLOOR_NS`'s own comment for the full
    calibration): refuses only if `|dns|` exceeds BOTH an absolute floor
    and a bound relative to the bucket's own `max(ns_a, ns_b)` -- i.e.
    `|dns| > max(jitter_floor_ns, jitter_rel_tolerance * max(ns_a, ns_b))`
    -- never a relative bound alone (zero margin at any single calibrated
    value) and never an absolute floor alone (a genuinely huge bucket's
    proportionally-tiny-but-absolutely-large jitter would still need
    catching)."""
    denom = max(ns_a, ns_b, 1)
    rel = abs(dns) / denom
    bound_ns = max(jitter_floor_ns, jitter_rel_tolerance * denom)
    if abs(dns) <= bound_ns:
        return rel, None
    return rel, (
        f"{label} fixed-cost (launch count delta {dn}) time (ns) jitter {dns} "
        f"(|delta|={abs(dns)}ns, rel={rel:.4f}x max(time_a={ns_a}, time_b={ns_b})={denom}) "
        f"exceeds max(fixed_cost_jitter_floor_ns={jitter_floor_ns}, "
        f"fixed_cost_jitter_rel_tolerance={jitter_rel_tolerance}*{denom}="
        f"{jitter_rel_tolerance * denom:.1f})={bound_ns:.1f} -- not nanosecond-scale jitter"
    )


def build_report(
    path_a: str,
    path_b: str,
    steps_a: int,
    steps_b: int,
    launch_tolerance: int = DEFAULT_LAUNCH_TOLERANCE,
    time_tolerance_us: float = DEFAULT_TIME_TOLERANCE_US,
    fixed_cost_jitter_floor_ns: float = DEFAULT_FIXED_COST_JITTER_FLOOR_NS,
    fixed_cost_jitter_rel_tolerance: float = DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE,
    wall_a: float | None = None,
    wall_b: float | None = None,
    excluded_from_chain_attribution: bool = False,
    steps_measured_a: int | None = None,
    steps_measured_b: int | None = None,
) -> dict:
    """Builds the census-difference report dict, or raises one of this
    module's named exceptions (see module doc). Never partially writes a
    report on any error path -- the caller only serializes the return
    value once this function returns successfully."""
    if steps_b <= steps_a:
        raise ValueError(
            f"steps_b ({steps_b}) must be strictly greater than steps_a ({steps_a}) -- "
            "a census difference needs a genuine M>N same-workload pair"
        )
    # Tolerance-knob domain validation (phase-4 audit round-2 re-audit
    # BLOCK 2): an out-of-domain value for any of these three would
    # silently DEGRADE or INVERT a guard rather than raise -- caught here,
    # once, before any of the per-key classification below ever reads
    # them, rather than three separate ad-hoc checks scattered through the
    # loop. `ValueError` (the same usage-error family `steps_b <= steps_a`
    # above already uses -> `main()`'s existing `except ValueError`
    # handler -> exit 2) -- these are CALLER-supplied argument domain
    # errors, not a leg-INVALID measurement outcome.
    if not (math.isfinite(launch_tolerance) and launch_tolerance >= 0):
        raise ValueError(
            f"launch_tolerance must be finite and >= 0, got {launch_tolerance!r} -- a negative "
            "value would silently EMPTY the fixed-cost range ([-launch_tolerance, 0]) and invert "
            "the negative-count-delta refusal"
        )
    if not (math.isfinite(time_tolerance_us) and time_tolerance_us >= 0):
        raise ValueError(
            f"time_tolerance_us must be finite and >= 0, got {time_tolerance_us!r}"
        )
    if not (
        math.isfinite(fixed_cost_jitter_rel_tolerance)
        and 0.0 <= fixed_cost_jitter_rel_tolerance < 1.0
    ):
        raise ValueError(
            f"fixed_cost_jitter_rel_tolerance must be finite and in [0, 1), got "
            f"{fixed_cost_jitter_rel_tolerance!r} -- a NaN/inf/>=1.0 value would fail OPEN "
            "(silently never refusing a fixed-cost bucket's time jitter), and a negative value "
            "inverts the guard"
        )
    if wall_a is not None and wall_b is not None and not (wall_a > 0 and wall_b > wall_a):
        raise WallPairInvalidError(
            f"--wall-a={wall_a} --wall-b={wall_b} do not satisfy wall_b > wall_a > 0 -- not a "
            "valid same-workload M>N wall-clock pair"
        )
    if steps_measured_a is not None and steps_measured_a != steps_a:
        raise StepsMismatchError(
            f"declared steps_a={steps_a} but the report's own steps_measured_a="
            f"{steps_measured_a} -- the N-step run did not actually execute the declared step "
            "count"
        )
    if steps_measured_b is not None and steps_measured_b != steps_b:
        raise StepsMismatchError(
            f"declared steps_b={steps_b} but the report's own steps_measured_b="
            f"{steps_measured_b} -- the M-step run did not actually execute the declared step "
            "count"
        )

    ca, ma, _ = census(path_a)
    cb, mb, _ = census(path_b)
    d = steps_b - steps_a

    time_tolerance_ns = time_tolerance_us * 1000.0
    violations: list[str] = []

    rows = []
    fixed_cost_buckets = 0
    fixed_cost_time_ns = 0
    fixed_cost_jitter_max_rel = 0.0
    positive_count_buckets = 0
    for key in set(ca) | set(cb):
        na, nsa = ca.get(key, (0, 0))
        nb, nsb = cb.get(key, (0, 0))
        dn, dns = nb - na, nsb - nsa
        name = key[0]

        cls = _classify_delta(dn, launch_tolerance)

        if cls == "fixed_cost":
            # FIXED-COST bucket (see module doc's "EXCEPTION" paragraph):
            # either an exact dn==0 cancellation, or a launch-count jitter
            # the caller's own --launch-tolerance already treats as noise
            # -- either way this bucket carried no POSITIVE
            # differencing-relevant work, so it is excluded from the
            # per-step rows entirely (never checked against the FLAT
            # --time-tolerance-us the real added-work buckets use below).
            # Its own time delta is instead bounded by the HYBRID
            # floor/relative bound (phase-4 audit round-2 re-audit
            # BLOCK 1) -- see `_fixed_cost_violation`'s own doc.
            fixed_cost_buckets += 1
            fixed_cost_time_ns += abs(dns)
            rel, msg = _fixed_cost_violation(
                f"kernel {name}{key[1:]}",
                dn,
                dns,
                nsa,
                nsb,
                fixed_cost_jitter_floor_ns,
                fixed_cost_jitter_rel_tolerance,
            )
            fixed_cost_jitter_max_rel = max(fixed_cost_jitter_max_rel, rel)
            if msg:
                violations.append(msg)
            continue

        if cls == "positive":
            positive_count_buckets += 1
            _check_non_negative(
                f"kernel {name}{key[1:]} time (ns)", dns, time_tolerance_ns, violations
            )
            _, gx, gy, gz, bx, by, bz = key
            rows.append(
                {
                    "kernel": name,
                    "grid": [gx, gy, gz],
                    "block": [bx, by, bz],
                    "launches_per_step": dn / d,
                    "us_per_step": dns / d / 1000.0,
                }
            )
            continue

        # "negative_regression" (dn < -launch_tolerance): a real negative
        # launch-count regression -- always refuses (the fixed-cost range
        # above already claimed every dn in [-launch_tolerance, 0]).
        _check_non_negative(
            f"kernel {name}{key[1:]} launch count", dn, launch_tolerance, violations
        )

    # memcpy/memset aggregate deltas take the SAME three-way classification
    # (phase-4 audit round-2 re-audit advisory 2): a cancelling
    # (dn<=0-within-tolerance) delta is fixed-cost, checked against the
    # HYBRID bound, never the flat --time-tolerance-us (whose default of
    # 0.0 would otherwise refuse on a single nanosecond of negative
    # memcpy/memset jitter -- the driver passes neither flag).
    memcpy_dn = mb["memcpy"][0] - ma["memcpy"][0]
    memcpy_dns = mb["memcpy"][1] - ma["memcpy"][1]
    memset_dn = mb["memset"][0] - ma["memset"][0]
    memset_dns = mb["memset"][1] - ma["memset"][1]
    for label, dn, dns, ns_a, ns_b in (
        ("memcpy", memcpy_dn, memcpy_dns, ma["memcpy"][1], mb["memcpy"][1]),
        ("memset", memset_dn, memset_dns, ma["memset"][1], mb["memset"][1]),
    ):
        cls = _classify_delta(dn, launch_tolerance)
        if cls == "fixed_cost":
            _, msg = _fixed_cost_violation(
                label,
                dn,
                dns,
                ns_a,
                ns_b,
                fixed_cost_jitter_floor_ns,
                fixed_cost_jitter_rel_tolerance,
            )
            if msg:
                violations.append(msg)
        elif cls == "positive":
            _check_non_negative(f"{label} time (ns)", dns, time_tolerance_ns, violations)
        else:
            _check_non_negative(f"{label} count", dn, launch_tolerance, violations)

    if violations:
        raise NonComparablePairError(
            "one or more per-key deltas were negative beyond tolerance (or a fixed-cost "
            "bucket's time jitter exceeded the hybrid floor/relative bound) -- the two exports "
            "do not look like the declared same-workload (steps_a, steps_b) pair:\n  "
            + "\n  ".join(violations)
        )

    if positive_count_buckets == 0:
        raise EmptyDifferencedCensusError(
            "differenced census is empty -- the pair does not look like a declared M>N "
            "same-workload pair (zero kernel buckets carried a positive launch-count delta; "
            f"{fixed_cost_buckets} bucket(s) classified fixed-cost, 0 carried real added work)"
        )

    rows.sort(key=lambda r: -r["us_per_step"])
    tot_us = sum(r["us_per_step"] for r in rows)
    tot_n = sum(r["launches_per_step"] for r in rows)
    for r in rows:
        r["share"] = r["us_per_step"] / tot_us if tot_us else 0.0

    by_name: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_name[r["kernel"]][0] += r["launches_per_step"]
        by_name[r["kernel"]][1] += r["us_per_step"]
    summary = sorted(
        (
            {
                "kernel": k,
                "launches_per_step": v[0],
                "us_per_step": v[1],
                "share": v[1] / tot_us if tot_us else 0.0,
            }
            for k, v in by_name.items()
        ),
        key=lambda r: -r["us_per_step"],
    )

    report = {
        "nsys_sqlite_schema_ok": True,
        "steps_a": steps_a,
        "steps_b": steps_b,
        "steps_diff": d,
        "gpu_kernel_us_per_step": tot_us,
        "launches_per_step": tot_n,
        "memcpy_per_step": {"count": memcpy_dn / d, "us": memcpy_dns / d / 1000.0},
        "memset_per_step": {"count": memset_dn / d, "us": memset_dns / d / 1000.0},
        "by_kernel_name": summary,
        "by_kernel_and_grid": rows[:400],
        "excluded_from_chain_attribution": excluded_from_chain_attribution,
        "fixed_cost_buckets": fixed_cost_buckets,
        "fixed_cost_time_us": fixed_cost_time_ns / 1000.0,
        "fixed_cost_jitter_max_rel": fixed_cost_jitter_max_rel,
    }
    if wall_a is not None and wall_b is not None:
        report["wall_s_per_step"] = (wall_b - wall_a) / d
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="kernel_census.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s A.sqlite B.sqlite STEPS_A STEPS_B out.json "
        "[--launch-tolerance N] [--time-tolerance-us F] "
        "[--fixed-cost-jitter-floor-ns N] [--fixed-cost-jitter-rel-tolerance F] "
        "[--wall-a SECONDS --wall-b SECONDS] [--excluded-from-chain-attribution] "
        "[--steps-measured-a N --steps-measured-b M]",
    )
    ap.add_argument("sqlite_a", help="nsys sqlite export for the N-step run")
    ap.add_argument("sqlite_b", help="nsys sqlite export for the M-step run (M>N)")
    ap.add_argument("steps_a", type=int, help="measured training steps in sqlite_a (N)")
    ap.add_argument("steps_b", type=int, help="measured training steps in sqlite_b (M)")
    ap.add_argument("out_json", help="path to write the census-difference report")
    ap.add_argument(
        "--launch-tolerance",
        type=int,
        default=DEFAULT_LAUNCH_TOLERANCE,
        help=(
            "raw negative launch-count delta allowed before refusing "
            f"(default: {DEFAULT_LAUNCH_TOLERANCE})"
        ),
    )
    ap.add_argument(
        "--time-tolerance-us",
        type=float,
        default=DEFAULT_TIME_TOLERANCE_US,
        help=(
            "raw negative time delta (microseconds) allowed before refusing "
            f"(default: {DEFAULT_TIME_TOLERANCE_US})"
        ),
    )
    ap.add_argument(
        "--fixed-cost-jitter-floor-ns",
        type=float,
        default=DEFAULT_FIXED_COST_JITTER_FLOOR_NS,
        help=(
            "absolute floor (nanoseconds) below the relative bound a FIXED-COST bucket's own "
            f"time jitter must ALSO stay under before refusing (default: "
            f"{DEFAULT_FIXED_COST_JITTER_FLOOR_NS})"
        ),
    )
    ap.add_argument(
        "--fixed-cost-jitter-rel-tolerance",
        type=float,
        default=DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE,
        help=(
            "relative bound (|time delta| / max(time_a, time_b)) a FIXED-COST bucket's own "
            f"time jitter must stay under before refusing (default: "
            f"{DEFAULT_FIXED_COST_JITTER_REL_TOLERANCE})"
        ),
    )
    ap.add_argument(
        "--wall-a",
        type=float,
        default=None,
        help="the N-step run's own train_run_wall_s (seconds); requires --wall-b too",
    )
    ap.add_argument(
        "--wall-b",
        type=float,
        default=None,
        help="the M-step run's own train_run_wall_s (seconds); requires --wall-a too",
    )
    ap.add_argument(
        "--excluded-from-chain-attribution",
        action="store_true",
        help="stamp excluded_from_chain_attribution=true (E1's variable-width leg; see module doc)",
    )
    ap.add_argument(
        "--steps-measured-a",
        type=int,
        default=None,
        help="the N-step run's own report.steps_measured, cross-checked against steps_a",
    )
    ap.add_argument(
        "--steps-measured-b",
        type=int,
        default=None,
        help="the M-step run's own report.steps_measured, cross-checked against steps_b",
    )
    args = ap.parse_args(argv)

    if (args.wall_a is None) != (args.wall_b is None):
        print(
            "::error::kernel_census: --wall-a and --wall-b must be given together (or both "
            "omitted) -- refusing a one-sided wall denominator",
            file=sys.stderr,
        )
        return 2

    if args.steps_b <= args.steps_a:
        print(
            f"::error::kernel_census: steps_b ({args.steps_b}) must be strictly greater than "
            f"steps_a ({args.steps_a}) -- refusing (not a valid N<M same-workload pair)",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_report(
            args.sqlite_a,
            args.sqlite_b,
            args.steps_a,
            args.steps_b,
            launch_tolerance=args.launch_tolerance,
            time_tolerance_us=args.time_tolerance_us,
            fixed_cost_jitter_floor_ns=args.fixed_cost_jitter_floor_ns,
            fixed_cost_jitter_rel_tolerance=args.fixed_cost_jitter_rel_tolerance,
            wall_a=args.wall_a,
            wall_b=args.wall_b,
            excluded_from_chain_attribution=args.excluded_from_chain_attribution,
            steps_measured_a=args.steps_measured_a,
            steps_measured_b=args.steps_measured_b,
        )
    except KernelTableMissingError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 3
    except NonComparablePairError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 4
    except StepsMismatchError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 5
    except KernelTableEmptyError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 6
    except WallPairInvalidError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 7
    except CensusDatabaseError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 8
    except EmptyDifferencedCensusError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 9
    except ValueError as e:
        print(f"::error::kernel_census: {e}", file=sys.stderr)
        return 2

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)

    wall_note = (
        f" wall_s_per_step={report['wall_s_per_step']:.4f}"
        if "wall_s_per_step" in report
        else ""
    )
    gpu_ms = report["gpu_kernel_us_per_step"] / 1000
    print(
        f"steps_diff={report['steps_diff']} gpu_kernel_ms_per_step={gpu_ms:.1f} "
        f"launches/step={report['launches_per_step']:.0f} "
        f"memcpy/step={report['memcpy_per_step']['count']:.0f}{wall_note} "
        f"fixed_cost_buckets={report['fixed_cost_buckets']} "
        f"fixed_cost_time_us={report['fixed_cost_time_us']:.1f} "
        f"fixed_cost_jitter_max_rel={report['fixed_cost_jitter_max_rel']:.4f}"
    )
    for r in report["by_kernel_name"][:40]:
        print(
            f"{r['share']*100:5.1f}%  {r['us_per_step']/1000:7.2f} ms  "
            f"{r['launches_per_step']:7.0f}/step  {r['kernel']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
