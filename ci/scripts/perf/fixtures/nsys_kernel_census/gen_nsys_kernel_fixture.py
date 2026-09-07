#!/usr/bin/env python3
"""Generator for the committed `n.sqlite`/`m.sqlite` fixtures this
directory ships (esc-088 round-4 advisory, folded into #421's
follow-ups): a TINY, schema-valid nsys sqlite export -- the same
`CUPTI_ACTIVITY_KIND_KERNEL` (joined to `StringIds` on `shortName`) /
`CUPTI_ACTIVITY_KIND_MEMCPY` / `CUPTI_ACTIVITY_KIND_MEMSET` shape
`kernel_census.py::census`/`ci/scripts/perf/test_kernel_census.py`'s own
`_make_sqlite` helper build -- committed so `profile_421_legs.sh`'s
DRY_RUN stub can copy a REAL sqlite export to the `nsys export` output
path instead of touch-emptying it, letting `kernel_census.py` run for
real, hermetically, against real (if synthetic) kernel rows.

Deterministic and reproducible by construction: no RNG, no wall-clock
read, no environment-dependent ordering -- fixed kernel name, fixed
grid/block shape, fixed per-launch duration, launch COUNT the only
parameter. Regenerate both committed files with:

    python3 ci/scripts/perf/fixtures/nsys_kernel_census/gen_nsys_kernel_fixture.py \\
        --launches 5  --out ci/scripts/perf/fixtures/nsys_kernel_census/n.sqlite
    python3 ci/scripts/perf/fixtures/nsys_kernel_census/gen_nsys_kernel_fixture.py \\
        --launches 30 --out ci/scripts/perf/fixtures/nsys_kernel_census/m.sqlite

`n.sqlite` (5 launches) and `m.sqlite` (30 launches) give
`kernel_census.py` a genuine POSITIVE launch-count delta (dn=25) on one
kernel bucket -- enough to clear `EmptyDifferencedCensusError` (at least
one bucket must carry real added work) without tripping any of its
negative-delta/fixed-cost-jitter guards (memcpy/memset tables are present
but empty, so their own count deltas are exactly 0, never negative).

Self-test: `python3 gen_nsys_kernel_fixture.py --self-test` builds both
counts into a temp dir and asserts `kernel_census.build_report` accepts
the resulting pair cleanly (mirrors this file's own real call site in
`profile_421_legs.sh`'s DRY_RUN stub).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile

KERNEL_NAME = "void jammi_kernels::fixture_attention_block_kernel(float const*, float*)"
GRID = (1, 1, 1)
BLOCK = (32, 1, 1)
LAUNCH_DURATION_NS = 1000


def build(path: str, launches: int) -> None:
    """Writes a schema-valid, single-kernel-bucket nsys sqlite export with
    exactly `launches` back-to-back `LAUNCH_DURATION_NS`-long launches of
    `KERNEL_NAME` at grid `GRID`/block `BLOCK`, plus empty (but present)
    memcpy/memset tables. Overwrites `path` if it already exists."""
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        cur.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(shortName INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
            "blockX INTEGER, blockY INTEGER, blockZ INTEGER, start INTEGER, end INTEGER)"
        )
        cur.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)")
        cur.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER)")
        cur.execute("INSERT INTO StringIds (id, value) VALUES (1, ?)", (KERNEL_NAME,))
        gx, gy, gz = GRID
        bx, by, bz = BLOCK
        rows = [
            (1, gx, gy, gz, bx, by, bz, i * LAUNCH_DURATION_NS, (i + 1) * LAUNCH_DURATION_NS)
            for i in range(launches)
        ]
        cur.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        con.commit()
    finally:
        con.close()


def self_test() -> int:
    perf_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, perf_dir)
    import kernel_census  # noqa: E402  -- perf dir, added to sys.path just above

    with tempfile.TemporaryDirectory() as tmp:
        n_path = os.path.join(tmp, "n.sqlite")
        m_path = os.path.join(tmp, "m.sqlite")
        build(n_path, 5)
        build(m_path, 30)
        report = kernel_census.build_report(
            n_path, m_path, steps_a=100, steps_b=600, wall_a=1.0, wall_b=6.0
        )
        assert report["nsys_sqlite_schema_ok"] is True, report
        assert report["launches_per_step"] > 0, report
        assert report["fixed_cost_buckets"] == 0, report
        print("gen_nsys_kernel_fixture: self-test OK", report)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launches", type=int, help="number of kernel launches to write")
    ap.add_argument("--out", help="output sqlite path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.launches is None or not args.out:
        ap.error("--launches and --out are required unless --self-test is given")
    build(args.out, args.launches)
    print(f"gen_nsys_kernel_fixture: wrote {args.launches} launches to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
