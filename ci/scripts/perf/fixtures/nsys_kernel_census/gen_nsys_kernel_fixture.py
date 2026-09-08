#!/usr/bin/env python3
"""Generator for the committed `n.sqlite`/`m.sqlite` fixtures this
directory ships (esc-088 round-4 advisory, folded into #421's
follow-ups): a TINY, schema-valid nsys sqlite export -- the same
`CUPTI_ACTIVITY_KIND_KERNEL` (joined to `StringIds` on BOTH `shortName`
AND `demangledName`, the real nsys 2025.3.2 export shape) /
`CUPTI_ACTIVITY_KIND_MEMCPY` / `CUPTI_ACTIVITY_KIND_MEMSET` shape
`kernel_census.py::census`/`ci/scripts/perf/test_kernel_census.py`'s own
`_make_sqlite` helper build -- committed so `profile_421_legs.sh`'s
DRY_RUN stub can copy a REAL sqlite export to the `nsys export` output
path instead of touch-emptying it, letting `kernel_census.py` run for
real, hermetically, against real (if synthetic) kernel rows.

`CUPTI_ACTIVITY_KIND_KERNEL` carries TWO string-id columns, not one:
`shortName` (the bare kernel symbol, no template arguments -- e.g.
`fixture_attention_block_kernel`) and `demangledName` (the full
demangled signature `census()`'s own query keys on, per its module
doc's "Kernel identity" paragraph -- e.g. `void jammi_kernels::
fixture_attention_block_kernel(float const*, float*)`). Both columns
are present and hold DISTINCT StringIds rows, each carrying the string
its own name promises -- a fixture that collapsed the two (storing one
string and using it as both, or storing only a `demangledName`-shaped
value AS `shortName`) would not be the shape a real nsys export takes
(a genuine `shortName` is never a full signature) and would make
`census()`'s `k.demangledName` reference fail outright against a
schema that never carried that column.

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
counts into a temp dir and checks four independent things, none of which
is "the committed files are byte-identical to a fresh regeneration" --
raw sqlite file bytes are NOT a stable function of logical content across
sqlite library versions (header fields, default page size, freelist
bookkeeping, and page layout all vary release to release even for two
files holding byte-for-byte identical rows), so a CI runner's system
`sqlite3` need not produce the same bytes as the developer Mac's even
when nothing about this generator or its fixtures has drifted:

  1. GENERATOR determinism: two independent fresh regenerations, on the
     SAME sqlite library, of each count ARE asserted byte-identical --
     this proves the generator itself is deterministic (no RNG, no
     wall-clock read, fixed StringIds ids and row order), a claim that
     is honest to make within one library version, unlike a cross-version
     byte-identity claim.
  2. `kernel_census.build_report` accepts a fresh-built pair cleanly
     (mirrors this file's own real call site in `profile_421_legs.sh`'s
     DRY_RUN stub).
  3. LOGICAL identity between a fresh regeneration and each of the two
     committed files this directory ships (`n.sqlite`/`m.sqlite`): the
     `sqlite_master` schema text (normalized) and every table's full row
     set (canonically ordered) must match exactly -- this is the actual
     "did someone forget to regenerate the committed fixture" oracle,
     and it survives a benign byte-level-only difference (a different
     `PRAGMA user_version`, a `VACUUM`ed copy) between two files with the
     same logical content.
  4. Consumer usability: `kernel_census.build_report` run directly over
     the COMMITTED files (never the fresh copies) must yield the exact
     expected census -- proves the committed bytes are actually usable
     by the real consumer today, not merely "regenerable to the same
     logical content" in the abstract.

A final internal meta-check drives (3)'s own comparator against a
scratch row-mutated copy and a scratch schema-altered copy of a
committed file (each must RED) and against a scratch `PRAGMA
user_version`-changed copy and a scratch `VACUUM`ed copy (each must stay
GREEN) -- so the oracle's own discriminating power is itself gated, not
merely asserted in a comment. Wired into `.github/workflows/ci.yml`
("perf nsys kernel fixture self-test"), not a doc claim a future edit to
this generator (or a regeneration someone forgets to run) can silently
invalidate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile

KERNEL_SHORT_NAME = "fixture_attention_block_kernel"
KERNEL_DEMANGLED_NAME = "void jammi_kernels::fixture_attention_block_kernel(float const*, float*)"
GRID = (1, 1, 1)
BLOCK = (32, 1, 1)
LAUNCH_DURATION_NS = 1000


def build(path: str, launches: int) -> None:
    """Writes a schema-valid, single-kernel-bucket nsys sqlite export with
    exactly `launches` back-to-back `LAUNCH_DURATION_NS`-long launches of
    `KERNEL_SHORT_NAME`/`KERNEL_DEMANGLED_NAME` at grid `GRID`/block
    `BLOCK`, plus empty (but present) memcpy/memset tables. The kernel
    table carries BOTH `shortName` and `demangledName` string-id columns
    (real nsys 2025.3.2 shape; see module doc) as two DISTINCT StringIds
    rows -- `shortName` the bare symbol, `demangledName` the full
    signature `census()` keys on. Overwrites `path` if it already
    exists."""
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        cur.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(shortName INTEGER, demangledName INTEGER, gridX INTEGER, gridY INTEGER, "
            "gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER, "
            "start INTEGER, end INTEGER)"
        )
        cur.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)")
        cur.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER)")
        cur.execute("INSERT INTO StringIds (id, value) VALUES (1, ?)", (KERNEL_SHORT_NAME,))
        cur.execute("INSERT INTO StringIds (id, value) VALUES (2, ?)", (KERNEL_DEMANGLED_NAME,))
        gx, gy, gz = GRID
        bx, by, bz = BLOCK
        rows = [
            (1, 2, gx, gy, gz, bx, by, bz, i * LAUNCH_DURATION_NS, (i + 1) * LAUNCH_DURATION_NS)
            for i in range(launches)
        ]
        cur.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        con.commit()
    finally:
        con.close()


_FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
_N_LAUNCHES = 5
_M_LAUNCHES = 30


def _normalize_sql(sql: str | None) -> str:
    """Collapses whitespace in a `sqlite_master.sql` `CREATE TABLE` string
    so an incidental reformatting (never observed from this module's own
    literal DDL strings, but not something this comparator should rely on)
    does not register as a logical schema difference."""
    return " ".join((sql or "").split())


def _logical_snapshot(path: str) -> tuple[dict[str, str], dict[str, list[tuple]]]:
    """Reads `path` as a sqlite file and returns `(schema, data)`:
    `schema` maps each table name to its normalized `CREATE TABLE` text
    (`sqlite_master.sql`), `data` maps each table name to its FULL row set
    in a CANONICAL (sorted-tuple) order -- independent of insertion order,
    rowid assignment, page layout, or any other on-disk-only detail. Two
    sqlite files with equal `(schema, data)` snapshots describe the
    identical logical content even if their raw bytes differ (a different
    `PRAGMA user_version`, a `VACUUM`ed copy, a different sqlite library
    version's header/page-layout choices)."""
    con = sqlite3.connect(path)
    try:
        names = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        schema: dict[str, str] = {}
        data: dict[str, list[tuple]] = {}
        for name in names:
            (sql,) = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            schema[name] = _normalize_sql(sql)
            # `name` is read from this same file's own `sqlite_master`, never caller input.
            rows = con.execute(f"SELECT * FROM {name}").fetchall()
            data[name] = sorted(rows)
        return schema, data
    finally:
        con.close()


def _assert_logically_identical(path_a: str, path_b: str, label: str) -> None:
    """Raises `AssertionError` unless `path_a` and `path_b` have the exact
    same `_logical_snapshot` -- the SCHEMA and ROW-DATA identity oracle
    this module's self-test uses in place of raw byte equality (see module
    doc for why byte equality is not a claim this generator can honestly
    make across sqlite library versions)."""
    schema_a, data_a = _logical_snapshot(path_a)
    schema_b, data_b = _logical_snapshot(path_b)
    if schema_a != schema_b:
        raise AssertionError(
            f"{label}: SCHEMA differs between {path_a!r} and {path_b!r} -- {schema_a!r} != "
            f"{schema_b!r}"
        )
    if data_a != data_b:
        raise AssertionError(
            f"{label}: ROW DATA differs between {path_a!r} and {path_b!r} -- one or more "
            "tables' row sets do not match"
        )


def _self_test_oracle_meta(committed_path: str) -> None:
    """Proves `_assert_logically_identical` itself discriminates real
    drift from benign byte-level-only differences, against scratch copies
    of `committed_path` (never the committed file itself, which is never
    mutated): a row-mutated copy and a schema-altered copy must each RED
    (raise `AssertionError`); a copy with a different `PRAGMA user_version`
    and a `VACUUM`ed copy -- two ways to change a sqlite file's raw bytes
    without changing its logical content -- must each stay GREEN (compare
    logically identical to the original). Without this, the comparator
    above could silently degrade into a no-op (e.g. a typo that always
    returns without raising) and every other self-test assertion using it
    would keep passing for the wrong reason."""
    with tempfile.TemporaryDirectory() as tmp:
        mutated = os.path.join(tmp, "row_mutated.sqlite")
        shutil.copyfile(committed_path, mutated)
        con = sqlite3.connect(mutated)
        try:
            con.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end = end + 1 WHERE rowid = 1")
            con.commit()
        finally:
            con.close()
        try:
            _assert_logically_identical(committed_path, mutated, "meta-row-mutation")
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "gen_nsys_kernel_fixture self-test meta-check: a row-mutated scratch copy "
                "compared logically IDENTICAL to the committed fixture -- "
                "_assert_logically_identical is not detecting row-level drift"
            )

        schema_changed = os.path.join(tmp, "schema_changed.sqlite")
        shutil.copyfile(committed_path, schema_changed)
        con = sqlite3.connect(schema_changed)
        try:
            con.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN extra INTEGER")
            con.commit()
        finally:
            con.close()
        try:
            _assert_logically_identical(committed_path, schema_changed, "meta-schema-change")
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "gen_nsys_kernel_fixture self-test meta-check: a schema-altered scratch copy "
                "compared logically IDENTICAL to the committed fixture -- "
                "_assert_logically_identical is not detecting schema drift"
            )

        user_version_changed = os.path.join(tmp, "user_version_changed.sqlite")
        shutil.copyfile(committed_path, user_version_changed)
        con = sqlite3.connect(user_version_changed)
        try:
            con.execute("PRAGMA user_version = 42")
            con.commit()
        finally:
            con.close()
        _assert_logically_identical(committed_path, user_version_changed, "meta-user-version")

        vacuumed = os.path.join(tmp, "vacuumed.sqlite")
        shutil.copyfile(committed_path, vacuumed)
        con = sqlite3.connect(vacuumed)
        try:
            con.execute("VACUUM")
        finally:
            con.close()
        _assert_logically_identical(committed_path, vacuumed, "meta-vacuum")


def self_test() -> int:
    perf_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, perf_dir)
    import kernel_census  # noqa: E402  -- perf dir, added to sys.path just above

    committed_n = os.path.join(_FIXTURE_DIR, "n.sqlite")
    committed_m = os.path.join(_FIXTURE_DIR, "m.sqlite")

    with tempfile.TemporaryDirectory() as tmp:
        n_path = os.path.join(tmp, "n.sqlite")
        m_path = os.path.join(tmp, "m.sqlite")
        build(n_path, _N_LAUNCHES)
        build(m_path, _M_LAUNCHES)

        # (1) Generator determinism: two independent fresh regenerations of
        # each count, on THIS SAME sqlite library, must be byte-identical --
        # proves the generator's own determinism (no RNG, no wall-clock
        # read, fixed StringIds ids/row order). This is deliberately NOT
        # the oracle for "is the committed fixture stale" (see (3) below) --
        # it only ever compares two builds made moments apart by the same
        # process, so it cannot mask cross-library byte drift as a false
        # negative.
        n_path_repeat = os.path.join(tmp, "n_repeat.sqlite")
        m_path_repeat = os.path.join(tmp, "m_repeat.sqlite")
        build(n_path_repeat, _N_LAUNCHES)
        build(m_path_repeat, _M_LAUNCHES)
        for a, b, label in (
            (n_path, n_path_repeat, "n.sqlite"),
            (m_path, m_path_repeat, "m.sqlite"),
        ):
            with open(a, "rb") as fa, open(b, "rb") as fb:
                assert fa.read() == fb.read(), (
                    f"{label}: two fresh regenerations on this SAME sqlite library produced "
                    "different bytes -- the generator itself is not deterministic (check for "
                    "RNG, a wall-clock read, or non-fixed StringIds/row ordering)"
                )

        # (2) `kernel_census.build_report` accepts a fresh-built pair
        # cleanly -- mirrors this file's own real call site in
        # `profile_421_legs.sh`'s DRY_RUN stub.
        report = kernel_census.build_report(
            n_path, m_path, steps_a=100, steps_b=600, wall_a=1.0, wall_b=6.0
        )
        assert report["nsys_sqlite_schema_ok"] is True, report
        assert report["launches_per_step"] > 0, report
        assert report["fixed_cost_buckets"] == 0, report

        # (3) LOGICAL identity (schema + full row set, canonically ordered)
        # between a fresh regeneration and each committed file -- the
        # actual "is the committed fixture stale relative to this
        # generator" oracle. Deliberately NOT raw byte equality: sqlite
        # file bytes are not a stable function of logical content across
        # sqlite library versions (header fields, page layout, freelist
        # bookkeeping all vary release to release), so a byte-equality
        # check here would be a false claim about cross-version stability,
        # not merely a strict one -- see the module doc.
        _assert_logically_identical(committed_n, n_path, "n.sqlite")
        _assert_logically_identical(committed_m, m_path, "m.sqlite")

        # (4) Consumer usability: `kernel_census.build_report` run directly
        # over the COMMITTED files (never the fresh copies built above)
        # must yield the EXACT expected census -- proves the committed
        # bytes this directory ships are actually usable by the real
        # consumer today, not merely "regenerable to the same logical
        # content" in the abstract. `steps_a=0, steps_b=1` (steps_diff=1)
        # makes `launches_per_step` land exactly on the raw launch-count
        # delta (`_M_LAUNCHES - _N_LAUNCHES == 25`) with no per-step
        # division to reason through.
        expected_dn = _M_LAUNCHES - _N_LAUNCHES
        committed_report = kernel_census.build_report(
            committed_n, committed_m, steps_a=0, steps_b=1
        )
        assert committed_report["steps_diff"] == 1, committed_report
        assert committed_report["launches_per_step"] == expected_dn, committed_report
        assert committed_report["fixed_cost_buckets"] == 0, committed_report
        by_name = committed_report["by_kernel_name"]
        assert len(by_name) == 1, by_name
        assert by_name[0]["kernel"] == KERNEL_DEMANGLED_NAME, by_name
        assert by_name[0]["launches_per_step"] == expected_dn, by_name
        by_grid = committed_report["by_kernel_and_grid"]
        assert len(by_grid) == 1, by_grid
        assert by_grid[0]["kernel"] == KERNEL_DEMANGLED_NAME, by_grid
        assert by_grid[0]["grid"] == list(GRID), by_grid
        assert by_grid[0]["block"] == list(BLOCK), by_grid

        # Meta-check: the oracle used in (3) actually discriminates real
        # drift (row/schema mutation, must RED) from benign byte-level-only
        # differences (a different PRAGMA user_version, a VACUUMed copy,
        # must stay GREEN) -- see `_self_test_oracle_meta`'s own doc.
        _self_test_oracle_meta(committed_n)

        print("gen_nsys_kernel_fixture: self-test OK", report, committed_report)
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
