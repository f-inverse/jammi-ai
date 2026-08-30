#!/usr/bin/env python3
"""`kernel_census.py`'s own suite (P4, CONTRACT
`scratchpad/contract-356-profile.md` v4): drives the real `build_report`/
`census` functions against a TINY synthetic sqlite fixture built in-test
(the same `CUPTI_ACTIVITY_KIND_KERNEL`/`StringIds`/`CUPTI_ACTIVITY_KIND_
MEMCPY`/`CUPTI_ACTIVITY_KIND_MEMSET` schema shape a real nsys export
carries), never a real nsys capture -- stdlib-only (`unittest`+`sqlite3`),
no network, no GPU, no `nsys` binary required.

Covers: the happy-path (M-N)-step differencing arithmetic (launches/step,
us/step, memcpy/memset/step); the missing-kernel-table refusal (leg
INVALID, no report written); the M<=N usage refusal; the negative-delta
"not a comparable pair" refusal; the v4 additions (`wall_s_per_step`
when `--wall-a`/`--wall-b` given, `excluded_from_chain_attribution`
stamping); and the phase-4 audit's CLASS 4 census domain guards: a
PRESENT-but-EMPTY kernel table, a corrupt/unreadable sqlite export, an
invalid wall pair (`wall_b > wall_a > 0` violated), and a declared-vs-
measured steps mismatch.

Run: `python3 ci/scripts/perf/test_kernel_census.py`
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_census  # noqa: E402


def _k1(launches: int, name: str = "k1") -> list[tuple]:
    """`launches` back-to-back 1000ns launches of `name` at grid
    (1,1,1)/block (32,1,1) -- the one synthetic kernel shape every test
    below needs, factored out so call sites stay under the line-length
    limit."""
    return [(name, 1, 1, 1, 32, 1, 1, i * 1000, i * 1000 + 1000) for i in range(launches)]


def _make_sqlite(
    path: str,
    kernel_rows: list[tuple],
    memcpy=(0, 0),
    memset=(0, 0),
    with_kernel_table: bool = True,
) -> None:
    """`kernel_rows`: list of (name, gx, gy, gz, bx, by, bz, start, end)."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    if with_kernel_table:
        cur.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(shortName INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
            "blockX INTEGER, blockY INTEGER, blockZ INTEGER, start INTEGER, end INTEGER)"
        )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER)"
    )

    name_to_id: dict[str, int] = {}
    next_id = 1
    for row in kernel_rows:
        name = row[0]
        if name not in name_to_id:
            name_to_id[name] = next_id
            cur.execute("INSERT INTO StringIds VALUES (?, ?)", (next_id, name))
            next_id += 1
    if with_kernel_table:
        for name, gx, gy, gz, bx, by, bz, start, end in kernel_rows:
            cur.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?)",
                (name_to_id[name], gx, gy, gz, bx, by, bz, start, end),
            )
    mcount, mtime = memcpy
    for _ in range(mcount):
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?)", (0, mtime // max(mcount, 1))
        )
    scount, stime = memset
    for _ in range(scount):
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?)", (0, stime // max(scount, 1))
        )
    con.commit()
    con.close()


class CensusHappyPathTests(unittest.TestCase):
    def test_two_step_difference_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            # a.sqlite: 10 steps -> kernel "k1" launched 10 times, 1000ns each (10000ns total).
            _make_sqlite(a, _k1(10))
            # b.sqlite: 20 steps -> kernel "k1" launched 20 times, 1000ns each (20000ns total).
            _make_sqlite(b, _k1(20))
            report = kernel_census.build_report(a, b, steps_a=10, steps_b=20)
            self.assertEqual(report["steps_diff"], 10)
            self.assertEqual(report["steps_a"], 10)
            self.assertEqual(report["steps_b"], 20)
            self.assertTrue(report["nsys_sqlite_schema_ok"])
            self.assertFalse(report["excluded_from_chain_attribution"])
            self.assertNotIn("wall_s_per_step", report)
            row = report["by_kernel_and_grid"][0]
            self.assertEqual(row["kernel"], "k1")
            self.assertAlmostEqual(row["launches_per_step"], 1.0)  # (20-10) launches / 10 steps
            self.assertAlmostEqual(row["us_per_step"], 1.0)  # (20000-10000)ns / 10 steps / 1000

    def test_wall_denominator_when_both_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(3))
            report = kernel_census.build_report(a, b, steps_a=1, steps_b=4, wall_a=1.0, wall_b=7.0)
            self.assertIn("wall_s_per_step", report)
            self.assertAlmostEqual(report["wall_s_per_step"], (7.0 - 1.0) / (4 - 1))

    def test_excluded_from_chain_attribution_stamps_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            report = kernel_census.build_report(
                a, b, steps_a=1, steps_b=2, excluded_from_chain_attribution=True
            )
            self.assertTrue(report["excluded_from_chain_attribution"])


class CensusRefusalTests(unittest.TestCase):
    def test_missing_kernel_table_refuses_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, [], with_kernel_table=False)
            _make_sqlite(b, _k1(1))
            with self.assertRaises(kernel_census.KernelTableMissingError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_m_le_n_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(1))
            with self.assertRaises(ValueError):
                kernel_census.build_report(a, b, steps_a=5, steps_b=5)
            with self.assertRaises(ValueError):
                kernel_census.build_report(a, b, steps_a=5, steps_b=4)

    def test_negative_delta_beyond_tolerance_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            # a.sqlite (the "more steps" export, mislabeled as steps_a here to
            # force a negative delta): 50 launches of k1.
            _make_sqlite(a, _k1(50))
            # b.sqlite: only 5 launches of k1 -- fewer than a, a same-workload
            # N<M pair should never see this.
            _make_sqlite(b, _k1(5))
            with self.assertRaises(kernel_census.NonComparablePairError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_negative_delta_within_tolerance_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(10))
            _make_sqlite(b, _k1(9))
            # b has ONE fewer launch (and ~1000ns less time) than a -- within
            # an explicit tolerance for both.
            report = kernel_census.build_report(
                a, b, steps_a=1, steps_b=2, launch_tolerance=1, time_tolerance_us=2.0
            )
            self.assertEqual(report["steps_diff"], 1)


class CensusFixedCostGuardTests(unittest.TestCase):
    """Round-4 pod-run fix: the first real 14-leg census hit buckets like
    `layernorm_f32(8192,1,1,32,1,1)` -- kernels the init probe / held-out
    eval launch the SAME number of times in both exports (count delta
    exactly 0) whose summed time still jitters by nanoseconds. See the
    module doc's "EXCEPTION" paragraph. Covers all three of its named
    cases: count-delta-zero jitter (allowed, excluded, tallied),
    count-delta-negative (still refused), count-delta-positive with a
    negative time delta (still refused)."""

    def test_fixed_cost_bucket_negative_time_jitter_allowed_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            # k1: the genuine training-step kernel, accumulates 10 -> 20
            # launches (a real, differencing-relevant delta).
            # k2: a FIXED-COST kernel (init probe / held-out eval) -- the
            # SAME launch count (5) in both exports, but B's summed time is
            # slightly SMALLER than A's -- nanosecond jitter, never a real
            # regression, and must never refuse.
            rows_a = _k1(10) + [
                ("k2", 1, 1, 1, 32, 1, 1, i * 1000, i * 1000 + 1000) for i in range(5)
            ]
            rows_b = _k1(20) + [
                ("k2", 1, 1, 1, 32, 1, 1, i * 1000, i * 1000 + 900) for i in range(5)
            ]
            _make_sqlite(a, rows_a)
            _make_sqlite(b, rows_b)
            report = kernel_census.build_report(a, b, steps_a=10, steps_b=20)
            self.assertEqual(report["fixed_cost_buckets"], 1)
            # 5 launches * (1000ns - 900ns) = 500ns total jitter -> 0.5us.
            self.assertAlmostEqual(report["fixed_cost_time_us"], 0.5)
            self.assertNotIn("k2", {r["kernel"] for r in report["by_kernel_and_grid"]})
            self.assertNotIn("k2", {r["kernel"] for r in report["by_kernel_name"]})

    def test_count_delta_negative_beyond_tolerance_still_refuses(self):
        # Same class the pre-existing negative-delta test already covers,
        # named explicitly per the round-4 fix's own three-way guard split:
        # a NONZERO negative count delta (not exactly 0) is never fixed-cost
        # and must still refuse.
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(10))
            _make_sqlite(b, _k1(9))  # count delta -1, not 0 -- not fixed-cost.
            with self.assertRaises(kernel_census.NonComparablePairError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_count_delta_positive_time_delta_negative_still_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            # a: 10 launches @ 1000ns each = 10000ns total.
            _make_sqlite(a, _k1(10))
            # b: 12 launches (count delta +2, real added work) but only
            # 700ns each = 8400ns total (time delta -1600ns) -- more
            # launches with LESS total time is impossible for real added
            # work, and this bucket's count delta is nonzero so it is not
            # fixed-cost either; must still refuse.
            rows_b = [("k1", 1, 1, 1, 32, 1, 1, i * 700, i * 700 + 700) for i in range(12)]
            _make_sqlite(b, rows_b)
            with self.assertRaises(kernel_census.NonComparablePairError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)


class CensusClass4GuardTests(unittest.TestCase):
    """Phase-4 adversarial audit, CLASS 4: census domain guards that
    `CensusRefusalTests` above did not yet cover."""

    def test_present_but_empty_kernel_table_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, [])  # table created, zero rows -- NOT with_kernel_table=False.
            _make_sqlite(b, _k1(1))
            with self.assertRaises(kernel_census.KernelTableEmptyError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_present_but_empty_kernel_table_on_b_also_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, [])
            with self.assertRaises(kernel_census.KernelTableEmptyError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_corrupt_database_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            # Non-empty but NOT a sqlite file at all -- sqlite3 raises
            # DatabaseError("file is not a database") the first time a real
            # query runs against it (unlike a genuinely empty/0-byte file,
            # which sqlite3 treats as a valid, table-less database).
            with open(a, "wb") as f:
                f.write(b"this is not a sqlite database, just garbage bytes\x00\x01\x02")
            _make_sqlite(b, _k1(1))
            with self.assertRaises(kernel_census.CensusDatabaseError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2)

    def test_wall_pair_zero_or_negative_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            with self.assertRaises(kernel_census.WallPairInvalidError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2, wall_a=0.0, wall_b=1.0)
            with self.assertRaises(kernel_census.WallPairInvalidError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2, wall_a=-1.0, wall_b=1.0)

    def test_wall_pair_inverted_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            with self.assertRaises(kernel_census.WallPairInvalidError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2, wall_a=5.0, wall_b=5.0)
            with self.assertRaises(kernel_census.WallPairInvalidError):
                kernel_census.build_report(a, b, steps_a=1, steps_b=2, wall_a=5.0, wall_b=4.0)

    def test_wall_pair_valid_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            report = kernel_census.build_report(
                a, b, steps_a=1, steps_b=2, wall_a=1.0, wall_b=2.0
            )
            self.assertAlmostEqual(report["wall_s_per_step"], 1.0)

    def test_steps_measured_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            with self.assertRaises(kernel_census.StepsMismatchError):
                kernel_census.build_report(
                    a, b, steps_a=100, steps_b=600, steps_measured_a=97, steps_measured_b=600
                )
            with self.assertRaises(kernel_census.StepsMismatchError):
                kernel_census.build_report(
                    a, b, steps_a=100, steps_b=600, steps_measured_a=100, steps_measured_b=599
                )

    def test_steps_measured_matching_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            report = kernel_census.build_report(
                a, b, steps_a=1, steps_b=2, steps_measured_a=1, steps_measured_b=2
            )
            self.assertEqual(report["steps_diff"], 1)

    def test_steps_measured_omitted_is_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            # No --steps-measured-a/-b at all -- a build predating that
            # field (or a caller who cannot supply it) is not refused.
            report = kernel_census.build_report(a, b, steps_a=1, steps_b=2)
            self.assertEqual(report["steps_diff"], 1)


class MainCliTests(unittest.TestCase):
    def test_main_writes_report_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            rc = kernel_census.main([a, b, "1", "2", out])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out))

    def test_main_missing_kernel_table_exit_3_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, [], with_kernel_table=False)
            _make_sqlite(b, _k1(1))
            rc = kernel_census.main([a, b, "1", "2", out])
            self.assertEqual(rc, 3)
            self.assertFalse(os.path.exists(out))

    def test_main_m_le_n_exit_2_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(1))
            rc = kernel_census.main([a, b, "5", "5", out])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(out))

    def test_main_wall_one_sided_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            rc = kernel_census.main([a, b, "1", "2", out, "--wall-a", "1.0"])
            self.assertEqual(rc, 2)

    def test_main_steps_mismatch_exit_5_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            rc = kernel_census.main(
                [a, b, "100", "600", out, "--steps-measured-a", "97", "--steps-measured-b", "600"]
            )
            self.assertEqual(rc, 5)
            self.assertFalse(os.path.exists(out))

    def test_main_empty_kernel_table_exit_6_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, [])
            _make_sqlite(b, _k1(1))
            rc = kernel_census.main([a, b, "1", "2", out])
            self.assertEqual(rc, 6)
            self.assertFalse(os.path.exists(out))

    def test_main_wall_pair_invalid_exit_7_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            _make_sqlite(a, _k1(1))
            _make_sqlite(b, _k1(2))
            rc = kernel_census.main([a, b, "1", "2", out, "--wall-a", "5.0", "--wall-b", "5.0"])
            self.assertEqual(rc, 7)
            self.assertFalse(os.path.exists(out))

    def test_main_corrupt_database_exit_8_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.sqlite")
            b = os.path.join(tmp, "b.sqlite")
            out = os.path.join(tmp, "out.json")
            with open(a, "wb") as f:
                f.write(b"garbage, not a sqlite file")
            _make_sqlite(b, _k1(1))
            rc = kernel_census.main([a, b, "1", "2", out])
            self.assertEqual(rc, 8)
            self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
