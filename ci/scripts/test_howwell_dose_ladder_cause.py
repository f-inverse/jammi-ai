#!/usr/bin/env python3
"""Hermetic `unittest` suite for `howwell_dose_ladder_cause.py` (unit-63
round-13 audit F1) -- drives the real `dose_ladder_cause` pure function
against in-memory synthetic `finetune_run_ab_report.json`-shaped dicts,
mirroring `test_check_kernel_oracles.py`'s own "drive the real entry points
against throwaway fixtures" shape for this repo's `test_*.py` gate-suite
convention.

Run: `python3 ci/scripts/test_howwell_dose_ladder_cause.py`
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "perf"))

import ab_merge  # noqa: E402
import howwell_dose_ladder_cause as namer  # noqa: E402

_SCRIPT = str(Path(__file__).resolve().parent / "howwell_dose_ladder_cause.py")
_HOWWELL_SH = Path(__file__).resolve().parent / "runpod_gpu_howwell.sh"


def _extract_status_case_arm_groups(script_text: str) -> list[frozenset[str]]:
    """Mechanically extracts `runpod_gpu_howwell.sh`'s own
    `case "$STATUS" in ... esac` arm patterns -- never a hand-copied literal
    set here, so a shell-side edit to the case block is picked up the next
    time this test runs (unit-63 round-15 audit, round-14 F6 sibling class).

    Returns one `frozenset` of literal status names per case arm, in arm
    order, EXCLUDING the catch-all `*)` arm (which by construction names no
    specific status) -- a `|`-joined arm like `RED|RED_FOR_INVESTIGATION|
    INVALID)` becomes one three-member set; `GREEN)` becomes one
    single-member set. Raises `AssertionError` if no `case "$STATUS"` block
    is found at all (a moved/renamed case statement must fail this test
    loudly, never silently report an empty, vacuously-passing set).
    """
    block = re.search(r'case\s+"\$STATUS"\s+in\n(.*?\n)\s*esac\b', script_text, re.DOTALL)
    if block is None:
        raise AssertionError('no `case "$STATUS" in ... esac` block found in runpod_gpu_howwell.sh')
    groups = []
    for line in block.group(1).splitlines():
        arm = re.match(r"^\s*([A-Za-z0-9_|]+)\)", line)
        if arm is None:
            continue
        pattern = arm.group(1)
        if pattern == "*":
            continue
        groups.append(frozenset(pattern.split("|")))
    return groups


class DoseLadderCauseTests(unittest.TestCase):
    def test_red_proof_only_cause(self):
        # unit-63 round-13 audit F1's own named failure shape: primary
        # decision GREEN, RED-proof undischarged, no other dose-ladder
        # cause present. Pre-fix (74fd69ef), this fell through to the
        # "unknown" fallback -- the exact unexplained-contradiction shape
        # this namer exists to prevent.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "redproof-nobc", "detected": "not-detected"}],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", cause)
        self.assertNotIn("unknown", cause)

    def test_mixed_causes_all_named(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": "dose_label 'eps-bogus' failed to parse",
                "dose_anomalies": [{"dose_label": "eps-0.50"}],
                "doses": [
                    {"dose_label": "eps-0.10", "detected": "INVALID"},
                    {"dose_label": "redproof-nobc", "detected": "not-detected"},
                ],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("sensitivity_error", cause)
        self.assertIn("invalid_doses=eps-0.10", cause)
        self.assertIn("dose_anomalies", cause)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", cause)

    def test_proven_red_proof_never_named_as_a_cause(self):
        # PROVEN contributes nothing to ab_merge.py's own exit code (CONTRACT
        # F4) -- the namer must never name a PROVEN red_proof_verdict as a
        # GREEN-but-nonzero cause.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": "dose_label 'eps-bogus' failed to parse",
                "dose_anomalies": [],
                "doses": [],
                "red_proof_verdict": "PROVEN",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("sensitivity_error", cause)
        self.assertNotIn("red_proof_verdict", cause)

    def test_all_clear_fallback_enumerates_all_four_causes(self):
        # unit-63 round-13 audit F1: the fallback text must name every
        # cause class this namer checked, not just the eps-family three --
        # a bare "unknown" (pre-fix) looks like this namer forgot to check
        # something, rather than affirmatively ruling all four out.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "eps0.50", "detected": "RED"}],
                "red_proof_verdict": None,
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("unknown", cause)
        self.assertIn("dose_anomalies", cause)
        self.assertIn("sensitivity_error", cause)
        self.assertIn("invalid dose column", cause)
        self.assertIn("red_proof_verdict", cause)

    def test_no_mutant_dose_ladder_key_falls_back_cleanly(self):
        cause = namer.dose_ladder_cause({"status": "GREEN"})
        self.assertIn("unknown", cause)


class DoseLadderCauseNamesBoundToAbMergeExitFoldTests(unittest.TestCase):
    """Unit-63 round-14 audit F6: the namer's own checked-cause set must
    equal `ab_merge.py`'s own `main()` dose-ladder exit-fold cause set --
    imports BOTH modules and asserts equality, so a fifth cause added to one
    side without the other is a RED test here, never silent drift (the prior
    state: `_ALL_CAUSE_NAMES`'s own comment CLAIMED this with nothing
    mechanical enforcing it).
    """

    def test_namer_cause_names_equal_ab_merge_dose_ladder_exit_cause_names(self):
        self.assertEqual(set(namer._ALL_CAUSE_NAMES), set(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES))

    def test_namer_reuses_the_constant_directly_never_a_hand_duplicated_literal(self):
        # The strongest binding available: literally the same list contents,
        # imported from the one place `ab_merge.py`'s own exit fold is
        # itself asserted against (see that module's own `main()` doc).
        self.assertEqual(list(namer._ALL_CAUSE_NAMES), list(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES))

    def test_exactly_four_causes_today(self):
        # Pinned count -- a change here is a real fifth-cause addition (or a
        # removal), never an incidental refactor; re-derive, never bump to
        # make this test pass.
        self.assertEqual(len(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES), 4)


class ShellStatusCaseArmsBoundToFinetuneRunStatusesTests(unittest.TestCase):
    """Unit-63 round-15 audit (docs-ci preemptive sweep after round-14 F6):
    `runpod_gpu_howwell.sh`'s own `case "$STATUS"` arms (~lines 208-244)
    hand-copy `ab_merge.py`'s `build_finetune_run_report`'s finetune-run
    status vocabulary (`RED|RED_FOR_INVESTIGATION|INVALID` / `GREEN` /
    `DRY_RUN|INCOMPLETE`) with no mechanical oracle binding them to that
    module's own `FINETUNE_RUN_STATUSES` constant -- exactly the round-14 F6
    class ("one capability enumerated by hand in two modules with no
    mechanical oracle"), one release before an auditor would have had to
    name it a second time. A merger status added on the Python side without
    a matching shell arm falls through to that case block's own `*) ...
    unrecognised` warning arm -- fail-legible, but ungated drift.

    This suite parses the shell script's own `case` arms MECHANICALLY (see
    `_extract_status_case_arm_groups`, never a hand-copied literal set here)
    and asserts they exactly cover `ab_merge.FINETUNE_RUN_STATUSES` (no
    extras, no gaps), AND that the arm GROUPING itself matches the
    constant's own gating/green/record-only partition -- a status that
    landed in set-equality-satisfying but wrongly-grouped shell arm (e.g.
    spliced into the wrong case arm) would still pass a bare set-equality
    check.

    What this oracle does NOT cover: a `$STATUS`/`status` value consumed
    ANYWHERE ELSE outside this one case block -- e.g. embedded in some other
    script's own log/error text, or read by a different consumer entirely.
    Only THIS case block's own named arms are bound to
    `ab_merge.FINETUNE_RUN_STATUSES` here.
    """

    def test_shell_case_arms_exactly_cover_finetune_run_statuses(self):
        groups = _extract_status_case_arm_groups(_HOWWELL_SH.read_text())
        shell_names = frozenset().union(*groups) if groups else frozenset()
        self.assertEqual(shell_names, frozenset(ab_merge.FINETUNE_RUN_STATUSES))

    def test_shell_arm_grouping_matches_the_gating_green_record_only_partition(self):
        groups = _extract_status_case_arm_groups(_HOWWELL_SH.read_text())
        self.assertIn(frozenset(ab_merge.FINETUNE_RUN_GATING_STATUSES), groups)
        self.assertIn(frozenset({ab_merge.FINETUNE_RUN_GREEN_STATUS}), groups)
        self.assertIn(frozenset(ab_merge.FINETUNE_RUN_RECORD_ONLY_STATUSES), groups)

    def test_no_status_named_in_two_different_shell_arms(self):
        groups = _extract_status_case_arm_groups(_HOWWELL_SH.read_text())
        seen = set()
        for group in groups:
            self.assertTrue(seen.isdisjoint(group), f"status named in >1 case arm: {seen & group}")
            seen |= group

    def test_exactly_six_statuses_today(self):
        # Pinned count (mirrors DoseLadderCauseNamesBoundToAbMergeExitFoldTests's
        # own `test_exactly_four_causes_today`) -- a change here is a real
        # seventh-status addition (or removal), never an incidental refactor;
        # re-derive, never bump to make this test pass.
        self.assertEqual(len(ab_merge.FINETUNE_RUN_STATUSES), 6)


class DosesFieldHardeningTests(unittest.TestCase):
    """Unit-63 round-14 audit A4: `ladder["doses"]` is a producer/merger
    artifact field, never assumed well-shaped -- `null`, a non-list value, or
    a list carrying a `null`/non-dict element must degrade to a NAMED cause,
    never an uncaught exception the shell's own `2>/dev/null || echo
    "unknown (could not inspect ...)"` fallback would silently swallow into
    an opaque, indistinguishable-from-"nothing wrong" "unknown".
    """

    def test_doses_field_absent_is_not_a_malformation(self):
        report = {"status": "GREEN", "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": []}}
        cause = namer.dose_ladder_cause(report)
        self.assertIn("unknown", cause)
        self.assertNotIn("malformed", cause)
        self.assertNotIn("doses_field", cause)

    def test_doses_field_null_degrades_to_a_named_cause_never_a_crash(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": None},
        }
        cause = namer.dose_ladder_cause(report)  # must not raise
        self.assertIn("doses_field_is_null", cause)

    def test_doses_field_not_a_list_degrades_to_a_named_cause(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": "not-a-list"},
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_is_not_a_list", cause)
        self.assertIn("type=str", cause)

    def test_doses_field_with_null_elements_degrades_to_a_named_cause_and_still_scans_the_rest(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [None, {"dose_label": "eps-0.10", "detected": "INVALID"}, "also-not-a-dict"],
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_has_2_malformed_entries", cause)
        self.assertIn("invalid_doses=eps-0.10", cause)

    def test_doses_field_with_one_null_element_uses_singular_wording(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": [None]},
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_has_1_malformed_entry", cause)
        self.assertNotIn("entries", cause)


class MainEntryPointTests(unittest.TestCase):
    """Unit-63 round-14 audit A5: `main()` (the actual CLI entry
    `runpod_gpu_howwell.sh` invokes) had zero execution coverage -- every
    existing test drove `dose_ladder_cause` directly. Covers argv handling,
    a missing file, and a valid file, via BOTH a real subprocess invocation
    (the exact shape `runpod_gpu_howwell.sh` uses) and a direct `main()`
    call (for exit-code assertions without process-spawn overhead).
    """

    def test_wrong_argv_count_prints_usage_and_returns_2(self):
        self.assertEqual(namer.main([]), 2)
        self.assertEqual(namer.main(["a", "b"]), 2)

    def test_missing_file_subprocess_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, _SCRIPT, "/nonexistent/path/does-not-exist.json"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_valid_file_subprocess_prints_cause_and_exits_zero(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "redproof-nobc", "detected": "not-detected"}],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(report, fh)
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, _SCRIPT, path], capture_output=True, text=True)
        finally:
            Path(path).unlink()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", proc.stdout)

    def test_wrong_argv_count_subprocess_exits_2_with_usage_on_stderr(self):
        proc = subprocess.run([sys.executable, _SCRIPT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)

    def test_main_direct_call_matches_dose_ladder_cause_output(self):
        # A direct `main()` call over a real file, cross-checked against
        # `dose_ladder_cause` called directly on the same dict -- proves
        # `main()` is not a second, independently-drifting read path.
        report = {"status": "GREEN", "mutant_dose_ladder": {"sensitivity_error": "boom", "dose_anomalies": []}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(report, fh)
            path = fh.name
        try:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = namer.main([path])
        finally:
            Path(path).unlink()
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), namer.dose_ladder_cause(report))


class AbMergeImportFailureHardeningTests(unittest.TestCase):
    """Unit-63 round-15 audit advisory 3: `howwell_dose_ladder_cause.py`'s
    own module-level `sys.path.insert(0, .../perf); import ab_merge` is a
    crash surface upstream of `_inspect_doses`'s own A4 hardening -- an
    import-time failure (a broken `perf/ab_merge.py`, or the module simply
    missing) must degrade to a NAMED cause on stdout, exit 0, never an
    uncaught exception that `runpod_gpu_howwell.sh`'s own
    `2>/dev/null || echo "unknown (could not inspect ...)"` wrapper would
    collapse into the opaque "unknown" text. Proven here by copying the real
    `howwell_dose_ladder_cause.py` into a throwaway directory alongside a
    deliberately broken `perf/ab_merge.py` stub and invoking it as a real
    subprocess -- the exact shape `runpod_gpu_howwell.sh` uses, with the
    ONE variable under test (whether `import ab_merge` succeeds) swapped
    out, never the real `ci/scripts/perf/ab_merge.py` touched.

    Pre-fix (803ae6c7), this same setup crashed with an uncaught
    `ImportError`, non-zero exit, and empty stdout -- captured RED before
    this test's own fix landed.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="howwell-dose-ladder-cause-import-fail-")
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
        shutil.copy(_SCRIPT, Path(self._tmpdir) / "howwell_dose_ladder_cause.py")
        perf_dir = Path(self._tmpdir) / "perf"
        perf_dir.mkdir()
        (perf_dir / "ab_merge.py").write_text(
            'raise ImportError("simulated broken ab_merge -- round-15 audit advisory 3 RED-proof")\n'
        )
        self._script = str(Path(self._tmpdir) / "howwell_dose_ladder_cause.py")

    def _run(self, report: dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=self._tmpdir) as fh:
            json.dump(report, fh)
            path = fh.name
        return subprocess.run([sys.executable, self._script, path], capture_output=True, text=True)

    def test_broken_ab_merge_degrades_to_a_named_cause_not_a_crash(self):
        report = {"status": "GREEN", "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": []}}
        proc = self._run(report)
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("ab_merge_import_failed(", proc.stdout)
        self.assertIn("simulated broken ab_merge", proc.stdout)
        self.assertNotEqual(proc.stdout.strip(), "")

    def test_broken_ab_merge_never_produces_bare_unknown(self):
        # The whole point: a real crash here must not collapse into the
        # SAME opaque text a genuinely-no-cause-found run produces.
        report = {"status": "GREEN", "mutant_dose_ladder": {}}
        proc = self._run(report)
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertNotIn("unknown (no", proc.stdout)
        self.assertIn("ab_merge_import_failed(", proc.stdout)

    def test_broken_ab_merge_named_cause_is_stable_regardless_of_report_shape(self):
        # The import failure fires before `report` is even inspected --
        # same named cause whether the report is well-formed, malformed, or
        # would otherwise have hit the four-cause fallback text.
        for report in (
            {"status": "GREEN", "mutant_dose_ladder": {"doses": None}},
            {"status": "GREEN"},
            {},
        ):
            with self.subTest(report=report):
                proc = self._run(report)
                self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
                self.assertTrue(proc.stdout.startswith("ab_merge_import_failed("), msg=proc.stdout)


if __name__ == "__main__":
    unittest.main()
