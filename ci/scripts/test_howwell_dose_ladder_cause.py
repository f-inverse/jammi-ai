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


def _call_main_capturing_stdout(argv: list[str]) -> tuple[int, str]:
    """Drives the real `namer.main(argv)` in-process, capturing its own
    stdout -- used by `ReportReadHardeningTests` so the file-read hardening
    is exercised through the REAL entry point (not a re-implementation of
    its open/parse logic here)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = namer.main(argv)
    return rc, buf.getvalue().strip()


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

    The catch-all `*)` arm is excluded by the arm-pattern regex's own
    character class (`[A-Za-z0-9_|]+`, which does not include `*`) failing
    to match that line at all -- `arm` is `None` for it and the `continue`
    above already skips it; there is deliberately no SEPARATE `pattern ==
    "*"` check below, since the regex itself already excludes it structurally
    (a bare `*` line can never reach the point where `pattern` is bound to
    the literal string `"*"` in the first place).
    """
    block = re.search(r'case\s+"\$STATUS"\s+in\n(.*?\n)\s*esac\b', script_text, re.DOTALL)
    if block is None:
        raise AssertionError('no `case "$STATUS" in ... esac` block found in runpod_gpu_howwell.sh')
    groups = []
    for line in block.group(1).splitlines():
        arm = re.match(r"^\s*([A-Za-z0-9_|]+)\)", line)
        if arm is None:
            continue
        groups.append(frozenset(arm.group(1).split("|")))
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
    `ab_merge.FINETUNE_RUN_STATUSES` here. In particular, this suite does NOT
    cover the OTHER drift direction (a `build_finetune_run_report` fold
    branch that assigns a status never added to `FINETUNE_RUN_STATUSES` in
    the first place) -- that is `ab_merge.py`'s own producer-side runtime
    guard's job (the `status not in FINETUNE_RUN_STATUSES` check immediately
    after the fold, unit-63 round-16 audit), a SEPARATE mechanism from this
    shell-arm binding, exercised by `test_ab_merge.py`'s own
    `FinetuneRunStatusRuntimeGuardTests`, not by anything here.
    """

    def test_shell_case_arms_exactly_cover_finetune_run_statuses(self):
        groups = _extract_status_case_arm_groups(_HOWWELL_SH.read_text())
        shell_names = frozenset().union(*groups) if groups else frozenset()
        self.assertEqual(shell_names, frozenset(ab_merge.FINETUNE_RUN_STATUSES))

    def test_shell_arm_grouping_matches_the_gating_green_record_only_partition(self):
        # unit-63 round-16 audit advisory 2: this asserts the LITERAL
        # `|`-joined arm grouping (e.g. `DRY_RUN|INCOMPLETE)` as ONE arm),
        # never merely "these statuses end up handled the same way" in some
        # looser, body-comparing sense. Deliberate, not an accidental gap: a
        # shell rewrite that split `DRY_RUN|INCOMPLETE)` into two SEPARATE
        # arms (`DRY_RUN) : ;;` and `INCOMPLETE) : ;;`) with byte-identical
        # bodies is semantically a no-op today, but this test would (and
        # should) still fail it -- verifying "same arm" is a simple,
        # mechanical frozenset-membership check on the arm PATTERN alone;
        # verifying "same arm OR two arms with textually-identical bodies"
        # would require this helper to also parse and diff arm BODIES, a
        # meaningfully more complex parser for a rewrite this codebase has
        # never made and has no reason to prefer over just re-joining the
        # pattern. The minimal honest choice is to keep the simple check and
        # state the limitation here, not to grow the parser speculatively:
        # if a real future edit legitimately wants two separately-bodied
        # arms per status, that is a real code-shape change this test is
        # right to force a look at, not a false positive to silently absorb.
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


class ReportShapeHardeningTests(unittest.TestCase):
    """Unit-63 round-17 audit shapes (a)/(b): a `json.loads`-parsed
    `report` that is valid JSON but not an object (`null`, `[]`, `"str"`,
    `3`), or an object whose `mutant_dose_ladder` value is present but not
    itself an object (e.g. a list), used to crash `dose_ladder_cause` with
    an uncaught `AttributeError` from calling `.get` on a non-dict -- which
    `runpod_gpu_howwell.sh`'s own `2>/dev/null || echo "unknown (could not
    inspect ...)"` wrapper collapsed into the same opaque "unknown" text a
    genuinely-no-cause-found run also produces (rc 1, empty stdout). Both
    now degrade to a NAMED cause, exit 0, driven here through the real CLI
    subprocess (the exact shape `runpod_gpu_howwell.sh` invokes), pinned
    RED at 668a3206 (each shape below crashed with the errors named in its
    own comment before this suite's own fix).

    Pre-fix RED, captured directly (each run via
    `python3 ci/scripts/howwell_dose_ladder_cause.py <path>` at 668a3206):
      null       -> AttributeError: 'NoneType' object has no attribute 'get' (rc=1)
      []         -> AttributeError: 'list' object has no attribute 'get' (rc=1)
      "str"      -> AttributeError: 'str' object has no attribute 'get' (rc=1)
      3          -> AttributeError: 'int' object has no attribute 'get' (rc=1)
      {"mutant_dose_ladder": [1, 2]} -> AttributeError: 'list' object has no attribute 'get' (rc=1)
    """

    def _run(self, report_text: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            fh.write(report_text)
            path = fh.name
        try:
            return subprocess.run([sys.executable, _SCRIPT, path], capture_output=True, text=True)
        finally:
            Path(path).unlink()

    def test_top_level_null_degrades_to_a_named_cause(self):
        proc = self._run("null")
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_is_not_an_object(type=NoneType)", proc.stdout)

    def test_top_level_empty_list_degrades_to_a_named_cause(self):
        proc = self._run("[]")
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_is_not_an_object(type=list)", proc.stdout)

    def test_top_level_string_degrades_to_a_named_cause(self):
        proc = self._run('"str"')
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_is_not_an_object(type=str)", proc.stdout)

    def test_top_level_number_degrades_to_a_named_cause(self):
        proc = self._run("3")
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_is_not_an_object(type=int)", proc.stdout)

    def test_non_dict_mutant_dose_ladder_degrades_to_a_named_cause(self):
        proc = self._run('{"mutant_dose_ladder": [1, 2]}')
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("mutant_dose_ladder_is_not_an_object(type=list)", proc.stdout)

    def test_falsy_non_dict_mutant_dose_ladder_still_degrades_to_the_empty_ladder_case(self):
        # `mutant_dose_ladder` falsy-but-non-dict (e.g. an empty list) takes
        # the SAME "treat as empty ladder" path a `null`/absent value takes
        # -- this is pre-existing behavior (`or {}`, now `if not ladder`)
        # this fix does not change, only the TRUTHY-non-dict case (above)
        # is newly guarded.
        proc = self._run('{"status": "GREEN", "mutant_dose_ladder": []}')
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("unknown", proc.stdout)
        self.assertNotIn("mutant_dose_ladder_is_not_an_object", proc.stdout)

    def test_dose_ladder_cause_direct_call_also_names_non_dict_report(self):
        # Same shape, driven directly through `dose_ladder_cause` (not just
        # through `main()`'s subprocess CLI) -- proves the guard lives in
        # the function itself, not merely something `main()` papers over.
        self.assertEqual(namer.dose_ladder_cause(None), "report_is_not_an_object(type=NoneType)")
        self.assertEqual(namer.dose_ladder_cause([1, 2]), "report_is_not_an_object(type=list)")
        self.assertEqual(
            namer.dose_ladder_cause({"mutant_dose_ladder": "not-a-dict"}),
            "mutant_dose_ladder_is_not_an_object(type=str)",
        )


class ReportUndecodableHardeningTests(unittest.TestCase):
    """Unit-63 round-17 audit shape (c): a report file that is not valid
    UTF-8 raised `UnicodeDecodeError` from INSIDE `main()`'s own `fh.read()`
    -- a `ValueError` subclass, not an `OSError` subclass, so the pre-fix
    `except OSError` arm alone did not catch it; it propagated uncaught
    (rc 1, empty stdout, a traceback on stderr). Pinned RED at 668a3206:
    `python3 ci/scripts/howwell_dose_ladder_cause.py <non-utf8-file>` raised
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0:
    invalid start byte` uncaught. Now degrades to a NAMED
    `report_undecodable(...)` cause, exit 0.
    """

    def test_non_utf8_file_degrades_to_a_named_cause_rc_zero(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            fh.write(b"\xff\xfe\x00{not valid")
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, _SCRIPT, path], capture_output=True, text=True)
        finally:
            Path(path).unlink()
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_undecodable(", proc.stdout)
        self.assertIn("UnicodeDecodeError", proc.stdout)

    def test_non_utf8_file_through_main_direct_call_matches_subprocess(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            fh.write(b"\xff\xfe\x00{not valid")
            path = fh.name
        try:
            rc, stdout = _call_main_capturing_stdout([path])
        finally:
            Path(path).unlink()
        self.assertEqual(rc, 0)
        self.assertIn("report_undecodable(", stdout)


class ShadowedAbMergeAttributesHardeningTests(unittest.TestCase):
    """Unit-63 round-17 audit shapes (d)/(e): a module named `ab_merge`
    that IMPORTS cleanly (no `ImportError`/`SyntaxError`) but is stale or
    shadowed -- lacking one of the three attribute names this module reads
    off it at module load (`DOSE_LADDER_EXIT_CAUSE_NAMES`,
    `MUTANT_DOSE_DETECTED_INVALID`, `RED_PROOF_VERDICT_NOT_PROVEN_PREFIX`)
    -- used to raise an uncaught `AttributeError` straight out of module
    load (rc 1, empty stdout, a traceback on stderr), reachable even though
    `AbMergeImportFailureHardeningTests` (an outright `import ab_merge`
    failure) was already hardened. Pinned RED at 668a3206: a stub `ab_merge`
    module defining only an unrelated name crashed with `AttributeError:
    module 'ab_merge' has no attribute 'DOSE_LADDER_EXIT_CAUSE_NAMES'` at
    module load, and a stub defining `DOSE_LADDER_EXIT_CAUSE_NAMES` alone
    (but not the other two) crashed with `AttributeError: module 'ab_merge'
    has no attribute 'MUTANT_DOSE_DETECTED_INVALID'`. Both now degrade to
    the SAME `ab_merge_import_failed(...)` named cause an outright import
    failure produces -- same setup shape as
    `AbMergeImportFailureHardeningTests` (copy the real script into a
    throwaway directory alongside a deliberately-stale `perf/ab_merge.py`
    stub, invoke as a real subprocess), with a stale-but-importable stub
    swapped in for a broken one.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="howwell-dose-ladder-cause-shadowed-attrs-")
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
        shutil.copy(_SCRIPT, Path(self._tmpdir) / "howwell_dose_ladder_cause.py")
        (Path(self._tmpdir) / "perf").mkdir()
        self._script = str(Path(self._tmpdir) / "howwell_dose_ladder_cause.py")

    def _write_stub(self, stub_source: str) -> None:
        (Path(self._tmpdir) / "perf" / "ab_merge.py").write_text(stub_source)

    def _run(self, report: dict) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=self._tmpdir) as fh:
            json.dump(report, fh)
            path = fh.name
        return subprocess.run([sys.executable, self._script, path], capture_output=True, text=True)

    def test_stub_missing_all_three_attributes_degrades_to_a_named_cause(self):
        self._write_stub("SOME_OTHER_CONST = 1\n")
        proc = self._run({"status": "GREEN", "mutant_dose_ladder": {}})
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("ab_merge_import_failed(", proc.stdout)
        self.assertIn("AttributeError", proc.stdout)
        self.assertIn("DOSE_LADDER_EXIT_CAUSE_NAMES", proc.stdout)

    def test_stub_missing_only_the_second_and_third_attribute_degrades_to_a_named_cause(self):
        self._write_stub('DOSE_LADDER_EXIT_CAUSE_NAMES = ("a", "b", "c", "d")\n')
        proc = self._run({"status": "GREEN", "mutant_dose_ladder": {}})
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("ab_merge_import_failed(", proc.stdout)
        self.assertIn("MUTANT_DOSE_DETECTED_INVALID", proc.stdout)

    def test_stub_missing_attributes_never_produces_bare_unknown(self):
        self._write_stub("SOME_OTHER_CONST = 1\n")
        proc = self._run({"status": "GREEN", "mutant_dose_ladder": {}})
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertNotIn("unknown (no", proc.stdout)


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

    def test_missing_file_subprocess_degrades_to_a_named_cause_rc_zero(self):
        # unit-63 round-16 audit advisory 3: a missing REPORT_JSON_PATH used
        # to crash with an uncaught FileNotFoundError (non-zero exit, empty
        # stdout) -- exactly the opaque-collapse shape this repo's own
        # "unknown (could not inspect ...)" wrapper text warns about one
        # layer up. It now degrades to a NAMED cause on stdout, exit 0, same
        # discipline as `_inspect_doses`/`_AB_MERGE_IMPORT_ERROR`.
        proc = subprocess.run(
            [sys.executable, _SCRIPT, "/nonexistent/path/does-not-exist.json"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr={proc.stderr!r}")
        self.assertIn("report_unreadable(", proc.stdout)
        self.assertIn("does-not-exist.json", proc.stdout)

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


class ReportReadHardeningTests(unittest.TestCase):
    """Unit-63 round-16 audit advisory 3: `main()`'s own file open + JSON
    parse used to sit entirely OUTSIDE the named-degradation discipline
    `dose_ladder_cause`/`_inspect_doses`/`_AB_MERGE_IMPORT_ERROR` provide --
    a missing/unreadable/malformed REPORT_JSON_PATH crashed straight into
    `runpod_gpu_howwell.sh`'s own opaque wrapper, reachable regardless of
    whether `ab_merge` itself imported cleanly (an independent failure
    axis from `AbMergeImportFailureHardeningTests`, below). Both a missing
    file and malformed JSON now degrade to a NAMED cause on stdout, exit 0,
    driven through the REAL `main()` entry point directly (not merely
    `dose_ladder_cause`, which never sees the raw file at all).
    """

    def test_missing_file_through_main_degrades_to_a_named_cause_rc_zero(self):
        rc, stdout = _call_main_capturing_stdout(["/nonexistent/path/does-not-exist.json"])
        self.assertEqual(rc, 0)
        self.assertIn("report_unreadable(", stdout)
        self.assertIn("does-not-exist.json", stdout)

    def test_malformed_json_through_main_degrades_to_a_named_cause_rc_zero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            fh.write("{not valid json at all")
            path = fh.name
        try:
            rc, stdout = _call_main_capturing_stdout([path])
        finally:
            Path(path).unlink()
        self.assertEqual(rc, 0)
        self.assertIn("report_malformed_json(", stdout)

    def test_a_directory_path_is_unreadable_not_a_crash(self):
        # `open()` on a directory raises `IsADirectoryError` (an `OSError`
        # subclass) -- the SAME `report_unreadable(...)` path a missing file
        # takes, never a distinct uncaught exception shape.
        with tempfile.TemporaryDirectory() as dir_path:
            rc, stdout = _call_main_capturing_stdout([dir_path])
        self.assertEqual(rc, 0)
        self.assertIn("report_unreadable(", stdout)

    def test_missing_file_subprocess_matches_direct_call(self):
        # Cross-checks the subprocess-level assertion in
        # `MainEntryPointTests.test_missing_file_subprocess_degrades_to_a_named_cause_rc_zero`
        # against a direct in-process `main()` call -- proves the subprocess
        # shape is not accidentally exercising a different code path.
        rc, stdout = _call_main_capturing_stdout(["/nonexistent/path/does-not-exist.json"])
        proc = subprocess.run(
            [sys.executable, _SCRIPT, "/nonexistent/path/does-not-exist.json"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(rc, proc.returncode)
        self.assertEqual(stdout, proc.stdout.strip())


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
