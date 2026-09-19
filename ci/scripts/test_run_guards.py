#!/usr/bin/env python3
"""`run_guards.py`'s suite: the parser's refusals, path selection, need
provisioning, lanes, and the committed guard list itself."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_guards as rg  # noqa: E402

LIST = """
[scope.rig]
paths = ["ci/**", "**/Cargo.toml"]

[need.tool]
probe = "true"
provide = "true"

[need.rig-host]
probe = "true"
provide = "false"

[lane.elsewhere]
host = "a host the CI image is not"

[[guard]]
name = "always"
run = "true"
why = "w"

[[guard]]
name = "scoped"
run = "true"
why = "w"
when = ["rig"]
needs = ["tool"]

[[guard]]
name = "elsewhere always"
run = "true"
why = "w"
lane = "elsewhere"
needs = ["rig-host"]

[[guard]]
name = "elsewhere scoped"
run = "true"
why = "w"
lane = "elsewhere"
when = ["rig"]
needs = ["rig-host"]
"""


def names(guards) -> list[str]:
    return [g.name for g in guards]


class Parse(unittest.TestCase):
    def refused(self, text: str, fragment: str) -> None:
        with self.assertRaisesRegex(rg.GuardListError, fragment):
            rg.parse_guard_list(text)

    def test_reads_a_well_formed_list(self):
        parsed = rg.parse_guard_list(LIST)
        self.assertEqual(
            names(parsed.guards), ["always", "scoped", "elsewhere always", "elsewhere scoped"]
        )
        self.assertEqual(parsed.guards[1].when, ("rig",))
        self.assertEqual(parsed.needs["tool"].provide, "true")
        self.assertEqual([g.lane for g in parsed.guards], [None, None, "elsewhere", "elsewhere"])
        self.assertEqual(parsed.lanes, {"elsewhere": "a host the CI image is not"})

    def test_refuses_an_unknown_field(self):
        self.refused(LIST + '\n[[guard]]\nname="x"\nrun="true"\nwhy="w"\nadvisory=true\n', "unknown.*advisory")

    def test_refuses_a_guard_without_a_why(self):
        self.refused(LIST + '\n[[guard]]\nname="x"\nrun="true"\n', "missing.*why")

    def test_refuses_a_duplicate_name(self):
        self.refused(LIST + '\n[[guard]]\nname="always"\nrun="true"\nwhy="w"\n', "duplicate.*always")

    def test_refuses_an_unknown_scope_and_need(self):
        self.refused(LIST.replace('when = ["rig"]', 'when = ["rigg"]'), "unknown scope")
        self.refused(LIST.replace('needs = ["tool"]', 'needs = ["tol"]'), "unknown need")

    def test_refuses_an_unused_scope_and_need(self):
        self.refused(LIST.replace('when = ["rig"]\n', ""), "scope 'rig' is used by no guard")
        self.refused(LIST.replace('needs = ["tool"]\n', ""), "need 'tool' is used by no guard")

    def test_refuses_an_unknown_and_an_unused_lane(self):
        self.refused(LIST.replace('lane = "elsewhere"\nneeds', 'lane = "elsewher"\nneeds', 1), "unknown lane 'elsewher'")
        self.refused(
            LIST + '\n[lane.nowhere]\nhost = "h"\n', "lane 'nowhere' is used by no guard"
        )

    def test_refuses_a_lane_guard_that_needs_nothing_of_its_host(self):
        self.refused(
            LIST + '\n[[guard]]\nname="x"\nrun="true"\nwhy="w"\nlane="elsewhere"\n',
            "lane 'elsewhere' but no needs",
        )

    def test_refuses_a_glob_it_cannot_honour(self):
        self.refused(LIST.replace('"ci/**"', '"!ci/**"'), "unsupported path glob")


class Globs(unittest.TestCase):
    def matches(self, pattern: str, path: str) -> bool:
        return rg.glob_to_regex(pattern).fullmatch(path) is not None

    def test_double_star_crosses_directories(self):
        self.assertTrue(self.matches("ci/**", "ci/scripts/perf/x.py"))
        self.assertFalse(self.matches("ci/**", "cia/x.py"))
        self.assertFalse(self.matches("ci/**", "docs/ci/x.py"))

    def test_leading_double_star_also_matches_the_root(self):
        self.assertTrue(self.matches("**/Cargo.toml", "Cargo.toml"))
        self.assertTrue(self.matches("**/Cargo.toml", "crates/jammi-db/Cargo.toml"))
        self.assertFalse(self.matches("**/Cargo.toml", "crates/jammi-db/Cargo.toml.bak"))

    def test_single_star_stays_inside_a_segment(self):
        self.assertTrue(self.matches("ci/*.toml", "ci/guards.toml"))
        self.assertFalse(self.matches("ci/*.toml", "ci/scripts/x.toml"))

    def test_a_dot_is_literal(self):
        self.assertFalse(self.matches("Cargo.lock", "Cargoxlock"))


class Select(unittest.TestCase):
    parsed = rg.parse_guard_list(LIST)

    def selected(self, changed, lane=None) -> list[str]:
        return names(rg.select(self.parsed, None if changed is None else frozenset(changed), lane))

    def test_an_unknown_change_runs_everything(self):
        self.assertEqual(self.selected(None), ["always", "scoped"])

    def test_a_scoped_guard_runs_only_for_its_scope(self):
        self.assertEqual(self.selected(["crates/jammi-db/src/lib.rs"]), ["always"])
        self.assertEqual(self.selected(["crates/jammi-db/Cargo.toml"]), ["always", "scoped"])
        self.assertEqual(self.selected([]), ["always"])

    def test_a_change_to_the_selection_itself_runs_everything(self):
        for path in rg.SELF_PATHS:
            narrowed = rg.parse_guard_list(LIST.replace('"ci/**", ', ""))
            self.assertEqual(names(rg.select(narrowed, frozenset([path]))), ["always", "scoped"])

    def test_a_lane_selects_its_own_guards_and_no_other_lane_does(self):
        self.assertEqual(self.selected(None, "elsewhere"), ["elsewhere always", "elsewhere scoped"])
        self.assertEqual(self.selected(["crates/jammi-db/src/lib.rs"], "elsewhere"), ["elsewhere always"])
        for changed in (None, list(rg.SELF_PATHS), ["crates/jammi-db/Cargo.toml"]):
            self.assertNotIn("elsewhere always", self.selected(changed))

    def test_the_needs_of_another_lane_are_never_provided(self):
        in_lane = rg.select(self.parsed, None)
        self.assertEqual([n.name for n in rg.needs_of(self.parsed, in_lane)], ["tool"])

    def test_needs_are_those_of_the_selection_once_each(self):
        twice = rg.parse_guard_list(LIST + '\n[[guard]]\nname="third"\nrun="true"\nwhy="w"\nneeds=["tool"]\n')
        self.assertEqual([n.name for n in rg.needs_of(twice, twice.guards)], ["tool", "rig-host"])
        self.assertEqual(rg.needs_of(twice, twice.guards[:1]), ())


class Execute(unittest.TestCase):
    def test_a_present_need_is_not_provided_again(self):
        self.assertIsNone(rg.provide(rg.Need("n", probe="true", provide="exit 9")))

    def test_an_absent_need_is_provided_then_probed_again(self):
        with self.subTest("provide succeeds but the need is still absent"):
            self.assertIn("'n' is absent", rg.provide(rg.Need("n", probe="false", provide="true")))
        with self.subTest("provide fails"):
            why = rg.provide(rg.Need("n", probe="echo what-is-missing; false", provide="echo no-yum-here; exit 1"))
            for expected in ("what-is-missing", "no-yum-here"):
                self.assertIn(expected, why)

    def test_a_guard_passes_or_fails_by_its_exit_status(self):
        guard = lambda run: rg.Guard("g", run, "the property", (), ())  # noqa: E731
        self.assertTrue(rg.run_guard(guard("true")).passed)
        failed = rg.run_guard(guard("echo the-evidence; false"))
        self.assertFalse(failed.passed)
        for expected in ("FAIL  g", "the property", "the-evidence"):
            self.assertIn(expected, rg.report(failed))

    def test_a_failure_mid_pipeline_fails_the_guard(self):
        self.assertFalse(rg.run_guard(rg.Guard("g", "false | cat", "w", (), ())).passed)

    def test_a_guard_reading_stdin_sees_end_of_file(self):
        self.assertTrue(rg.run_guard(rg.Guard("g", "test -z \"$(cat)\"", "w", (), ())).passed)


class CommittedList(unittest.TestCase):
    parsed = rg.parse_guard_list((rg.REPO_ROOT / rg.GUARDS_FILE).read_text())

    def test_every_guard_runs_a_tracked_file(self):
        for guard in self.parsed.guards:
            scripts = re.findall(r"(?:^|\s)((?:ci|tests|crates)/\S+\.(?:py|sh))", guard.run)
            self.assertTrue(scripts, f"{guard.name}: `run` names no script")
            for script in scripts:
                self.assertTrue((rg.REPO_ROOT / script).is_file(), f"{guard.name}: {script} does not exist")

    def test_an_undeclared_lane_is_refused(self):
        self.assertEqual(rg.main(["--list", "--lane", "no-such-lane"]), 2)

    def test_the_paths_that_rerun_everything_exist(self):
        for path in rg.SELF_PATHS:
            self.assertTrue((rg.REPO_ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
