#!/usr/bin/env python3
"""`run_guards.py`'s suite, and `checks.py`'s through it: the parser's
refusals, path globs and selection, need provisioning, and the committed
guard list itself."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks  # noqa: E402
import run_guards as rg  # noqa: E402

NEEDS = checks.parse_needs("""
[need.tool]
probe = "true"
provide = "true"

[need.rig-host]
probe = "true"
provide = "false"
""")

LIST = """
[scope.rig]
paths = ["ci/**", "**/Cargo.toml"]

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
name = "third"
run = "true"
why = "w"
when = ["rig"]
needs = ["rig-host"]
"""


def names(guards) -> list[str]:
    return [g.name for g in guards]


class Parse(unittest.TestCase):
    def refused(self, text: str, fragment: str) -> None:
        with self.assertRaisesRegex(checks.CheckListError, fragment):
            rg.parse_guard_list(text, NEEDS)

    def test_reads_a_well_formed_list(self):
        parsed = rg.parse_guard_list(LIST, NEEDS)
        self.assertEqual(names(parsed.guards), ["always", "scoped", "third"])
        self.assertEqual(parsed.guards[1].when, ("rig",))
        self.assertEqual(parsed.needs["tool"].provide, "true")

    def test_refuses_an_unknown_field(self):
        self.refused(LIST + '\n[[guard]]\nname="x"\nrun="true"\nwhy="w"\nadvisory=true\n', "unknown.*advisory")

    def test_refuses_a_guard_without_a_why(self):
        self.refused(LIST + '\n[[guard]]\nname="x"\nrun="true"\n', "missing.*why")

    def test_refuses_a_duplicate_name(self):
        self.refused(LIST + '\n[[guard]]\nname="always"\nrun="true"\nwhy="w"\n', "duplicate.*always")

    def test_refuses_an_unknown_scope_and_need(self):
        self.refused(LIST.replace('when = ["rig"]', 'when = ["rigg"]'), "unknown scope")
        self.refused(LIST.replace('needs = ["tool"]', 'needs = ["tol"]'), "unknown need")

    def test_refuses_an_unused_scope(self):
        self.refused(LIST.replace('when = ["rig"]\n', ""), "scope 'rig' is used by no guard")

    def test_refuses_a_glob_it_cannot_honour(self):
        self.refused(LIST.replace('"ci/**"', '"!ci/**"'), "unsupported path glob")


class Globs(unittest.TestCase):
    def matches(self, pattern: str, path: str) -> bool:
        return checks.glob_to_regex(pattern).fullmatch(path) is not None

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
    parsed = rg.parse_guard_list(LIST, NEEDS)

    def selected(self, changed) -> list[str]:
        return names(rg.select(self.parsed, None if changed is None else frozenset(changed)))

    def test_an_unknown_change_runs_everything(self):
        self.assertEqual(self.selected(None), ["always", "scoped", "third"])

    def test_a_scoped_guard_runs_only_for_its_scope(self):
        self.assertEqual(self.selected(["crates/jammi-db/src/lib.rs"]), ["always"])
        self.assertEqual(self.selected(["crates/jammi-db/Cargo.toml"]), ["always", "scoped", "third"])
        self.assertEqual(self.selected([]), ["always"])

    def test_a_change_to_the_selection_itself_runs_everything(self):
        for path in rg.SELF_PATHS:
            narrowed = rg.parse_guard_list(LIST.replace('"ci/**", ', ""), NEEDS)
            self.assertEqual(names(rg.select(narrowed, frozenset([path]))), ["always", "scoped", "third"])

    def test_needs_are_those_of_the_selection_once_each(self):
        self.assertEqual([n.name for n in checks.needs_of(tuple(g.needs for g in self.parsed.guards), self.parsed.needs)], ["tool", "rig-host"])
        self.assertEqual(checks.needs_of(tuple(g.needs for g in self.parsed.guards[:1]), self.parsed.needs), ())


class Execute(unittest.TestCase):
    def test_a_present_need_is_not_provided_again(self):
        self.assertIsNone(checks.provide(checks.Need("n", probe="true", provide="exit 9")))

    def test_an_absent_need_is_provided_then_probed_again(self):
        with self.subTest("provide succeeds but the need is still absent"):
            self.assertIn("'n' is absent", checks.provide(checks.Need("n", probe="false", provide="true")))
        with self.subTest("provide fails"):
            why = checks.provide(checks.Need("n", probe="echo what-is-missing; false", provide="echo no-yum-here; exit 1"))
            for expected in ("what-is-missing", "no-yum-here"):
                self.assertIn(expected, why)

    def test_a_check_passes_or_fails_by_its_exit_status(self):
        self.assertTrue(checks.run("g", "true").passed)
        failed = checks.run("g", "echo the-evidence; false")
        self.assertFalse(failed.passed)
        for expected in ("FAIL  g", "the property", "the-evidence"):
            self.assertIn(expected, checks.report(failed, "  why: the property\n"))

    def test_a_failure_mid_pipeline_fails_the_check(self):
        self.assertFalse(checks.run("g", "false | cat").passed)

    def test_a_check_reading_stdin_sees_end_of_file(self):
        self.assertTrue(checks.run("g", "test -z \"$(cat)\"").passed)


class CommittedList(unittest.TestCase):
    parsed = rg.parse_guard_list((rg.REPO_ROOT / rg.GUARDS_FILE).read_text(), checks.load_needs())

    def test_every_guard_runs_a_tracked_file(self):
        for guard in self.parsed.guards:
            scripts = re.findall(r"(?:^|\s)((?:ci|tests|crates|cookbook)/\S+\.(?:py|sh))", guard.run)
            self.assertTrue(scripts, f"{guard.name}: `run` names no script")
            for script in scripts:
                self.assertTrue((rg.REPO_ROOT / script).is_file(), f"{guard.name}: {script} does not exist")

    def test_the_paths_that_rerun_everything_exist(self):
        for path in rg.SELF_PATHS:
            self.assertTrue((rg.REPO_ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
