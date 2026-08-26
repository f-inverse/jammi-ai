#!/usr/bin/env python3
"""Tests for `check_ci_guard_wiring.py` (advisory iii, PR #372 round 2).

Every test here builds a THROWAWAY git repo under a `tempfile.TemporaryDirectory`
(`git init` + `git add -A`, no commit needed — `git ls-files` reads the index,
not history) and monkeypatches `check_ci_guard_wiring`'s module-level
`REPO_ROOT`/`SCRIPTS_DIR`/`WORKFLOWS_DIR`/`ALLOWLIST_PATH` constants to point
at it, so these tests are independent of THIS repo's own current gate/suite
inventory (which will keep changing) and drive the real `main()` entry point
against a fixture built to isolate exactly one predicate per test — never
`workflow_run_text()`/`tracked_test_suites()` asserted on in isolation with a
hand-built list standing in for the real filesystem/git state.

Run directly: `python3 ci/scripts/test_check_ci_guard_wiring.py`
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_ci_guard_wiring as cgw  # noqa: E402


class GuardWiringFixture(unittest.TestCase):
    """Base class: builds an isolated throwaway repo per test and patches
    the module's path constants onto it. `_run_main` drives the REAL `main()`
    entry point (implementer-acceptance clause 8), never `workflow_run_text()`
    or `tracked_test_suites()` called directly and asserted on in isolation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True
        )
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.root, check=True)

        self._orig = {
            "REPO_ROOT": cgw.REPO_ROOT,
            "SCRIPTS_DIR": cgw.SCRIPTS_DIR,
            "WORKFLOWS_DIR": cgw.WORKFLOWS_DIR,
            "ALLOWLIST_PATH": cgw.ALLOWLIST_PATH,
        }
        cgw.REPO_ROOT = self.root
        cgw.SCRIPTS_DIR = self.root / "ci" / "scripts"
        cgw.WORKFLOWS_DIR = self.root / ".github" / "workflows"
        cgw.ALLOWLIST_PATH = cgw.SCRIPTS_DIR / "ci_guard_wiring_allowlist.txt"
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._orig.items():
            setattr(cgw, k, v)

    def _write(self, rel: str, content: str = "pass\n"):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _git_add_all(self):
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)

    def _run_main(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cgw.main()
        return code, out.getvalue(), err.getvalue()


class CommentVsCodeLineTests(GuardWiringFixture):
    """THE F7 REGRESSION this round's own module doc names: a script named
    ONLY in a `#` comment must not count as wired.
    """

    def test_script_named_only_in_a_comment_is_unwired(self):
        self._write("ci/scripts/check_foo.py", "print('gate')\n")
        self._write(
            ".github/workflows/fake.yml",
            "name: fake\n# this workflow does not actually run check_foo.py, only mentions it\njobs: {}\n",
        )
        self._git_add_all()

        code, out, err = self._run_main()

        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("check_foo.py", err)

    def test_script_named_in_a_run_line_is_wired(self):
        self._write("ci/scripts/check_foo.py", "print('gate')\n")
        self._write(
            ".github/workflows/fake.yml",
            "name: fake\njobs:\n  guard:\n    steps:\n      - run: python3 ci/scripts/check_foo.py\n",
        )
        self._git_add_all()

        code, out, err = self._run_main()

        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")
        self.assertIn("check_foo.py]: OK", out)

    def test_script_named_only_in_a_matrix_cmd_field_is_wired(self):
        """This repo's OWN indirection convention (the `Guard` job in
        `ci.yml`): a matrix `cmd:` field consumed by a single shared
        `run: ${{ matrix.cmd }}` step — the script's name never appears on a
        literal `run:` line, only on the `cmd:` line. Must still count as
        wired (a strict "only run: lines" rule would false-redden this
        repo's entire existing guard matrix).
        """
        self._write("ci/scripts/check_foo.py", "print('gate')\n")
        self._write(
            ".github/workflows/fake.yml",
            "name: fake\njobs:\n  guard:\n    strategy:\n      matrix:\n        include:\n"
            "          - name: foo\n            cmd: python3 ci/scripts/check_foo.py\n"
            "    steps:\n      - run: ${{ matrix.cmd }}\n",
        )
        self._git_add_all()

        code, out, err = self._run_main()

        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")
        self.assertIn("check_foo.py]: OK", out)


class RootGeneralizationTests(GuardWiringFixture):
    """Advisory (iii): the three roots are PREFIX patterns, not one
    hand-picked crate name. Every positive case below is wired via a
    matching `run:` line; the negative case (a `tests/` directory that is
    NOT one of the three named roots) must be invisible to the scan even
    though it is a plausible-looking near miss.
    """

    def _workflow_wiring_all(self, *names: str) -> str:
        lines = ["name: fake\njobs:\n  guard:\n    steps:\n"]
        for n in names:
            lines.append(f"      - run: python3 {n}\n")
        return "".join(lines)

    def test_nested_ci_scripts_test_suite_is_found(self):
        self._write("ci/scripts/perf/test_bar.py")
        self._write(
            ".github/workflows/fake.yml", self._workflow_wiring_all("ci/scripts/perf/test_bar.py")
        )
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertIn("test_bar.py", suites)
        code, out, err = self._run_main()
        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")

    def test_any_crate_reference_dir_is_found_not_just_jammi_bench(self):
        """The generalization's whole point: `crates/widget/reference/` (an
        arbitrary crate name, never mentioned in this checker's source) must
        be found the same way `crates/jammi-bench/reference/` is — proving
        the root is a PATTERN, not a hard-coded path.
        """
        self._write("crates/widget/reference/test_baz.py")
        self._write(
            ".github/workflows/fake.yml",
            self._workflow_wiring_all("crates/widget/reference/test_baz.py"),
        )
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertIn("test_baz.py", suites)

    def test_top_level_tests_dir_is_found(self):
        self._write("tests/test_qux.py")
        self._write(".github/workflows/fake.yml", self._workflow_wiring_all("tests/test_qux.py"))
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertIn("test_qux.py", suites)

    def test_crate_tests_dir_not_reference_is_not_a_matching_root(self):
        """NEGATIVE CONTROL: `crates/widget/tests/` (note: `tests`, not
        `reference`) is deliberately OUT of scope (see this module's own
        doc / `check_ci_guard_wiring.py`'s `_CRATES_REFERENCE_RE` comment —
        those suites are wired by a `pytest`-over-a-directory convention
        this checker does not model). Must NOT be picked up, proving the
        generalization did not silently become "any `test_*.py` anywhere".
        """
        self._write("crates/widget/tests/test_notcovered.py")
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertNotIn("test_notcovered.py", suites)

    def test_non_test_prefixed_python_file_is_not_a_suite(self):
        """A `.py` file under a matching root that does NOT start with
        `test_` (e.g. a helper module) must not be swept in.
        """
        self._write("ci/scripts/perf/helper_not_a_suite.py")
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertNotIn("helper_not_a_suite.py", suites)


class ShSuiteWideningTests(GuardWiringFixture):
    """Round-2 audit on PR #387: `tracked_test_suites()` widened from
    `.py`-only to also match `test_*.sh` (`ci/scripts/test_gpu_dev_lifecycle.sh`
    landed as a hermetic bash regression suite and was structurally invisible
    to the original `.py`-only filter — the same F6/F7 blind-spot shape, one
    extension over). Pinned here so a future PR cannot silently narrow the
    filter back to `.py`-only without a test noticing, and so a non-`test_`-
    prefixed `.sh` file (a helper, not a suite) stays correctly excluded.
    """

    def test_tracked_test_sh_under_a_scanned_root_is_a_suite(self):
        self._write("ci/scripts/test_widget.sh", "#!/usr/bin/env bash\ntrue\n")
        self._write(
            ".github/workflows/fake.yml",
            "name: fake\njobs:\n  guard:\n    steps:\n      - run: bash ci/scripts/test_widget.sh\n",
        )
        self._git_add_all()

        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertIn("test_widget.sh", suites)

        code, out, err = self._run_main()
        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")
        self.assertIn("test_widget.sh]: OK", out)

    def test_non_test_prefixed_sh_file_is_not_a_suite(self):
        """A `.sh` file under a matching root that does NOT start with
        `test_` (e.g. a helper script sourced by a suite) must not be swept
        in — mirrors `test_non_test_prefixed_python_file_is_not_a_suite`
        for the `.sh` extension.
        """
        self._write("ci/scripts/helper_not_a_suite.sh", "#!/usr/bin/env bash\ntrue\n")
        self._git_add_all()
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertNotIn("helper_not_a_suite.sh", suites)


class GateScriptsRecursionTests(GuardWiringFixture):
    """F7 (this round's own self-inflicted regression, see
    `gate_scripts`'s own doc): the ORIGINAL `gate_scripts()` was top-level-
    only and would have been structurally blind to
    `ci/scripts/perf/check_citations.py` (advisory i, added in this SAME
    round). `gate_scripts()` is now tracked-and-recursive, matching
    `tracked_test_suites()`'s own shape — pinned here so a future PR cannot
    silently narrow it back to top-level-only without a test noticing.
    """

    def test_nested_check_script_is_found(self):
        self._write("ci/scripts/perf/check_widget.py")
        self._git_add_all()
        found = {p.name for p in cgw.gate_scripts()}
        self.assertIn("check_widget.py", found)

    def test_top_level_check_script_is_still_found(self):
        self._write("ci/scripts/check_widget.py")
        self._git_add_all()
        found = {p.name for p in cgw.gate_scripts()}
        self.assertIn("check_widget.py", found)

    def test_nested_check_sh_script_is_found(self):
        self._write("ci/scripts/perf/check_widget.sh")
        self._git_add_all()
        found = {p.name for p in cgw.gate_scripts()}
        self.assertIn("check_widget.sh", found)

    def test_non_check_prefixed_script_under_ci_is_not_a_gate(self):
        self._write("ci/scripts/perf/helper_widget.py")
        self._git_add_all()
        found = {p.name for p in cgw.gate_scripts()}
        self.assertNotIn("helper_widget.py", found)

    def test_check_script_outside_ci_is_not_a_gate(self):
        """`gate_scripts()` is rooted at `ci/`, same as `tracked_test_suites()`
        -- a `check_*.py` living somewhere else entirely (not this repo's
        gate-script convention at all) must not be swept in.
        """
        self._write("crates/widget/check_widget.py")
        self._git_add_all()
        found = {p.name for p in cgw.gate_scripts()}
        self.assertNotIn("check_widget.py", found)


class AllowlistTests(GuardWiringFixture):
    def test_unallowlisted_unwired_script_fails(self):
        self._write("ci/scripts/check_orphan.py")
        self._git_add_all()
        code, out, err = self._run_main()
        self.assertEqual(code, 1)
        self.assertIn("check_orphan.py", err)

    def test_allowlisted_unwired_script_passes(self):
        self._write("ci/scripts/check_orphan.py")
        self._write(
            "ci/scripts/ci_guard_wiring_allowlist.txt",
            "# deliberately unwired for this test\ncheck_orphan.py\n",
        )
        self._git_add_all()
        code, out, err = self._run_main()
        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")
        self.assertIn("check_orphan.py]: ALLOWLISTED", out)

    def test_allowlist_comment_and_blank_lines_are_ignored(self):
        self._write("ci/scripts/check_orphan.py")
        self._write(
            "ci/scripts/ci_guard_wiring_allowlist.txt",
            "\n# a comment\n\ncheck_orphan.py\n# trailing comment\n",
        )
        self._git_add_all()
        code, _out, _err = self._run_main()
        self.assertEqual(code, 0)


class NoGitCheckoutSurfacesLoudly(GuardWiringFixture):
    def test_untracked_file_is_invisible_to_the_scan(self):
        """A `test_*.py` under a matching root that was NEVER `git add`-ed
        must not be picked up — `tracked_test_suites()` reads `git ls-files`
        (what CI's own checkout contains), not a raw filesystem walk.
        """
        self._write("ci/scripts/test_untracked.py")
        # deliberately no self._git_add_all() here
        suites = {p.name for p in cgw.tracked_test_suites()}
        self.assertNotIn("test_untracked.py", suites)


if __name__ == "__main__":
    unittest.main()
