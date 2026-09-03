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

# CI incident (run 33230050451, main, "Guard (arch validation freshness
# self-test)"): `shutil.rmtree` during a `tempfile.TemporaryDirectory`'s
# teardown can hit `OSError: [Errno 39] Directory not empty: '.git'` — a
# race between tempdir cleanup and a background `git maintenance`/
# `gc --auto` process the scratch repo below (`GuardWiringFixture.setUp`)
# can spawn. `-c gc.auto=0 -c gc.autoDetach=false -c maintenance.auto=false`
# kills the background writer AT THE SOURCE.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _scratch_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=cwd, check=True)


# A minimal, CORRECTLY wired stand-in for `.github/workflows/cookbook-book.yml`:
# selector first, `build-server` bound to that selector's output, render step
# putting `target/release` on PATH. Synthetic on purpose — a fixture that read
# the real workflow would stop proving anything the day the real workflow's
# shape changed, the same reason `select_render_chapters.py`'s own self-test
# never reads the real book. Each RED-proof below mutates exactly one of the
# three facts and asserts the checker reddens naming that one.
WELL_WIRED_BOOK_WORKFLOW = """name: book
jobs:
  book:
    steps:
      - name: Select the chapters this diff must render
        id: select
        run: |
          echo "needs_server=true" >> "$GITHUB_OUTPUT"
      - name: Build + install the HEAD wheels
        uses: ./.github/actions/setup-jammi-py
        with:
          mode: wheel
          build-server: ${{ steps.select.outputs.needs_server }}
      - name: Render this diff's selected chapters
        run: |
          export PATH="$GITHUB_WORKSPACE/target/release:$PATH"
          quarto render
"""


class GuardWiringFixture(unittest.TestCase):
    """Base class: builds an isolated throwaway repo per test and patches
    the module's path constants onto it. `_run_main` drives the REAL `main()`
    entry point (implementer-acceptance clause 8), never `workflow_run_text()`
    or `tracked_test_suites()` called directly and asserted on in isolation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        _scratch_git(["init", "-q"], self.root)
        _scratch_git(["config", "user.email", "test@example.com"], self.root)
        _scratch_git(["config", "user.name", "test"], self.root)

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

        # `main()` also asserts the Book gate's provisioning wiring, and it
        # treats a MISSING `cookbook-book.yml` as a violation rather than as
        # a skip (fail-closed: a gate that passes when its subject disappears
        # is the fail-open shape the checker exists to close). So every
        # fixture starts from a WELL-WIRED copy — the tests below that care
        # about that property mutate it, and the tests that do not are
        # isolated from it exactly as they are from this repo's real gate
        # inventory.
        self._write(".github/workflows/cookbook-book.yml", WELL_WIRED_BOOK_WORKFLOW)

    def _restore(self):
        for k, v in self._orig.items():
            setattr(cgw, k, v)

    def _write(self, rel: str, content: str = "pass\n"):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _git_add_all(self):
        _scratch_git(["add", "-A"], self.root)

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

    def test_script_wired_only_from_a_yaml_workflow_counts_as_wired(self):
        """BLOCK B7 audit fix (same class as `check_gpu_prove_once.py`'s own):
        GitHub Actions runs BOTH `.yml` and `.yaml` workflow files -- a script
        mentioned ONLY in a `.yaml` workflow (never a `.yml` one) must still
        read as wired, not structurally invisible to a `*.yml`-only glob.
        """
        self._write("ci/scripts/check_something.py")
        self._write(
            ".github/workflows/fake.yaml",
            self._workflow_wiring_all("ci/scripts/check_something.py"),
        )
        self._git_add_all()
        text = cgw.workflow_run_text()
        self.assertIn("ci/scripts/check_something.py", text)
        code, out, err = self._run_main()
        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")


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


class BookProvisioningWiringTests(GuardWiringFixture):
    """RED-proofs for the second property: the Book gate's step ORDER and its
    input BINDINGS, each broken one at a time on a temp copy of the workflow.

    Every case drives the real `main()` and asserts BOTH the exit code and
    that the finding names the fact that was broken — a checker that reddens
    for some other reason would pass an exit-code-only assertion while
    proving nothing about the pin under test.
    """

    def setUp(self):
        super().setUp()
        # `main()` refuses a repo with NO gate scripts at all before it ever
        # reaches the Book pins, so without this every case below would exit
        # 1 for the wrong reason — the RED assertions would pass on the
        # empty-inventory message and prove nothing. One wired gate script,
        # in a SEPARATE workflow file so `cookbook-book.yml` stays exactly the
        # shape under test (and so the deletion case below still has a wired
        # script to find).
        self._write("ci/scripts/check_present.py", "print('gate')\n")
        self._write(
            ".github/workflows/guard.yml",
            "name: guard\njobs:\n  guard:\n    steps:\n"
            "      - run: python3 ci/scripts/check_present.py\n",
        )
        self._git_add_all()

    def _mutate_book(self, old: str, new: str):
        path = self.root / ".github/workflows/cookbook-book.yml"
        text = path.read_text()
        assert text.count(old) == 1, f"fixture drift: {old!r} appears {text.count(old)}x"
        path.write_text(text.replace(old, new))
        self._git_add_all()

    def test_well_wired_workflow_passes(self):
        """The GREEN control. Without it, every RED below could be reddening
        on the fixture rather than on the mutation."""
        self._git_add_all()
        code, out, err = self._run_main()
        self.assertEqual(code, 0, f"stdout={out!r} stderr={err!r}")
        self.assertIn("cookbook-book.yml provisioning]: OK", out)

    def test_selector_after_the_wheel_build_is_red(self):
        """The order fact: a selector that runs after the build cannot tell
        the build whether to produce a server."""
        path = self.root / ".github/workflows/cookbook-book.yml"
        lines = path.read_text().splitlines(keepends=True)
        # Located by CONTENT, not by a hardcoded index: a fixture edit that
        # shifted the step would otherwise silently turn this into a test of
        # some other mutation.
        start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("- name: Select"))
        end = next(i for i, l in enumerate(lines) if l.lstrip().startswith("- name: Build"))
        select_step = lines[start:end]
        self.assertIn("        id: select\n", select_step)
        path.write_text("".join(lines[:start] + lines[end:] + select_step))
        self._git_add_all()

        code, out, err = self._run_main()

        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("runs AFTER setup-jammi-py", err)

    def test_build_server_hardcoded_false_is_red(self):
        """The binding fact, in its most dangerous direction: a literal
        `false` still LOOKS wired and silently drops every grpc:// chapter's
        proof."""
        self._mutate_book(
            "build-server: ${{ steps.select.outputs.needs_server }}",
            "build-server: false",
        )
        code, out, err = self._run_main()
        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("build-server:", err)
        self.assertIn("steps.select.outputs.needs_server", err)

    def test_build_server_bound_to_a_different_step_output_is_red(self):
        """A near miss the pin must still catch: bound to an output, but not
        to the SELECTOR's — the flag stops tracking the selected set."""
        self._mutate_book(
            "build-server: ${{ steps.select.outputs.needs_server }}",
            "build-server: ${{ steps.other.outputs.needs_server }}",
        )
        code, out, err = self._run_main()
        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("steps.select.outputs.needs_server", err)

    def test_missing_target_release_on_path_is_red(self):
        """The reachability fact: a `jammi-server` built into
        `target/release` and never put on PATH was not built."""
        self._mutate_book(
            'export PATH="$GITHUB_WORKSPACE/target/release:$PATH"',
            'export PATH="$GITHUB_WORKSPACE/target/debug:$PATH"',
        )
        code, out, err = self._run_main()
        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("target/release", err)
        self.assertIn("PATH", err)

    def test_wiring_described_only_in_a_comment_does_not_satisfy_the_pins(self):
        """THE fail-open this check exists to refuse, and the reason it is a
        comment-STRIPPED scan rather than a raw-text one: the real workflow
        already carries a comment saying `target/release` goes on PATH. If
        prose could satisfy a pin, this gate would assert that somebody once
        DESCRIBED the wiring — exactly the discipline-only state it replaces.
        """
        self._mutate_book(
            '          export PATH="$GITHUB_WORKSPACE/target/release:$PATH"\n',
            '          # export PATH="$GITHUB_WORKSPACE/target/release:$PATH" goes here\n',
        )
        code, out, err = self._run_main()
        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("target/release", err)

    def test_missing_workflow_file_is_red_not_skipped(self):
        """Fail-closed on absence. A gate that passes when its subject is
        deleted or renamed enforces nothing."""
        (self.root / ".github/workflows/cookbook-book.yml").unlink()
        self._git_add_all()
        code, out, err = self._run_main()
        self.assertEqual(code, 1, f"stdout={out!r} stderr={err!r}")
        self.assertIn("is missing from", err)
        self.assertIn("BOOK_WORKFLOW_NAME", err)


if __name__ == "__main__":
    unittest.main()
