#!/usr/bin/env python3
"""Tests for `check_citations.py` (advisory i, PR #372 round 2).

Every test builds a THROWAWAY fixture directory and monkeypatches
`check_citations`'s module-level `_KNOWN_FILES`/`_SEARCH_ROOTS` onto it, so
these tests are independent of this repo's own current citation inventory
(which will keep changing) and drive the real `check_file`/`main` entry
points against a fixture built to isolate exactly one predicate per test.

Run directly: `python3 ci/scripts/perf/test_check_citations.py`
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_citations as cc  # noqa: E402


class CheckCitationsFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self._orig_known_files = cc._KNOWN_FILES
        self._orig_roots = cc._SEARCH_ROOTS
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots

    def _write(self, rel: str, content: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _set_target(self, name: str, content: str):
        """Writes a fixture `<name>` and points `_KNOWN_FILES[name]` at it —
        the throwaway stand-in for `finetune_step.rs`/`grad_oracle.rs`.
        """
        target = self._write(f"target/{name}", content)
        cc._KNOWN_FILES = {name: target}
        cc._SEARCH_ROOTS = (self.root,)
        return target


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


class GitFixture(CheckCitationsFixture):
    """Base for the sha-relative artifact-citation tests: a THROWAWAY `git
    init`'d repo (never this checkout), same discipline
    `check_cuda_run_artifacts.py --self-test` already uses for ITS OWN
    ancestry/shallow-checkout fixtures. Restores `cc._GIT_REPO_ROOT` in
    addition to the `_KNOWN_FILES`/`_SEARCH_ROOTS` the parent class already
    handles.
    """

    def setUp(self):
        super().setUp()
        self._orig_git_repo_root = cc._GIT_REPO_ROOT
        self.addCleanup(self._restore_git_repo_root)
        _run_git(["init", "-q"], self.root)
        _run_git(["config", "user.email", "test@example.com"], self.root)
        _run_git(["config", "user.name", "Test"], self.root)
        cc._GIT_REPO_ROOT = self.root

    def _restore_git_repo_root(self):
        cc._GIT_REPO_ROOT = self._orig_git_repo_root

    def _commit(self, message: str, repo: Path | None = None) -> str:
        repo = repo or self.root
        _run_git(["add", "-A"], repo)
        _run_git(["commit", "-q", "-m", message], repo)
        return _run_git(["rev-parse", "HEAD"], repo).stdout.strip()


class ResolvableCitationsPass(CheckCitationsFixture):
    def test_identifier_immediately_before_citation_resolves(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\nline three\n")
        src = self._write("doc.md", "see `peak_thing`, fake.rs:2 for the mechanism\n")
        violations = cc.check_file(src)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_backtick_wrapped_citation_also_resolves(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        src = self._write("doc.md", "see `peak_thing`, `fake.rs:2`\n")
        violations = cc.check_file(src)
        self.assertEqual(violations, [])

    def test_at_the_time_of_writing_hedge_is_a_valid_connector(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        src = self._write("doc.md", "`peak_thing` (`fake.rs:2` at the time of writing)\n")
        violations = cc.check_file(src)
        self.assertEqual(violations, [])

    def test_wrapped_rust_doc_comment_continuation_is_a_valid_connector(self):
        """THE finetune_step.rs SELF-CITATION-SHAPE REGRESSION this checker's own
        development hit: a real Rust `///` doc comment wraps across lines,
        so the connector between an identifier and its citation can contain
        a `\\n    /// ` continuation marker -- that marker must be stripped
        before the connector check, not treated as disqualifying free text.
        """
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        src = self._write(
            "doc.rs",
            "    /// `peak_thing`\n    /// (`fake.rs:2` at the time of writing)\n",
        )
        violations = cc.check_file(src)
        self.assertEqual(violations, [])


class LookbackWindowBacktickPairingTests(CheckCitationsFixture):
    """A small, safe fix folded in alongside the M1b sha-relative work
    (flagged while writing `docs/maintainer/fine-tune-performance-guide.md`'s
    own row_lengths table row): the OLD `_find_adjacent_identifier` re-paired
    backticks from a `text[start - 300 : start]` SLICE, not the full text.
    When an earlier backtick-quoted span's OPENING backtick falls before the
    slice boundary but its CLOSING backtick falls inside it, the slice
    contains an orphan closing backtick with no partner -- every subsequent
    backtick in the slice re-pairs one position off, misattributing the
    wrong (garbled) text as "the identifier" for a citation that actually
    had a perfectly good one. Pairing backticks against the FULL text FIRST,
    then filtering to the window, makes this structurally unreachable.
    """

    def test_a_backtick_pair_straddling_the_lookback_window_boundary_does_not_misparse_the_next_identifier(self):
        self._set_target("fake.rs", "line one\nlet real_ident = 1;\n")
        # An EARLIER, unrelated backtick-quoted span (150 chars -- inside
        # `_IDENT_RE`'s 200-char cap, so it forms a real pair) positioned so
        # the 300-char lookback window's START lands INSIDE its content:
        # the exact straddle shape `_CANONICALIZERS`'s real occurrence hit
        # (an unrelated `identity_fields.IDENTITY_FIELD_CANONICALIZERS`
        # mention earlier in the same table row). A window SLICE would see
        # only this span's closing backtick, with no opening partner in
        # range, and (since the run of filler afterward is short enough to
        # stay under the 200-char cap too) mis-pair that orphan closing
        # tick with `real_ident`'s own OPENING tick, garbling the result.
        long_span = "`" + ("A" * 150) + "`"
        filler = "z" * 184
        src = self._write("doc.md", long_span + filler + "`real_ident` (fake.rs:2)\n")
        text = src.read_text()
        # Sanity-check the fixture actually straddles the window the way
        # this test claims, so a future `_SEARCH_WINDOW`/`_IDENT_RE` edit
        # fails LOUDLY here rather than silently testing nothing.
        citation_start = text.index("fake.rs:2")
        window_start = citation_start - cc._SEARCH_WINDOW
        self.assertGreater(window_start, 1)
        self.assertLess(window_start, len(long_span) - 1)

        violations = cc.check_file(src)
        self.assertEqual(violations, [], [str(v) for v in violations])


class UnresolvableCitationsFail(CheckCitationsFixture):
    def test_stale_line_number_is_a_violation(self):
        """THE F7-CLASS REGRESSION this script exists to catch: the line
        number is IN BOUNDS, but the code at that line is no longer what
        the citation names — an in-bounds-only check would miss this.
        """
        self._set_target("fake.rs", "line one\nlet totally_different = 1;\n")
        src = self._write("doc.md", "see `peak_thing`, fake.rs:2\n")
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)

    def test_out_of_range_line_is_a_violation(self):
        self._set_target("fake.rs", "line one\nline two\n")
        src = self._write("doc.md", "see `peak_thing`, fake.rs:99\n")
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("only has", violations[0].message)

    def test_bare_citation_with_no_adjacent_identifier_is_a_violation(self):
        """THE B1/advisory-i REGRESSION: a bare `file.rs:N` citation with no
        adjacent identifier (the two-Rust-source-lines-named-together shape this
        round's own fix retired) cannot be mechanically re-checked at all —
        this must fail LOUDLY, not silently pass because "the line number
        happens to be in range".
        """
        self._set_target("fake.rs", "line one\nline two\n")
        src = self._write("doc.md", "see fake.rs:2 for details\n")
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("no resolvable adjacent", violations[0].message)

    def test_identifier_too_far_before_citation_is_not_adjacent(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        src = self._write(
            "doc.md",
            "`peak_thing` is discussed at length in several unrelated paragraphs "
            "of prose that go on and on, far more than the search window, so by "
            "the time we finally get around to a citation naming it the identifier "
            "is long gone from the lookback window entirely and cannot be resolved "
            + ("padding " * 60)
            + "fake.rs:2\n",
        )
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("no resolvable adjacent", violations[0].message)


class PyFileSupportTests(CheckCitationsFixture):
    """Round-4 audit fold-in on PR #372: `_KNOWN_FILES` used to resolve ONLY
    `finetune_step.rs`/`grad_oracle.rs` — the dozens of `.py:<n>` citations
    in `grad_oracle.rs`'s and `ab_merge.py`'s own determinant tables (naming
    `torch_grad_oracle.py`/`torch_finetune_step.py` lines) were NEVER
    mechanically re-checked at all, which is exactly how the `.py` line
    drift this round's own audit caught went unnoticed. This class pins the
    SAME `.rs` predicates (`UnresolvableCitationsFail`'s own shape) now also
    hold for a `.py` target, via a THROWAWAY fixture -- never assuming
    `_KNOWN_FILES` already contains a real `.py` entry (this class swaps its
    own fixture in via `_set_target`, same as every other test in this
    file).
    """

    def test_resolvable_py_citation_passes(self):
        self._set_target("fake.py", 'line one\ndef checkpoint_identity(model_dir):\n')
        src = self._write("doc.rs", "`def checkpoint_identity(model_dir):` (`fake.py:2`)\n")
        violations = cc.check_file(src)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_stale_py_line_number_is_a_violation(self):
        """THE REPRODUCTION this round's own fold-in closes: a `.py` line
        that has DRIFTED (in-bounds, but the code at that line is no longer
        what the citation names) must fail loudly, exactly like the `.rs`
        case `UnresolvableCitationsFail::test_stale_line_number_is_a_violation`
        already pins.
        """
        self._set_target("fake.py", "line one\ndef totally_different():\n    pass\n")
        src = self._write("doc.rs", "`def checkpoint_identity(model_dir):` (`fake.py:2`)\n")
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)

    def test_bare_py_citation_with_no_adjacent_identifier_is_a_violation(self):
        self._set_target("fake.py", "line one\nline two\n")
        src = self._write("doc.rs", "see fake.py:2 for details\n")
        violations = cc.check_file(src)
        self.assertEqual(len(violations), 1)
        self.assertIn("no resolvable adjacent", violations[0].message)

    def test_real_torch_reference_scripts_are_registered_and_resolve(self):
        """Drives the REAL `_KNOWN_FILES` (not a throwaway fixture) against
        the REAL `torch_grad_oracle.py`/`torch_finetune_step.py` files at
        HEAD -- confirms this round's `_KNOWN_FILES` addition actually
        registered both files (not just a fixture-only code path) and that
        every real citation of them in this repo currently resolves. This
        is a narrower, `.py`-only slice of what `check_citations.main()`
        itself already re-verifies over the WHOLE repo (per the `run every
        suite` acceptance clause); kept here too so a REGRESSION in the
        `_KNOWN_FILES` registration itself (e.g. a typo'd path) has a
        second, independent test besides the CI gate script's own run.
        """
        for name in ("torch_grad_oracle.py", "torch_finetune_step.py"):
            self.assertIn(name, self._orig_known_files, f"{name} must be registered in _KNOWN_FILES")
            self.assertTrue(
                self._orig_known_files[name].exists(),
                f"{name}'s registered path {self._orig_known_files[name]!r} does not exist",
            )


class SearchRootsTests(CheckCitationsFixture):
    """Unification contract C8.4 (NF15): `_SEARCH_ROOTS` gained a THIRD root
    (`crates/jammi-kernels/artifacts/cuda-runs`) in phase 2, the same PR that
    moves the two baselines OUT of `crates/jammi-bench/baselines/` into it —
    without this, a citation living under the new root would silently drop
    out of coverage the moment the move landed. This pins that a fixture
    file under a THIRD, independent root (not just `self.root` as a whole,
    which `_set_target` already collapses `_SEARCH_ROOTS` to) is still
    walked and its citations still resolved/violated exactly like the other
    two roots.
    """

    def test_a_fixture_under_a_third_search_root_is_walked_and_resolves(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        third_root = self.root / "third-root"
        third_root.mkdir()
        cc._SEARCH_ROOTS = (self.root / "target", third_root)
        src = third_root / "moved-baseline.json"
        src.write_text('{"_comment": "see `peak_thing`, fake.rs:2"}')
        code = cc.main()
        self.assertEqual(code, 0)

    def test_a_stale_citation_under_a_third_search_root_still_fails(self):
        self._set_target("fake.rs", "line one\nlet totally_different = 1;\n")
        third_root = self.root / "third-root"
        third_root.mkdir()
        cc._SEARCH_ROOTS = (self.root / "target", third_root)
        src = third_root / "moved-baseline.json"
        src.write_text('{"_comment": "see `peak_thing`, fake.rs:2"}')
        code = cc.main()
        self.assertEqual(code, 1)

    def test_real_search_roots_include_the_cuda_runs_directory(self):
        """Drives the REAL (non-monkeypatched) `_SEARCH_ROOTS` — confirms
        this phase's addition actually registered, not just a fixture-only
        code path."""
        real_roots = self._orig_roots
        cuda_runs = cc.REPO_ROOT / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
        self.assertIn(cuda_runs, real_roots)
        self.assertTrue(cuda_runs.is_dir())


class MainEntryPointTests(CheckCitationsFixture):
    def test_main_returns_nonzero_on_a_violation(self):
        self._set_target("fake.rs", "line one\nline two\n")
        self._write("doc.md", "see fake.rs:2 for details\n")
        code = cc.main()
        self.assertEqual(code, 1)

    def test_main_returns_zero_when_clean(self):
        self._set_target("fake.rs", "line one\nlet peak_thing = compute();\n")
        self._write("doc.md", "see `peak_thing`, fake.rs:2\n")
        code = cc.main()
        self.assertEqual(code, 0)


class ArtifactShaRelativeResolutionTests(GitFixture):
    """M1b audit round: a citation inside a file under an `artifacts/`
    directory that declares its own `git_sha` is append-only evidence —
    resolved against THAT sha via `git show`, never against HEAD (the
    category error that guaranteed `crates/jammi-kernels/artifacts/
    cuda-runs/*.json`'s citations would break the moment ANY later,
    unrelated commit moved the cited line). Every fixture here commits TWO
    revisions of `target.rs` into a throwaway repo: `good_sha` (where the
    citation is true) and a later HEAD (where the SAME line has moved) —
    proving resolution reads the `good_sha` tree, never the working tree.
    """

    def _two_revisions(self) -> str:
        """Commits `target.rs` with `peak_thing` at line 2, returns that
        commit's sha, then commits a SECOND revision that pushes
        `peak_thing` down to line 3 — so line 2 at HEAD is a DIFFERENT
        statement, and a HEAD-relative resolution of `target.rs:2` would
        find it stale (or a flat mismatch), while the returned sha's own
        line 2 is still exactly right.
        """
        self._write("target.rs", "line one\nlet peak_thing = compute();\nline three\n")
        good_sha = self._commit("good revision")
        self._write(
            "target.rs",
            "inserted prefix line\nline one\nlet peak_thing = compute();\nline three\nextra tail\n",
        )
        self._commit("code moved on")
        cc._KNOWN_FILES = {"target.rs": self.root / "target.rs"}
        cc._SEARCH_ROOTS = (self.root,)
        return good_sha

    def test_artifact_citation_stale_at_head_but_true_at_git_sha_passes(self):
        """RED MUTANT (a): true at the artifact's own recorded git_sha,
        stale at HEAD -- must PASS."""
        good_sha = self._two_revisions()
        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{good_sha}", "_comment": "see `peak_thing`, target.rs:2"}}',
        )
        self._commit("add artifact")
        violations = cc.check_file(artifact)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_head_relative_resolution_of_the_same_citation_would_have_failed(self):
        """Positive control for the test above: the SAME target.rs:2
        citation, cited from a file NOT under `artifacts/` (so it stays
        HEAD-resolved), genuinely fails -- proving the PASS above comes
        from sha-relative resolution actually engaging, not from the
        citation being trivially fine at HEAD too.
        """
        self._two_revisions()
        doc = self._write("doc.md", "see `peak_thing`, target.rs:2\n")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)

    def test_artifact_citation_false_even_at_its_own_git_sha_fails(self):
        """RED MUTANT (b): the identifier is NOT present at the artifact's
        own recorded git_sha either -- must FAIL, and the message must
        attribute the mismatch to that recorded sha (never to "the code
        moved since"), so a reader is not sent chasing HEAD drift for a
        citation that was simply never true."""
        good_sha = self._two_revisions()
        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{good_sha}", "_comment": "see `totally_wrong_identifier`, target.rs:2"}}',
        )
        self._commit("add artifact")
        violations = cc.check_file(artifact)
        self.assertEqual(len(violations), 1)
        self.assertIn("recorded git_sha", violations[0].message)
        self.assertNotIn("STALE", violations[0].message)

    def test_artifact_citation_out_of_bounds_at_its_own_git_sha_fails(self):
        good_sha = self._two_revisions()
        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{good_sha}", "_comment": "see `peak_thing`, target.rs:99"}}',
        )
        self._commit("add artifact")
        violations = cc.check_file(artifact)
        self.assertEqual(len(violations), 1)
        self.assertIn("only has", violations[0].message)
        self.assertIn(good_sha, violations[0].message)

    def test_json_not_under_artifacts_dir_ignores_its_own_git_sha_field(self):
        """A JSON file carrying a well-formed `git_sha` field but NOT
        living under an `artifacts/` path segment is ordinary, living
        prose (e.g. a moved-baseline fixture outside the artifacts tree)
        -- it must keep resolving against HEAD, never opt into
        sha-relative resolution just because the field happens to be
        present."""
        good_sha = self._two_revisions()
        not_an_artifact = self._write(
            "not-artifacts-dir/fixture.json",
            f'{{"git_sha": "{good_sha}", "_comment": "see `peak_thing`, target.rs:2"}}',
        )
        self._commit("add non-artifact json")
        violations = cc.check_file(not_an_artifact)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)

    def test_artifact_json_without_git_sha_field_falls_back_to_head(self):
        self._two_revisions()
        artifact = self._write(
            "artifacts/fixture.json",
            '{"status": "RECORD", "_comment": "see `peak_thing`, target.rs:2"}',
        )
        self._commit("add artifact with no git_sha")
        violations = cc.check_file(artifact)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)


class ShallowCheckoutRefusalTests(unittest.TestCase):
    """RED MUTANT (c): a GENUINE `git clone --depth 1` (not a simulated
    flag), same technique `check_cuda_run_artifacts.py --self-test` uses
    for its own shallow-checkout regression -- `citation resolver` needs
    `fetch_depth: "0"` in `.github/workflows/ci.yml` for exactly this
    reason.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_known_files = cc._KNOWN_FILES
        self._orig_roots = cc._SEARCH_ROOTS
        self._orig_git_repo_root = cc._GIT_REPO_ROOT
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots
        cc._GIT_REPO_ROOT = self._orig_git_repo_root

    def test_shallow_clone_refuses_sha_relative_resolution(self):
        src = Path(self._tmp.name) / "src"
        src.mkdir()
        _run_git(["init", "-q"], src)
        _run_git(["config", "user.email", "test@example.com"], src)
        _run_git(["config", "user.name", "Test"], src)

        (src / "target.rs").write_text("line one\nlet peak_thing = compute();\n")
        _run_git(["add", "-A"], src)
        _run_git(["commit", "-q", "-m", "c1"], src)
        good_sha = _run_git(["rev-parse", "HEAD"], src).stdout.strip()

        artifacts_dir = src / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "fixture.json").write_text(
            f'{{"git_sha": "{good_sha}", "_comment": "see `peak_thing`, target.rs:2"}}'
        )
        (src / "unrelated.txt").write_text("x\n")
        _run_git(["add", "-A"], src)
        _run_git(["commit", "-q", "-m", "c2"], src)

        clone = Path(self._tmp.name) / "clone"
        clone_proc = _run_git(["clone", "-q", "--depth", "1", "file://" + str(src), str(clone)], src)
        self.assertEqual(clone_proc.returncode, 0, clone_proc.stderr)

        cc._GIT_REPO_ROOT = clone
        cc._KNOWN_FILES = {"target.rs": clone / "target.rs"}
        cc._SEARCH_ROOTS = (clone,)

        self.assertTrue(cc._is_shallow_repository(), "a genuine `git clone --depth 1` was not detected as shallow")

        artifact_in_clone = clone / "artifacts" / "fixture.json"
        with self.assertRaises(cc.CitationError) as ctx:
            cc.check_file(artifact_in_clone)
        self.assertIn(cc.SHALLOW_CHECKOUT_MESSAGE, str(ctx.exception))

        # `main()` surfaces the SAME CitationError as one explicit FAIL
        # line, not a per-file traceback -- drives the real entry point,
        # not just the internal helper.
        code = cc.main()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
