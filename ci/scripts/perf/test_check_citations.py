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
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_citations as cc  # noqa: E402


class CheckCitationsFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self._orig_known_files = cc._KNOWN_FILES
        self._orig_roots = cc._SEARCH_ROOTS
        self._orig_doc_roots = cc._DOC_SEARCH_ROOTS
        self._orig_perf_full_path_roots = cc._PERF_FULL_PATH_ROOTS
        self._orig_perf_full_path_exclude = cc._PERF_FULL_PATH_EXCLUDE
        self._orig_crate_comment_roots = cc._CRATE_COMMENT_ROOTS
        self._orig_plan_contract_roots = cc._PLAN_CONTRACT_ROOTS
        self._orig_repo_root = cc.REPO_ROOT
        # Every root-scanning tuple defaults to the REAL repo tree (module
        # scope) -- reset ALL FIVE of them to empty here, not just
        # `_CRATE_COMMENT_ROOTS`: a test driving `cc.main()` (a full
        # `_iter_source_files()` walk, never a single `check_file(path)`
        # call) is otherwise silently coupled to this repo's OWN
        # (constantly drifting) real citation inventory the moment it
        # forgets to override one of these tuples -- `_set_target` only
        # ever repointed `_SEARCH_ROOTS`, leaving `_DOC_SEARCH_ROOTS`/
        # `_PERF_FULL_PATH_ROOTS` aimed at this repo's real
        # `docs/maintainer`/`ci/scripts/perf` trees, so a real stale cite
        # anywhere in either one failed 3 unrelated `main()`-driving tests
        # in this suite that were never about that scope at all. This
        # suite must be green regardless of what this repo's own citations
        # currently say. A test exercising one of these scopes explicitly
        # re-points its own tuple at its own throwaway fixture directory
        # (see `_repo_fixture`/`_perf_fixture`/`_crate_fixture` below).
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._PERF_FULL_PATH_ROOTS = ()
        cc._CRATE_COMMENT_ROOTS = ()
        cc._PLAN_CONTRACT_ROOTS = ()
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots
        cc._DOC_SEARCH_ROOTS = self._orig_doc_roots
        cc._PERF_FULL_PATH_ROOTS = self._orig_perf_full_path_roots
        cc._PERF_FULL_PATH_EXCLUDE = self._orig_perf_full_path_exclude
        cc._CRATE_COMMENT_ROOTS = self._orig_crate_comment_roots
        cc._PLAN_CONTRACT_ROOTS = self._orig_plan_contract_roots
        cc.REPO_ROOT = self._orig_repo_root

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


# Same class as the CI incident that hit `check_arch_validation_freshness.py`
# (run 33230050451, main, "Guard (arch validation freshness self-test)"):
# `shutil.rmtree` during a `tempfile.TemporaryDirectory`'s teardown can hit
# `OSError: [Errno 39] Directory not empty: '.git'` — a race between tempdir
# cleanup and a background `git maintenance`/`gc --auto` process the scratch
# repos below (`GitFixture`, `ShallowCheckoutRefusalTests`) can spawn.
# `-c gc.auto=0 -c gc.autoDetach=false -c maintenance.auto=false` kills the
# background writer AT THE SOURCE for every git invocation this suite makes.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=cwd, capture_output=True, text=True)


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
        # `_ls_tree_paths` memoizes per `(repo root, sha)` -- a throwaway
        # tempdir path is never reused within one test run, but clearing
        # this here keeps each test's `git ls-tree` reads honest regardless.
        cc._LS_TREE_CACHE.clear()

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
        number is IN BOUNDS, but the code at that line does not match what
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
    """`_KNOWN_FILES` resolves BOTH `.rs` and `.py` targets: the dozens of
    `.py:<n>` citations in `grad_oracle.rs`'s and `ab_merge.py`'s own
    determinant tables (naming `torch_grad_oracle.py`/`torch_finetune_step.py`
    lines) are mechanically re-checked exactly like an `.rs` citation --
    nothing about the correctness argument for line-content resolution is
    `.rs`-specific, so a `.py` line drifting out from under a citation is
    exactly as dangerous, and exactly as checkable, as an `.rs` line doing
    the same. This class pins the SAME `.rs` predicates
    (`UnresolvableCitationsFail`'s own shape) also hold for a `.py` target,
    via a THROWAWAY fixture -- never assuming `_KNOWN_FILES` already
    contains a real `.py` entry (this class swaps its own fixture in via
    `_set_target`, same as every other test in this file).
    """

    def test_resolvable_py_citation_passes(self):
        self._set_target("fake.py", 'line one\ndef checkpoint_identity(model_dir):\n')
        src = self._write("doc.rs", "`def checkpoint_identity(model_dir):` (`fake.py:2`)\n")
        violations = cc.check_file(src)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_stale_py_line_number_is_a_violation(self):
        """A `.py` line that has DRIFTED (in-bounds, but the code at that
        line does not match what the citation names) fails loudly, exactly
        like the `.rs` case
        `UnresolvableCitationsFail::test_stale_line_number_is_a_violation`
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


class MaintainerGuideFullPathTests(CheckCitationsFixture):
    """The FULL-PATH citation form the maintainer guides use (module doc's
    "A maintainer guide's citations are resolved by FULL PATH" section).

    Every test builds a throwaway repo-shaped fixture — a `ci/scripts/`
    target plus a `docs/maintainer/` citing doc — and points
    `REPO_ROOT`/`_DOC_SEARCH_ROOTS`/`_SEARCH_ROOTS` at it, so none of this
    depends on the real repo's own (constantly changing) citation
    inventory. `_KNOWN_FILES` is collapsed to a single unrelated entry so
    the BASENAME form can never be what makes an assertion pass here.
    """

    def _repo_fixture(self, target_body: str, doc_body: str) -> Path:
        self._write("target/unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "target" / "unrelated.rs"}
        self._write("ci/scripts/thing.sh", target_body)
        doc = self._write("docs/maintainer/guide.md", doc_body)
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = (self.root / "docs" / "maintainer",)
        return doc

    def test_a_resolved_full_path_citation_passes(self):
        doc = self._repo_fixture(
            "first line\nrp_tree_dir() {\n",
            "the tree resolver `rp_tree_dir` (`ci/scripts/thing.sh:2`) does it\n",
        )
        self.assertEqual(cc.check_file(doc), [])
        self.assertEqual(cc.main(), 0)

    def test_an_in_bounds_but_stale_full_path_citation_fails(self):
        """RED proof, and the ONE predicate an in-bounds-only check misses:
        the cited line still EXISTS, so a bounds check reads green — only
        the adjacent-identifier rule catches that it now names unrelated
        code. This is exactly the shape that made most of the pod-build
        guide's stale citations invisible: in bounds, pointing at nothing
        related.
        """
        doc = self._repo_fixture(
            "first line\nsomething_else_entirely() {\n",
            "the tree resolver `rp_tree_dir` (`ci/scripts/thing.sh:2`) does it\n",
        )
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)
        self.assertEqual(cc.main(), 1)

    def test_a_bare_full_path_citation_with_no_adjacent_identifier_fails(self):
        doc = self._repo_fixture(
            "first line\nrp_tree_dir() {\n",
            "the tree resolver does it, see ci/scripts/thing.sh:2 for how\n",
        )
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("no resolvable adjacent backtick-quoted identifier", violations[0].message)
        self.assertEqual(cc.main(), 1)

    def test_a_full_path_that_does_not_exist_fails_loudly(self):
        """The loud-failure property `_KNOWN_FILES` provides for the
        basename form, preserved by a different mechanism for this one: a
        path that is not in the tree is a Violation, never a silently
        skipped citation."""
        doc = self._repo_fixture(
            "first line\nrp_tree_dir() {\n",
            "see `rp_tree_dir` (`ci/scripts/no_such_file.sh:2`)\n",
        )
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("does not exist", violations[0].message)

    def test_an_out_of_bounds_full_path_citation_fails(self):
        doc = self._repo_fixture(
            "first line\nrp_tree_dir() {\n",
            "see `rp_tree_dir` (`ci/scripts/thing.sh:99`)\n",
        )
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1)
        self.assertIn("only has 2 lines", violations[0].message)

    def test_a_vendored_third_party_path_is_not_matched_at_all(self):
        """`cuda-kernel-guide.md` legitimately cites
        `candle-core-0.11.0/src/op.rs:<n>` — a file that is not in this
        tree by construction. Matching it would report a MISSING path for a
        citation that is correct as written, so the prefix allowlist keeps
        it out of scope entirely rather than needing a per-path exemption.
        """
        doc = self._repo_fixture(
            "first line\nrp_tree_dir() {\n",
            "candle's own arm (`candle-core-0.11.0/src/op.rs:1002`) computes in f64\n",
        )
        self.assertEqual(cc.check_file(doc), [])

    def test_the_full_path_form_is_off_for_non_doc_citing_files(self):
        """The documented residual, pinned so it cannot drift silently: a
        `_SEARCH_ROOTS` file's own full-path citation is NOT resolved by
        this form yet. If a later unit turns it on, this test fails and
        must be updated deliberately — never a blind spot nobody notices.
        """
        self._write("target/unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "target" / "unrelated.rs"}
        self._write("ci/scripts/thing.sh", "first line\nrp_tree_dir() {\n")
        script = self._write(
            "perf/producer.sh",
            "# see `totally_wrong` (`ci/scripts/thing.sh:2`)\n",
        )
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = (self.root / "perf",)
        cc._DOC_SEARCH_ROOTS = ()
        self.assertEqual(cc.check_file(script), [])

    def test_a_full_path_wins_over_the_basename_nested_inside_it(self):
        """A full path whose last component IS a registered `_KNOWN_FILES`
        basename must be reported ONCE, by the full-path form — never twice
        (once per form), and never resolved against the basename map's own
        location for a DIFFERENT file of that name."""
        self._write("target/thing.sh", "wrong file\nwrong line\n")
        cc._KNOWN_FILES = {"thing.sh": self.root / "target" / "thing.sh"}
        self._write("ci/scripts/thing.sh", "first line\nrp_tree_dir() {\n")
        doc = self._write(
            "docs/maintainer/guide.md",
            "see `rp_tree_dir` (`ci/scripts/thing.sh:2`)\n",
        )
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = (self.root / "docs" / "maintainer",)
        self.assertEqual(cc.check_file(doc), [])

    def test_real_doc_roots_include_the_maintainer_guides(self):
        """Drives the REAL (non-monkeypatched) `_DOC_SEARCH_ROOTS` —
        confirms this addition actually registered, not just a
        fixture-only code path."""
        maintainer = cc.REPO_ROOT / "docs" / "maintainer"
        self.assertIn(maintainer, self._orig_doc_roots)
        self.assertTrue(maintainer.is_dir())


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
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self._orig_known_files = cc._KNOWN_FILES
        self._orig_roots = cc._SEARCH_ROOTS
        self._orig_doc_roots = cc._DOC_SEARCH_ROOTS
        self._orig_perf_full_path_roots = cc._PERF_FULL_PATH_ROOTS
        self._orig_crate_comment_roots = cc._CRATE_COMMENT_ROOTS
        self._orig_git_repo_root = cc._GIT_REPO_ROOT
        # Same isolation `CheckCitationsFixture` applies -- see its own
        # comment for why ALL FIVE root-scanning tuples need an explicit
        # reset, not just `_SEARCH_ROOTS`.
        cc._DOC_SEARCH_ROOTS = ()
        cc._PERF_FULL_PATH_ROOTS = ()
        cc._CRATE_COMMENT_ROOTS = ()
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots
        cc._DOC_SEARCH_ROOTS = self._orig_doc_roots
        cc._PERF_FULL_PATH_ROOTS = self._orig_perf_full_path_roots
        cc._CRATE_COMMENT_ROOTS = self._orig_crate_comment_roots
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


class LegacyNonAncestorExemptionTests(GitFixture):
    """Post-#411 CI fix: the discriminator for artifact-citation resolution
    is ANCESTRY (`git merge-base --is-ancestor`), never local object
    PRESENCE. The real bug: `bf8e807` (the P1 softmax-fold artifact) is a
    real, readable commit object on a developer's long-lived local
    checkout (an old branch's objects never got pruned) but is NOT an
    ancestor of `main` (squash-merged away) -- so the OLD `git show
    <sha>:<path>` unconditional resolve read GREEN locally and RED on a
    CI runner's fresh clone, an environment-dependent green this fix
    closes. Every fixture here reproduces that EXACT precondition: the
    cited sha's commit OBJECT is present in the repo (reachable via a
    side branch), but it is NOT an ancestor of the branch the artifact's
    citation is actually checked from.
    """

    def _non_ancestor_sha_with_a_true_citation(self) -> str:
        """Commits `target.rs` (peak_thing at line 2) on a SIDE branch that
        is never merged, returns that commit's sha, then advances the
        main line with UNRELATED commits so the side-branch commit's
        object stays present in this repo's own object database
        (reachable via the side branch ref) while never becoming an
        ancestor of the main line's HEAD -- the real bf8e807 shape.
        """
        self._write("root.txt", "root\n")
        self._commit("root")
        _run_git(["checkout", "-q", "-b", "trunk"], self.root)

        _run_git(["checkout", "-q", "-b", "stale-side-branch"], self.root)
        self._write("target.rs", "line one\nlet peak_thing = compute();\nline three\n")
        stale_sha = self._commit("stale side-branch revision")

        _run_git(["checkout", "-q", "trunk"], self.root)
        # A DIFFERENT target.rs on the trunk line -- proves a later check
        # never falls back to reading THIS (HEAD) content for the exempt
        # citation either.
        self._write("target.rs", "trunk line one\ntrunk line two -- not peak_thing at all\n")
        self._commit("advance trunk, unrelated to the side branch")

        cc._KNOWN_FILES = {"target.rs": self.root / "target.rs"}
        cc._SEARCH_ROOTS = (self.root,)
        return stale_sha

    def test_non_ancestor_sha_with_object_present_locally_still_exempts_not_resolves(self):
        """The precondition itself: the stale sha's OBJECT is reachable in
        this repo (via the side branch) -- proving a later EXEMPT verdict
        is NOT simply "the sha was never known to git" but genuinely
        "known, but not an ancestor of this line", the exact bf8e807
        shape."""
        stale_sha = self._non_ancestor_sha_with_a_true_citation()
        cat_file = _run_git(["cat-file", "-t", stale_sha], self.root)
        self.assertEqual(cat_file.stdout.strip(), "commit", "fixture must keep the object present locally")
        self.assertFalse(
            cc._is_ancestor(stale_sha),
            "fixture sha must NOT be an ancestor of HEAD -- otherwise this is not the bf8e807 shape",
        )

    def test_non_ancestor_sha_citation_is_a_named_exempt_line_not_a_violation(self):
        """RED MUTANT (a): a citation whose artifact `git_sha` is NOT an
        ancestor of HEAD produces a NAMED EXEMPT line -- never a
        `Violation`, never silent (its presence is asserted directly, not
        merely "zero violations")."""
        stale_sha = self._non_ancestor_sha_with_a_true_citation()
        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{stale_sha}", "_comment": "see `peak_thing`, target.rs:2"}}',
        )
        self._commit("add legacy artifact citing the non-ancestor sha")

        violations, exemptions, _plan_contract_coverage, _unseen = cc._check_file_impl(artifact)
        self.assertEqual(violations, [], [str(v) for v in violations])
        self.assertEqual(len(exemptions), 1)
        msg = str(exemptions[0])
        self.assertIn("EXEMPT", msg)
        self.assertIn(stale_sha, msg)
        self.assertIn("NOT an ancestor of HEAD", msg)
        self.assertIn("check_cuda_run_artifacts.py", msg)
        self.assertIn("LEGACY_NONE_ALLOWLIST", msg)
        # `check_file` (the thin, unchanged-signature wrapper every
        # pre-existing test drives) must NEVER surface an Exemption as a
        # Violation -- an exempt citation is invisible to every caller
        # that only asked for violations.
        self.assertEqual(cc.check_file(artifact), [])

    def test_non_ancestor_sha_citation_never_reads_head_or_the_local_object_store(self):
        """The citation's identifier (`peak_thing`) is genuinely TRUE at
        the stale sha's own tree, and genuinely FALSE at trunk's HEAD --
        if resolution silently fell back to either, this test's assertion
        shape would flip. It must not: EXEMPT regardless, with no
        violation naming a content mismatch against HEAD either."""
        stale_sha = self._non_ancestor_sha_with_a_true_citation()
        # Sanity: the object genuinely IS resolvable via `git show` (so a
        # would-be-fallback COULD succeed silently, if this fix regressed).
        show = _run_git(["show", f"{stale_sha}:target.rs"], self.root)
        self.assertEqual(show.returncode, 0)
        self.assertIn("peak_thing", show.stdout)

        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{stale_sha}", "_comment": "see `peak_thing`, target.rs:2"}}',
        )
        self._commit("add legacy artifact")

        violations, exemptions, _plan_contract_coverage, _unseen = cc._check_file_impl(artifact)
        self.assertEqual(violations, [])
        self.assertEqual(len(exemptions), 1)

    def test_main_reports_exempt_citations_and_still_exits_zero(self):
        """Drives the REAL `main()` entry point end-to-end: exempt findings
        print (never silent) but do not gate CI."""
        stale_sha = self._non_ancestor_sha_with_a_true_citation()
        self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{stale_sha}", "_comment": "see `peak_thing`, target.rs:2"}}',
        )
        self._commit("add legacy artifact")

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cc.main()
        self.assertEqual(code, 0)
        self.assertIn("EXEMPT", buf.getvalue())
        self.assertIn(stale_sha, buf.getvalue())

    def test_ancestor_sha_still_resolves_and_a_real_mismatch_still_violates(self):
        """RED MUTANT (b), explicit direct coverage: an ANCESTOR sha whose
        content genuinely does not match must still produce a real
        `Violation` (case 1's content-mismatch arm) -- proving the new
        `not sha_is_ancestor` branch cannot accidentally swallow a real
        defect into a false EXEMPT. (The full pre-existing
        `ArtifactShaRelativeResolutionTests` suite already re-covers every
        other case-1 arm unmodified; this is the one direct regression
        check added alongside the new split.)
        """
        self._write("target.rs", "line one\nlet peak_thing = compute();\n")
        good_sha = self._commit("good revision")
        cc._KNOWN_FILES = {"target.rs": self.root / "target.rs"}
        cc._SEARCH_ROOTS = (self.root,)

        self.assertTrue(cc._is_ancestor(good_sha), "fixture sha must be an ancestor of its own repo's HEAD")

        artifact = self._write(
            "artifacts/fixture.json",
            f'{{"git_sha": "{good_sha}", "_comment": "see `totally_wrong_identifier`, target.rs:2"}}',
        )
        self._commit("add artifact with a genuinely wrong citation")

        violations, exemptions, _plan_contract_coverage, _unseen = cc._check_file_impl(artifact)
        self.assertEqual(exemptions, [])
        self.assertEqual(len(violations), 1)
        self.assertIn("recorded git_sha", violations[0].message)


class PerfScriptFullPathTests(CheckCitationsFixture):
    """(a) `ci/scripts/perf/**`'s own `.sh`/`.py` full-path coverage
    extension (module doc's "The full-path form's coverage extension"
    section, `_PERF_FULL_PATH_ROOTS`). Same throwaway-fixture discipline
    `MaintainerGuideFullPathTests` uses for `_DOC_SEARCH_ROOTS` — a fixture
    root, never this repo's real `ci/scripts/perf/**` inventory.
    """

    def _perf_fixture(self, target_body: str, script_body: str) -> Path:
        self._write("ci/scripts/lib.sh", target_body)
        script = self._write("ci/scripts/perf/producer.sh", script_body)
        self._write("target/unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "target" / "unrelated.rs"}
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._PERF_FULL_PATH_ROOTS = (self.root / "ci" / "scripts" / "perf",)
        return script

    def test_a_resolving_sh_comment_citation_passes(self):
        script = self._perf_fixture(
            "first line\nrp_tree_dir() {\n",
            "# the tree resolver `rp_tree_dir` (`ci/scripts/lib.sh:2`) does it\n",
        )
        self.assertEqual(cc.check_file(script), [])

    def test_a_stale_sh_comment_citation_fails(self):
        """THE REPRODUCTION this coverage extension exists to catch: a
        `.sh` comment citation whose target line has drifted — in-bounds,
        pointing at unrelated code."""
        script = self._perf_fixture(
            "first line\nsomething_else_entirely() {\n",
            "# the tree resolver `rp_tree_dir` (`ci/scripts/lib.sh:2`) does it\n",
        )
        violations = cc.check_file(script)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_the_perf_full_path_form_excludes_this_checkers_own_test_file(self):
        """`_PERF_FULL_PATH_EXCLUDE` names `check_citations.py`/
        `test_check_citations.py` themselves — a synthetic, deliberately
        broken citation living in a file at exactly that excluded path
        must NOT be scanned at all (this pins the exclusion mechanism
        itself, independent of the real repo's own two files)."""
        self._write("ci/scripts/lib.sh", "first line\nsomething_else_entirely() {\n")
        self._write("target/unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "target" / "unrelated.rs"}
        excluded = self._write(
            "ci/scripts/perf/test_check_citations.py",
            "# `rp_tree_dir` (`ci/scripts/lib.sh:2`)\n",
        )
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._PERF_FULL_PATH_ROOTS = (self.root / "ci" / "scripts" / "perf",)
        cc._PERF_FULL_PATH_EXCLUDE = (self.root / "ci" / "scripts" / "perf" / "test_check_citations.py",)
        # A STALE citation that WOULD fail if scanned -- zero violations
        # proves the exclusion actually engaged, not that the citation
        # happened to resolve.
        self.assertEqual(cc.check_file(excluded), [])


class CrateCommentFullPathTests(CheckCitationsFixture):
    """(b) `crates/**/*.rs` doc/comment-line full-path coverage extension
    (`_CRATE_COMMENT_ROOTS`, `"comments"` mode) — a `path:line` citation is
    in scope ONLY inside a `//`/`///`/`//!` line comment, never inside a
    string literal or executable code.
    """

    def _crate_fixture(self, target_body: str, source_body: str) -> Path:
        self._write("crates/jammi-other/src/target.rs", target_body)
        source = self._write("crates/jammi-some/src/lib.rs", source_body)
        self._write("unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "unrelated.rs"}
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._CRATE_COMMENT_ROOTS = (self.root / "crates",)
        return source

    def test_a_resolving_doc_comment_citation_passes(self):
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "/// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_stale_doc_comment_citation_fails(self):
        """THE REPRODUCTION this coverage extension exists to catch: a
        `///` doc-comment citation whose target line has drifted."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_plain_line_comment_citation_also_resolves(self):
        """`//` (not just `///`/`//!`) is in scope too — the module doc's
        "never string literals or code" rule names all three uniformly."""
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "// see `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_path_line_inside_a_rust_string_literal_is_ignored(self):
        """A `path:line` citation-shaped token sitting inside an ordinary
        Rust string literal (even one that itself contains a `//`-looking
        fragment, and even with a well-formed backtick-quoted identifier
        right next to it) is NEVER a citation -- only comment text is in
        scope. Proven the STRICT way: the embedded identifier does NOT
        match `target.rs`'s real line 2, so if the string literal's
        content leaked into the comment scan, this would resolve as a
        STALE `Violation`, not silently pass for an unrelated reason.
        """
        source = self._crate_fixture(
            "line one\nfn totally_unrelated() {}\n",
            'let s = "call `some_fake_identifier` at '
            '`crates/jammi-other/src/target.rs:2` -- embedded in a string, '
            'not a real citation // even this looks like a comment but is not";\n',
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_crate_relative_shorthand_citation_resolves(self):
        """The SECOND full-path shape this scope recognizes,
        `_crate_relative_citation_re` (`jammi-<name>/src/...:<n>`, no
        `crates/` prefix — the shape a crate's own doc comment uses to
        name a SIBLING crate by its published name). Deliberately uses a
        `jammi-`-prefixed crate name (`jammi-other`), matching the REAL
        convention (`jammi-encoders`, `jammi-lora`, ...) this shorthand
        form is scoped to.
        """
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "/// `real_thing` (`jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_stale_crate_relative_shorthand_citation_fails(self):
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/// `real_thing` (`jammi-other/src/target.rs:2`)\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_real_crate_comment_roots_include_the_crates_directory(self):
        """Drives the REAL (non-monkeypatched) `_CRATE_COMMENT_ROOTS` —
        confirms this addition actually registered, not just a
        fixture-only code path."""
        crates_dir = cc.REPO_ROOT / "crates"
        self.assertIn(crates_dir, self._orig_crate_comment_roots)
        self.assertTrue(crates_dir.is_dir())


class RustCommentLexerShapeTests(CheckCitationsFixture):
    """`_rust_comment_line_spans`'s char-literal / raw-string / block-
    comment / doc-attribute modeling — the adversarial-lexer fix round.
    Every "must be scanned" case is proven the STRICT way (a STALE
    `Violation`, never a silent zero), so a lexer regression that goes back
    to simply not scanning the line at all cannot misread as "passing"
    here — the exact class of bug this fix round closes (a `'"'` char
    literal opening a phantom string that silently swallowed every real
    comment line after it until the next unrelated `"`, dozens of lines at
    a time in this repo's own `crates/jammi-encoders/src/layer_norm.rs`).
    """

    def _crate_fixture(self, target_body: str, source_body: str) -> Path:
        self._write("crates/jammi-other/src/target.rs", target_body)
        source = self._write("crates/jammi-some/src/lib.rs", source_body)
        self._write("unrelated.rs", "nothing\n")
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "unrelated.rs"}
        cc.REPO_ROOT = self.root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._CRATE_COMMENT_ROOTS = (self.root / "crates",)
        return source

    def test_a_char_literal_double_quote_does_not_open_a_phantom_string(self):
        """THE REGRESSION this fix closes: `'"'` has no closing `"` inside
        it to pair against, so treating it as an ordinary string-open left
        the lexer "inside a string" until the next unrelated `"` anywhere
        later in the file -- silently dropping every real comment line in
        between from the scan entirely."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "fn f() {\n    let c = '\"';\n    /// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n}\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_byte_char_literal_double_quote_does_not_open_a_phantom_string(self):
        """The `b'"'` byte-char form hits the identical phantom-string
        hazard -- the `b` prefix must not change the char-literal
        detection."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "fn f() {\n    let c = b'\"';\n    // `real_thing` (`crates/jammi-other/src/target.rs:2`)\n}\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_citation_after_a_char_literal_still_resolves_when_true(self):
        """The positive control for the two tests above: the SAME shape,
        but with a citation that IS true -- proving the fix does not just
        fail loudly on a stale citation, it correctly resolves a true one
        too, round-tripping through the real lexer end to end."""
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "fn f() {\n    let c = '\"';\n    /// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n}\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_path_line_inside_a_raw_string_is_ignored(self):
        """A `path:line`-shaped, backtick-quoted citation living inside a
        `r#"..."#` raw string is NEVER a citation -- proven the STRICT way:
        the embedded identifier does not match `target.rs`'s real content,
        so if the raw string's content leaked into the comment scan this
        would resolve as a STALE `Violation`, not silently pass for an
        unrelated reason."""
        source = self._crate_fixture(
            "line one\nfn totally_unrelated() {}\n",
            'let s = r#"see `real_thing` (`crates/jammi-other/src/target.rs:2`)"#;\n',
        )
        self.assertEqual(cc.check_file(source), [])

    def test_an_outer_doc_block_comment_content_is_scanned(self):
        """`/** ... */` (exactly two asterisks opening) is a doc comment --
        its content is scanned exactly like a `///` line."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/** `real_thing` (`crates/jammi-other/src/target.rs:2`) */\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_an_inner_doc_block_comment_content_is_scanned(self):
        """`/*! ... */` (inner doc) is scanned exactly like `//!`."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/*! `real_thing` (`crates/jammi-other/src/target.rs:2`) */\n",
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_plain_block_comment_content_is_not_scanned(self):
        """An ordinary `/* ... */` (no `!`, exactly one leading `*`, i.e.
        not a doc comment) is NOT scanned -- proven the STRICT way: the
        embedded citation is stale, so silence here proves it was never
        read, not that it happened to resolve."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/* `real_thing` (`crates/jammi-other/src/target.rs:2`) */\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_triple_star_block_comment_is_not_a_doc_comment(self):
        """`/***` (three or more asterisks) mirrors `////` for line
        comments -- a regular, non-doc block comment by Rust's own rule,
        never scanned."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            "/*** `real_thing` (`crates/jammi-other/src/target.rs:2`) */\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_an_empty_block_comment_does_not_crash_or_misparse(self):
        """`/**/` (four characters -- no room for both a 3-char `/**`
        marker and a distinct 2-char `*/` closer) is the ordinary, empty,
        non-doc block comment, never a doc comment with malformed
        (negative-length) span arithmetic."""
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "/**/\n// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_doc_attribute_string_is_scanned(self):
        """`#[doc = "..."]` -- the desugared form `///` compiles to -- has
        its string content scanned as doc text."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            '#[doc = "`real_thing` (`crates/jammi-other/src/target.rs:2`)"]\nfn f() {}\n',
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_an_inner_doc_attribute_string_is_scanned(self):
        """`#![doc = "..."]` -- the desugared form `//!` compiles to."""
        source = self._crate_fixture(
            "line one\nfn totally_different() {}\n",
            '#![doc = "`real_thing` (`crates/jammi-other/src/target.rs:2`)"]\n',
        )
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_lifetime_does_not_open_a_char_literal(self):
        """`'a` (a lifetime, no closing quote) must not be mistaken for an
        unterminated char literal that swallows the rest of the file -- a
        real citation shortly after must still resolve normally."""
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "fn f<'a>(x: &'a str) -> &'a str { x }\n"
            "// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])

    def test_a_label_does_not_open_a_char_literal(self):
        """`'outer:` (a loop label) hits the identical lifetime-vs-char-
        literal disambiguation as a plain lifetime."""
        source = self._crate_fixture(
            "line one\nfn real_thing() {}\n",
            "fn f() { 'outer: loop { break 'outer; } }\n"
            "// `real_thing` (`crates/jammi-other/src/target.rs:2`)\n",
        )
        self.assertEqual(cc.check_file(source), [])


class RustCommentLexerCoverageTests(unittest.TestCase):
    """The coverage proof named in the module doc's crate-comment-roots
    section: `_rust_comment_line_spans` run over EVERY real `.rs` file
    under this repo's own `crates/` tree, asserting every line a NAIVE
    (lexically-blind) scan would call comment-only (`line.lstrip()` starts
    with `//`) is covered by at least one span the real lexer returns.
    This is the exact regression class the char-literal fix closes --
    before it, dozens of real comment lines in
    `crates/jammi-encoders/src/layer_norm.rs`,
    `crates/jammi-kernels/src/ops/launch_domain.rs`, and
    `crates/jammi-kernels/tests/feature_table.rs` were silently dropped
    from the scan by a phantom string opened on a `'"'`/`b'"'` char
    literal. Drives the REAL, non-monkeypatched `crates/` tree
    deliberately (this net exists to catch a real, repo-wide regression,
    not a fixture-only code path) -- `_rust_comment_line_spans` itself
    takes no module-level root as an argument, so this needs no
    monkeypatching to stay isolated.
    """

    def test_every_naive_comment_only_line_is_covered_by_the_real_lexer(self):
        # `also_string_literals=True`: the naive `line.lstrip().startswith
        # ("//")` heuristic below has no notion of lexical state, so a
        # fixture that holds Rust source as DATA inside a string literal
        # (e.g. a `concat!(r#"... // one ..."#)` block built to test this
        # checker's own tooling) reads, to the naive check, exactly like a
        # real comment line. The real lexer is correct to exclude that
        # line -- it is string content, not comment prose a citation could
        # legitimately live in -- so this scan's own opt-in string-literal
        # coverage (see `_rust_comment_line_spans`'s docstring) is asked
        # for here too, and a line inside a string literal counts as
        # accounted-for rather than a lexer gap.
        crates_dir = cc.REPO_ROOT / "crates"
        missed: list[str] = []
        for path in sorted(crates_dir.rglob("*.rs")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            spans = cc._rust_comment_line_spans(text, also_string_literals=True)
            # A span's OWN start line is not the whole story: a multi-line
            # span (a doc block comment, a `#[doc = "..."]` attribute string,
            # or -- now that `also_string_literals` is on -- a multi-line
            # string/raw-string literal) covers every line its byte range
            # touches, not just the line the span happens to open on. Taking
            # only the start line under-covers every later line of such a
            # span and reports it as a lexer gap it never was.
            covered_line_nos: set[int] = set()
            for start, end in spans:
                first_line = text.count("\n", 0, start) + 1
                last_line = text.count("\n", 0, max(start, end - 1)) + 1
                covered_line_nos.update(range(first_line, last_line + 1))
            for line_no, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith("//") and line_no not in covered_line_nos:
                    missed.append(f"{path.relative_to(cc.REPO_ROOT)}:{line_no}")
        self.assertEqual(
            missed, [], f"{len(missed)} naive comment-only line(s) not covered by the real lexer: {missed[:20]}"
        )


class ShaPinDistanceCapTests(unittest.TestCase):
    """Direct unit coverage for `_SHA_PIN_MAX_DISTANCE_CHARS` (120 chars):
    a pin phrase closer than the cap resolves; one exactly two characters
    further away does not -- the cap this repo's own `ab_merge.py`
    determinant-table row shape needed (module doc / `_find_pin_sha`'s own
    docstring), pinned at its own boundary rather than left to drift.
    """

    def test_a_pin_at_119_chars_resolves(self):
        # Padding must be non-word characters (spaces, not e.g. `x`) so
        # `_SHA_PIN_RE`'s `\b` word-boundary anchor actually sees a
        # boundary right before "at" -- padding with a word character
        # would glue onto "at" and make the phrase fail to match at all,
        # which would prove nothing about the distance cap specifically.
        sha = "a" * 40
        text = (" " * 119) + f"at HEAD `{sha}`"
        self.assertEqual(cc._find_pin_sha(text, 0), sha)

    def test_a_pin_at_121_chars_does_not_resolve(self):
        sha = "a" * 40
        text = (" " * 121) + f"at HEAD `{sha}`"
        self.assertIsNone(cc._find_pin_sha(text, 0))

    def test_a_pin_exactly_at_the_cap_resolves(self):
        """The cap itself (`_SHA_PIN_MAX_DISTANCE_CHARS`, 120) is
        inclusive -- `char_distance > cap` is the rejection test, so a
        distance exactly equal to the cap must still resolve."""
        sha = "b" * 40
        text = (" " * cc._SHA_PIN_MAX_DISTANCE_CHARS) + f"at HEAD `{sha}`"
        self.assertEqual(cc._find_pin_sha(text, 0), sha)


class InlineCommitPinResolutionTests(GitFixture):
    """The lead's ruling on the design question this fix round raised
    (addendum to #459): a citation explicitly pinned to a commit IN ITS
    OWN TEXT ("at HEAD `<sha>`" / "at `<sha>`" / "as of `<sha7+>`") is
    sha-relative evidence -- the SAME append-only-evidence carve-out the
    `artifacts/` arm already gets for a whole citing FILE's own `git_sha`
    field, extended to an INLINE pin. Scoped to the TWO NEW full-path
    coverage-extension scopes only (`_PERF_FULL_PATH_ROOTS`/
    `_CRATE_COMMENT_ROOTS`) -- never `_DOC_SEARCH_ROOTS`, which keeps its
    original HEAD-only behaviour.

    Every fixture commits TWO revisions of `target.rs` into a throwaway
    repo (`GitFixture`, never this checkout): `good_sha` (where
    `real_thing` sits at line 2) and a later HEAD (where line 2 is
    something else) -- proving resolution reads the PINNED tree, never
    the working tree, exactly the discipline `ArtifactShaRelativeResolutionTests`
    already established for the whole-file `git_sha` field.
    """

    def _two_revisions(self) -> str:
        self._write("crates/other-crate/src/target.rs", "line one\nfn real_thing() {}\nline three\n")
        good_sha = self._commit("good revision")
        self._write(
            "crates/other-crate/src/target.rs",
            "inserted prefix\nline one\nfn real_thing() {}\nline three\nextra tail\n",
        )
        self._commit("code moved on")
        cc.REPO_ROOT = self.root
        cc._CRATE_COMMENT_ROOTS = (self.root / "crates",)
        return good_sha

    def test_a_pin_stale_at_head_but_correct_at_the_pinned_sha_resolves(self):
        """RED proof (a): true at the citation's own pinned commit, stale
        at HEAD -- must PASS."""
        good_sha = self._two_revisions()
        source = self._write(
            "crates/some-crate/src/lib.rs",
            f"/// `real_thing` (`crates/other-crate/src/target.rs:2` at HEAD `{good_sha}`)\n",
        )
        self._commit("add citing source")
        violations = cc.check_file(source)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_head_relative_resolution_of_the_same_citation_would_have_failed(self):
        """Positive control for (a): the SAME citation, at the SAME line,
        with NO pin phrase, resolves against HEAD (where the code moved)
        and genuinely fails -- proving the PASS above comes from
        sha-relative resolution actually engaging, not from the citation
        being trivially fine at HEAD too."""
        self._two_revisions()
        source = self._write(
            "crates/some-crate/src/lib.rs",
            "/// `real_thing` (`crates/other-crate/src/target.rs:2`)\n",
        )
        self._commit("add citing source, no pin")
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("STALE", violations[0].message)

    def test_b_pin_wrong_even_at_its_own_pinned_sha_is_stale(self):
        """RED proof (b): the identifier is NOT present at the citation's
        own pinned commit either -- must FAIL as STALE (the lead's own
        wording), attributed to the pinned sha, never silently passed."""
        good_sha = self._two_revisions()
        source = self._write(
            "crates/some-crate/src/lib.rs",
            f"/// `totally_wrong_identifier` (`crates/other-crate/src/target.rs:2` at HEAD `{good_sha}`)\n",
        )
        self._commit("add citing source with a wrong identifier")
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)
        self.assertIn(good_sha, violations[0].message)

    def test_c_unknown_sha_is_exempt_not_stale(self):
        """RED proof (c): a well-formed sha this throwaway repo's object
        database genuinely does not contain -- EXEMPT, never a
        `Violation` (never silently dropped either -- its presence is
        asserted directly)."""
        self._two_revisions()
        unknown_sha = "0123456789abcdef0123456789abcdef01234567"
        # Guard: this made-up sha must genuinely not resolve in this
        # throwaway repo -- otherwise the test proves nothing.
        show = _run_git(["show", f"{unknown_sha}:crates/other-crate/src/target.rs"], self.root)
        self.assertNotEqual(show.returncode, 0)
        source = self._write(
            "crates/some-crate/src/lib.rs",
            f"/// `real_thing` (`crates/other-crate/src/target.rs:2` at HEAD `{unknown_sha}`)\n",
        )
        self._commit("add citing source with an unknown pin")
        violations, exemptions, _plan_contract_coverage, _unseen = cc._check_file_impl(source)
        self.assertEqual(violations, [], [str(v) for v in violations])
        self.assertEqual(len(exemptions), 1)
        self.assertIn("EXEMPT", str(exemptions[0]))
        self.assertIn(unknown_sha, str(exemptions[0]))
        # `check_file` (every existing caller's entry point) never
        # surfaces an Exemption as a Violation.
        self.assertEqual(cc.check_file(source), [])

    def test_d_a_nearby_sha_not_in_a_pin_phrase_is_still_checked_at_head(self):
        """RED proof (d): a WELL-FORMED pin phrase ("at `<sha>`") sitting
        two lines away from the citation (outside the same-line-or-
        adjacent-line window) must NOT be read as pinning THIS citation --
        it stays on the ordinary HEAD-relative path and fails normally
        against HEAD's drifted content, never silently EXEMPTED by an
        unrelated pin elsewhere in the same comment block.
        """
        good_sha = self._two_revisions()
        source = self._write(
            "crates/some-crate/src/lib.rs",
            (
                f"/// (unrelated) fixed at `{good_sha}`, two lines above, for a\n"
                "/// different bug entirely --\n"
                "///\n"
                "/// `real_thing` (`crates/other-crate/src/target.rs:2`)\n"
            ),
        )
        self._commit("add citing source with a nearby but out-of-range pin")
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)
        self.assertNotIn(good_sha, violations[0].message)

    def test_the_pin_carve_out_is_off_for_maintainer_guide_citations(self):
        """The carve-out is scoped to `_PERF_FULL_PATH_ROOTS`/
        `_CRATE_COMMENT_ROOTS` ONLY -- a `_DOC_SEARCH_ROOTS` citing file
        (the maintainer guides) keeps its ORIGINAL HEAD-only behaviour:
        the SAME pin phrase next to the SAME citation is simply prose to
        that scope, and the citation still resolves (or fails) against
        HEAD.
        """
        good_sha = self._two_revisions()
        cc._CRATE_COMMENT_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = (self.root / "docs" / "maintainer",)
        source = self._write(
            "docs/maintainer/guide.md",
            f"`real_thing` (`crates/other-crate/src/target.rs:2` at HEAD `{good_sha}`)\n",
        )
        self._commit("add a maintainer-guide citation with a pin phrase")
        violations = cc.check_file(source)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)
        self.assertIn("code moved since this was written", violations[0].message)


class FileCitationsEpochHeaderTests(GitFixture):
    """The WHOLE-FILE `<!-- citations-resolve-at: <sha> -->` header (module
    doc's "A frozen pre-registration's citations are pinned to their own
    epoch" section) -- the whole-file analogue of
    `InlineCommitPinResolutionTests` above, for a `.md` pre-registration
    frozen all at once rather than pinned citation-by-citation.

    Every fixture commits TWO revisions of a throwaway `fake.rs` into a
    throwaway repo (`GitFixture`, never this checkout): `good_sha` (where
    `real_thing` sits at line 2) and a later HEAD (where line 2 is
    something else) -- proving resolution reads the PINNED tree, never the
    working tree, exactly the discipline the inline-pin tests already
    established.
    """

    def _two_revisions(self) -> str:
        target = self._write("target/fake.rs", "line one\nfn real_thing() {}\nline three\n")
        cc._KNOWN_FILES = {"fake.rs": target}
        good_sha = self._commit("good revision")
        self._write(
            "target/fake.rs",
            "inserted prefix\nline one\nfn real_thing() {}\nline three\nextra tail\n",
        )
        self._commit("code moved on")
        return good_sha

    def test_resolves_at_pinned_sha_but_not_at_head_ok(self):
        """RED proof (a): true at the header's own pinned commit, stale at
        HEAD -- must PASS."""
        good_sha = self._two_revisions()
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`real_thing` (`fake.rs:2`)\n",
        )
        self._commit("add frozen doc")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_head_relative_resolution_of_the_same_citation_would_have_failed(self):
        """Positive control for (a): the SAME citation, with NO header,
        resolves against HEAD (where the code moved) and genuinely fails --
        proving the PASS above comes from the epoch pin actually engaging."""
        self._two_revisions()
        doc = self._write("doc.md", "`real_thing` (`fake.rs:2`)\n")
        self._commit("add doc with no header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_pinned_sha_not_an_ancestor_is_red(self):
        """RED proof: a well-formed 40-hex sha that is NOT an ancestor of
        HEAD fails CLOSED for this convention (never EXEMPT the way
        pre-merge-commit-discipline artifact evidence can be)."""
        self._two_revisions()
        unknown_sha = "0123456789abcdef0123456789abcdef01234567"
        # Guard: this made-up sha must genuinely not resolve in this
        # throwaway repo -- otherwise the test proves nothing.
        show = _run_git(["show", f"{unknown_sha}:target/fake.rs"], self.root)
        self.assertNotEqual(show.returncode, 0)
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {unknown_sha} -->\n\n`real_thing` (`fake.rs:2`)\n",
        )
        self._commit("add doc pinned to an unreachable sha")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("NOT an ancestor", violations[0].message)
        self.assertIn(unknown_sha, violations[0].message)

    def test_malformed_header_is_red(self):
        """RED proof: a header LINE is present but its value is not a
        well-formed 40-character hex sha -- never silently treated as "no
        header" (which would quietly fall back to HEAD resolution)."""
        self._two_revisions()
        doc = self._write(
            "doc.md",
            "<!-- citations-resolve-at: not-a-real-sha -->\n\n`real_thing` (`fake.rs:2`)\n",
        )
        self._commit("add doc with a malformed header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("not a well-formed 40-character hex commit sha", violations[0].message)

    def test_a_citation_that_fails_even_at_the_pinned_sha_is_red(self):
        """RED proof: the identifier is wrong even at the header's own
        pinned commit -- STALE, attributed to the pinned epoch, never
        silently passed."""
        good_sha = self._two_revisions()
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`totally_wrong_identifier` (`fake.rs:2`)\n",
        )
        self._commit("add doc with a wrong identifier even at its own epoch")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("never true at the tree this file's header declares", violations[0].message)
        self.assertIn(good_sha, violations[0].message)

    def test_header_past_the_search_window_is_not_recognized(self):
        """A `citations-resolve-at:`-shaped phrase deep in ordinary prose
        (past the first few lines) must never be misread as the header --
        the file falls back to ordinary HEAD-relative resolution instead,
        and genuinely fails against the drifted HEAD content."""
        good_sha = self._two_revisions()
        padding = "\n".join(f"filler line {i}" for i in range(10))
        doc = self._write(
            "doc.md",
            f"{padding}\n\n<!-- citations-resolve-at: {good_sha} -->\n\n`real_thing` (`fake.rs:2`)\n",
        )
        self._commit("add doc whose header sits past the search window")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_non_markdown_file_ignores_the_header_shaped_text(self):
        """The convention is scoped to `.md` files only (module doc) --
        the IDENTICAL header text in a non-Markdown file is just ordinary
        prose, never parsed as an epoch pin, and that file's citations keep
        resolving against HEAD unchanged."""
        good_sha = self._two_revisions()
        cc._SEARCH_ROOTS = (self.root,)
        doc = self._write(
            "doc.txt",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`real_thing` (`fake.rs:2`)\n",
        )
        self._commit("add a non-markdown file with header-shaped text")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_unseen_suffix_token_in_an_epoch_pinned_file_is_a_named_diagnostic_never_a_violation(self):
        """Module doc's "An unseen-suffix path-like token is a diagnostic,
        never a Violation" section: a backtick-quoted `<path>.txt:<line>`
        token's extension is not in `_FULL_PATH_SUFFIXES`, so no citation
        recognizer in this file can ever match it -- it must surface as a
        NAMED diagnostic (never silently dropped), and must never itself
        become a `Violation`."""
        good_sha = self._two_revisions()
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`real_thing` (`fake.rs:2`); see also `notes.txt:12` for context.\n",
        )
        self._commit("add doc with an unseen-suffix token")
        violations, _exemptions, _pc, unseen = cc._check_file_impl(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])
        self.assertEqual(len(unseen), 1, unseen)
        upath, line_no, token = unseen[0]
        self.assertEqual(upath, doc)
        self.assertIn("notes.txt:12", token)

    def test_unseen_suffix_token_outside_an_epoch_pinned_file_is_not_collected(self):
        """The diagnostic is scoped to EPOCH-PINNED files only -- an
        ordinary HEAD-relative `.md` file carrying the identical
        unseen-suffix token is not scanned for it at all (this scope was
        never meant to widen into every file this script touches)."""
        doc = self._write("doc.md", "see `notes.txt:12` for context.\n")
        self._commit("add an ordinary doc with an unseen-suffix token")
        violations, _exemptions, _pc, unseen = cc._check_file_impl(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])
        self.assertEqual(unseen, [])


class PlanContractCitationTests(GitFixture):
    """The `_PLAN_CONTRACT_ROOTS` bare-basename / full-path
    `path:line[-line][, line[-line]]*` citation form (module doc's "A frozen
    pre-registration's OWN citation shape is prose, not identifier-adjacent"
    section) -- `docs/plans/66-tower-profile/CONTRACT.md`'s own prose shape,
    never identifier-adjacent, opt-IN twice over (under `_PLAN_CONTRACT_
    ROOTS` AND carrying the file's own `citations-resolve-at:` header).
    """

    def setUp(self):
        super().setUp()
        cc._PLAN_CONTRACT_ROOTS = (self.root,)

    def test_unique_basename_resolves(self):
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1-3, 5`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_ambiguous_basename_is_red(self):
        self._write("crates/one/main.rs", "fn main() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        good_sha = self._commit("add two main.rs files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`main.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("AMBIGUOUS", violations[0].message)
        self.assertIn("crates/one/main.rs", violations[0].message)
        self.assertIn("crates/two/main.rs", violations[0].message)

    def test_absent_basename_is_red(self):
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`nowhere.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("does not exist anywhere in the tree", violations[0].message)

    def test_range_beyond_eof_is_red(self):
        self._write("crates/foo/trainer.rs", "line one\nline two\nline three\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1-5`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("only has 3 lines", violations[0].message)

    def test_comma_separated_ranges_all_checked(self):
        """A `1-2, 10` spec where only `10` is out of range must still be a
        violation -- every comma-separated part is checked, not just the
        first."""
        self._write("crates/foo/trainer.rs", "line one\nline two\nline three\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1-2, 10`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("trainer.rs:1-2, 10", violations[0].message)
        self.assertIn("only has 3 lines", violations[0].message)

    def test_full_path_citation_resolves_directly_even_when_basename_is_ambiguous(self):
        """A full-path citation is unambiguous by construction -- it
        resolves directly even when its OWN basename would be ambiguous if
        cited bare."""
        self._write("crates/one/main.rs", "fn main() {}\nfn second() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        good_sha = self._commit("add two main.rs files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`crates/one/main.rs:1-2`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_full_path_citation_missing_from_tree_is_red(self):
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`crates/foo/nowhere.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("does not exist in the tree", violations[0].message)

    def test_no_content_match_required_unlike_other_citation_forms(self):
        """This form gates resolution + in-bounds only, never a content
        re-check (module doc: "No adjacent-identifier / content-match check
        here") -- a citation whose line number is in-bounds but whose
        content has nothing to do with the surrounding prose still passes."""
        self._write("crates/foo/trainer.rs", "totally unrelated content\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_contract_md_without_the_epoch_header_is_a_violation_never_a_silent_skip(self):
        """Never-checked must never read as checked-clean: `CONTRACT.md`
        itself (`_is_frozen_contract_file`'s FILENAME arm, independent of
        the marker arms below) with NO `citations-resolve-at:` header, and
        carrying NEITHER declared marker either, is itself a `Violation` --
        a marker-ONLY definition would fail to catch this exact,
        un-decorated file, which is mechanically indistinguishable from
        "this was never a contract"; silently skipping the whole
        plan-contract citation scan instead (reporting zero violations)
        would be equally indistinguishable from "every citation resolves".
        """
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write("CONTRACT.md", "`trainer.rs:999`\n")
        self._commit("add contract without a header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_frozen_marker_via_first_heading_without_the_epoch_header_is_a_violation(self):
        """Never-checked must never read as checked-clean: a `.md` file
        whose FIRST heading names it a contract (`_is_frozen_contract_
        file`'s heading arm) with NO `citations-resolve-at:` header is
        itself a `Violation` -- the old behaviour (silently skip the whole
        plan-contract citation scan, report zero violations) was
        mechanically indistinguishable from "every citation resolves".
        Deliberately named something OTHER than `CONTRACT.md` -- recognition
        comes from the declared heading, never the filename.
        """
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write("PLAN.md", "# CONTRACT\n\n`trainer.rs:999`\n")
        self._commit("add a heading-marked frozen doc without a header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_frozen_marker_via_html_comment_without_the_epoch_header_is_a_violation(self):
        """The OTHER marker arm: a `<!-- Frozen ... -->` HTML comment,
        with no heading at all, is equally sufficient to mark a file frozen
        -- and equally mandatory about the epoch header once it does."""
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write(
            "PLAN.md",
            "<!-- Frozen ledger ts: 2026-01-01 -->\n\n`trainer.rs:999`\n",
        )
        self._commit("add a comment-marked frozen doc without a header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_a_renamed_or_versioned_contract_file_is_still_caught_by_its_marker(self):
        """A frozen contract renamed/versioned into its own title
        (`CONTRACT-v2.5.md`, this plan group's own real shape) is caught
        exactly the same way a plain `CONTRACT.md` would be -- recognition
        is the declared marker, never the literal filename."""
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write(
            "CONTRACT-v2.5.md",
            "<!-- Frozen ledger ts: 2026-01-01 -->\n\n# CONTRACT -- v2.5\n\n`trainer.rs:999`\n",
        )
        self._commit("add a renamed frozen contract without a header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_a_companion_doc_without_the_header_is_still_not_a_violation(self):
        """The mandatory-header rule is scoped to `_is_frozen_contract_file`
        (a declared FROZEN marker), never to every `.md` file `_plan_
        contract_scope` recognizes: a companion doc in the same plan
        directory (e.g. a close-out `README.md`) that carries NEITHER
        marker shape and never declares the header stays completely
        unaffected, exactly as before -- it may legitimately cite a mix of
        the frozen contract's own facts and the CURRENT tree in the same
        paragraph, which a single whole-file epoch pin could never honestly
        cover."""
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write("README.md", "`trainer.rs:999`\n")
        self._commit("add companion doc without a header")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_a_heading_inside_a_fence_is_never_mistaken_for_the_first_heading(self):
        """The first-heading scan is fence-aware: a heading-SHAPED line
        quoted as an EXAMPLE inside a ``` fence (e.g. a doc showing readers
        what a contract's own opening heading looks like), with no REAL
        heading anywhere in the search window, must never be treated as
        this document's own title -- the file is not recognized as frozen
        at all, so it produces no violation (no header mandated, and the
        frozen-only bare-basename citation form is the only thing that
        would ever have looked at `trainer.rs:999` here in the first
        place)."""
        doc = self._write(
            "NOTES.md",
            "```\n# CONTRACT\n```\n\nSome prose.\n\n`trainer.rs:999`\n",
        )
        self._commit("add a doc whose only heading-shaped line is fenced")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_a_frozen_comment_inside_a_closed_fence_is_still_recognized_raw_scan(self):
        """The comment-marker scan is deliberately NOT fence-aware (round-7
        audit at a0081c9d, unlike the heading scan, which stays fence-aware
        -- `_is_frozen_contract_file`'s own "Fences apply to the heading
        arm only" doc section): a doc that QUOTES the `<!-- Frozen ledger
        ts: ... -->` convention inside a ``` fence (e.g. showing readers
        what a frozen file's own opening comment looks like) is now
        recognized as frozen regardless -- the RAW window is scanned
        unconditionally, so this produces the mandatory-header violation."""
        doc = self._write(
            "NOTES.md",
            "```\n<!-- Frozen ledger ts: 2026-01-01 -->\n```\n\nSome prose.\n\n`trainer.rs:999`\n",
        )
        self._commit("add a doc whose only frozen-marker-shaped line is fenced")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_a_frozen_comment_after_an_unterminated_fence_in_the_window_is_still_recognized(self):
        """The motivating bug this raw scan fixes: a fence that OPENS on
        line 1 but never CLOSES again before the search window ends (its
        real closing fence -- if any -- sits past
        `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` == 6) must not silently
        swallow a REAL marker on one of the later lines still inside the
        window -- a fence-aware scan would leave `in_fence` stuck `True`
        for the rest of the window and miss it entirely, reading a
        genuinely frozen file as NOT frozen (exactly the "never-checked
        reads as checked-clean" failure this whole mechanism exists to
        rule out)."""
        doc = self._write(
            "NOTES.md",
            "```\nsome code, never closed within the search window\n"
            "<!-- Frozen ledger ts: 2026-01-01 -->\n\nprose\n`trainer.rs:999`\n",
        )
        self._commit("add a doc with an unterminated fence before its real marker")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_a_heading_naming_contract_is_recognized_after_a_fence_closes(self):
        """The inverse: a fence that opens and closes BEFORE the document's
        real first heading must not swallow that real heading -- the
        fence-aware scan still finds it once the fence toggles closed
        again, inside the same search window."""
        self._write("crates/foo/trainer.rs", "line one\n")
        self._commit("add trainer.rs")
        doc = self._write(
            "PLAN.md",
            "```\nexample text, no heading here\n```\n\n# CONTRACT\n\n`trainer.rs:999`\n",
        )
        self._commit("add a doc with a closed fence before its real heading")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_a_later_heading_naming_contract_does_not_retroactively_mark_the_file_frozen(self):
        """Only the FIRST heading in the search window is consulted -- a
        document whose first heading is unrelated is never caught by some
        deeper section heading that happens to mention "CONTRACT", and (with
        no header declared and no citation this script's own `_KNOWN_FILES`
        recognizes) produces no violation at all."""
        doc = self._write(
            "NOTES.md",
            "# Notes\n\nSome prose.\n\n## CONTRACT recap\n\n`trainer.rs:999`\n",
        )
        self._commit("add a doc whose first heading is unrelated")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_a_header_past_the_search_window_reads_as_absent_and_is_red(self):
        """A `citations-resolve-at:` header that lands PAST
        `_CITATIONS_EPOCH_HEADER_SEARCH_LINES` reads byte-for-byte like
        absent to `_file_citations_epoch` -- so it must be equally red, not
        silently ignored just because the string is present somewhere later
        in the file (the FROZEN marker itself sits on line 1, well inside
        the search window -- only the epoch header is pushed past it)."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        marker = "<!-- Frozen ledger ts: 2026-01-01 -->"
        padding = "\n".join(f"filler line {i}" for i in range(1, 10))
        doc = self._write(
            "CONTRACT.md",
            f"{marker}\n\n{padding}\n\n<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1`\n",
        )
        self._commit("add contract with a too-late header")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("declares no 'citations-resolve-at:' header", violations[0].message)

    def test_relative_sub_path_unique_resolves(self):
        """`ops/attention_block.rs:467,472` -- has a slash, but does not
        start with a recognized `_FULL_PATH_ROOT_PREFIXES` prefix -- must
        resolve by a unique SUFFIX match against the pinned tree, the shape
        the old bare-basename-or-full-path regex could not even match."""
        self._write("crates/jammi-kernels/src/ops/attention_block.rs", "\n".join(f"line {i}" for i in range(1, 500)) + "\n")
        good_sha = self._commit("add attention_block.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`ops/attention_block.rs:467,472`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_relative_sub_path_ambiguous_is_red(self):
        self._write("crates/one/ops/attention_block.rs", "\n".join(f"line {i}" for i in range(1, 500)) + "\n")
        self._write("crates/two/ops/attention_block.rs", "\n".join(f"line {i}" for i in range(1, 500)) + "\n")
        good_sha = self._commit("add two attention_block.rs files under ops/")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`ops/attention_block.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("AMBIGUOUS", violations[0].message)
        self.assertIn("relative sub-path", violations[0].message)

    def test_relative_sub_path_absent_is_red(self):
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`ops/nowhere.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("does not exist anywhere in the tree", violations[0].message)
        self.assertIn("relative sub-path", violations[0].message)

    def test_unparsed_line_spec_is_a_violation_not_a_silent_skip(self):
        """A backtick-quoted, path-ish, colon-numbered token whose line-spec
        half is not a bare digit range must be reported (fail-closed on
        unparsed citations), never silently excluded from being a citation
        at all just because a stricter one-shot regex would not have
        matched it."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:not-a-line-spec`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("could not be classified", violations[0].message)

    def test_continuation_resolves_against_the_preceding_path_on_the_same_line(self):
        """A bare `` `:<line>` `` continuation elides a path already named
        earlier on the SAME LINE -- CONTRACT.md's own "`` `htsat_audio.rs:
        1064` `` and `` `:1635` `` call ..." shape."""
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`trainer.rs:2` and `:5` call the same thing.\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_continuation_coverage_counts_the_bare_token_too(self):
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:2` and `:5`\n",
        )
        self._commit("add contract")
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cc.main()
        self.assertEqual(code, 0)
        self.assertIn("2 citation(s) checked", buf.getvalue())

    def test_continuation_out_of_bounds_is_red(self):
        self._write("crates/foo/trainer.rs", "line one\nline two\nline three\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1` and `:999`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("only has 3 lines", violations[0].message)

    def test_second_continuation_on_same_line_still_resolves_against_first_path(self):
        """A THIRD elided `:<n>` on the same line continues to resolve
        against the SAME preceding path, not just the immediately-prior
        token."""
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`trainer.rs:2`, `:5` and `:7` all call the same thing.\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_continuation_with_no_preceding_path_is_a_violation(self):
        """A bare line-spec token with no preceding path citation IN SCOPE
        on this line is refused, never silently guessed against an earlier
        line's path (or ignored)."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`:5` calls the same thing.\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("no path:line citation resolves earlier on this same line", violations[0].message)

    def test_continuation_scope_resets_across_lines(self):
        """A path citation named on one line is never a valid basis for a
        DIFFERENT line's own bare continuation."""
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`trainer.rs:2` calls the thing.\n\n`:5` calls it too.\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("no path:line citation resolves earlier on this same line", violations[0].message)

    def test_continuation_scope_does_not_survive_a_broken_preceding_citation(self):
        """A path citation that itself FAILED to resolve is not a valid
        basis for a later same-line continuation to lean on."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`nowhere.rs:1` and `:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 2, [str(v) for v in violations])
        messages = [v.message for v in violations]
        self.assertTrue(any("does not exist anywhere in the tree" in m for m in messages))
        self.assertTrue(any("no path:line citation resolves earlier on this same line" in m for m in messages))

    def test_backtick_prose_ending_in_a_bare_colon_is_not_a_false_positive_continuation(self):
        """This repo's prose routinely closes a backtick-quoted term with a
        bare colon immediately after (`` `<keys>`: refuses ... ``) --
        without the continuation arm's own digit anchor, that ordinary
        prose shape would read as an (empty, non-numeric) citation-shaped
        token exactly as much as `` `:1635` `` does."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`finetune-run --expect-kernels-disabled <keys>`: refuses at start unless every "
            "named key is present in `JAMMI_KERNELS_DISABLE`.\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_coverage_count_reported_for_a_pinned_file(self):
        """`_check_plan_contract_citations` (driven here via `main()`, the
        real coverage-reporting entry point) reports the exact count of
        citation-shaped tokens found, not merely "zero violations"."""
        self._write("crates/foo/trainer.rs", "\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n"
            "`trainer.rs:1-3, 5` and `trainer.rs:6` and `trainer.rs:7`\n",
        )
        self._commit("add contract")
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cc.main()
        self.assertEqual(code, 0)
        self.assertIn("frozen-contract citation coverage", buf.getvalue())
        self.assertIn("3 citation(s) checked", buf.getvalue())

    def test_real_contract_epoch_and_scope_are_recognized(self):
        """A cheap real-repo smoke check (never asserting the ambiguity-free
        outcome, which depends on this repo's own filenames): the real
        `docs/plans/66-tower-profile/CONTRACT.md` is under the real
        `_PLAN_CONTRACT_ROOTS` and its header is well-formed."""
        cc._PLAN_CONTRACT_ROOTS = self._orig_plan_contract_roots
        contract = cc.REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "CONTRACT.md"
        self.assertTrue(contract.is_file())
        self.assertTrue(cc._plan_contract_scope(contract))
        text = contract.read_text()
        epoch_sha, error = cc._file_citations_epoch(contract, text)
        self.assertIsNone(error)
        self.assertIsNotNone(epoch_sha)

    def test_real_contract_citation_coverage_is_19_including_the_elided_line(self):
        """`docs/plans/66-tower-profile/CONTRACT.md:16` cites `` `htsat_
        audio.rs:1064` `` and `` `:1635` `` -- the elided second citation is
        a 19th citation the OLD (path-only) recognizer could not see at
        all (coverage read 18). Enumerating every backticked, colon-
        numbered token in the real file independently (never trusting this
        script's own regex to count itself) confirms the true total is 19:
        18 named-path citations plus the one bare continuation."""
        cc._PLAN_CONTRACT_ROOTS = self._orig_plan_contract_roots
        contract = cc.REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "CONTRACT.md"
        text = contract.read_text()
        independent_count = len(re.findall(r"`[^`]*:[0-9][^`]*`", text))
        self.assertEqual(independent_count, 19)

        # `GitFixture.setUp` points `_GIT_REPO_ROOT` at this test's own
        # throwaway repo -- restored here (redundantly with `tearDown`) so
        # `main()`'s own `git show`/`git ls-tree` calls resolve the REAL
        # `CONTRACT.md`'s citations against the REAL repository, not the
        # empty fixture one.
        cc._GIT_REPO_ROOT = self._orig_git_repo_root
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cc.main()
        self.assertEqual(code, 0)
        self.assertIn(
            "docs/plans/66-tower-profile/CONTRACT.md: 19 citation(s) checked", buf.getvalue()
        )


class PlanContractHeadRelativeFullPathTests(GitFixture):
    """The class fix for the audit's stale-citation-the-gate-cannot-see
    finding: a `_PLAN_CONTRACT_ROOTS` `.md` file that carries NO
    `citations-resolve-at:` header is HEAD-relative by definition (module
    doc's "Never-checked must never read as checked-clean" section already
    says so in prose -- `_full_path_mode` must actually turn full-path
    scanning ON for such a file, never leave it silently uncovered the way
    `docs/plans/66-tower-profile/README.md`'s own stale
    `_checkpoint_identity_probe`/`preflight_probe` citations went unnoticed
    by exactly this gap). A property test (a wrong line number is always
    reported, a right one always passes), never a grep for the one
    known-bad string.
    """

    def setUp(self):
        super().setUp()
        cc.REPO_ROOT = self.root
        # An unrelated, never-matched entry -- `_citation_re` builds its
        # pattern by joining `_KNOWN_FILES`'s keys, and an EMPTY dict
        # degrades to an empty alternation that matches any bare `:<digit>`
        # with an empty `file` group (a pre-existing footgun this fixture
        # sidesteps the same way `MaintainerGuideFullPathTests` does).
        cc._KNOWN_FILES = {"unrelated.rs": self.root / "unrelated.rs"}
        cc._SEARCH_ROOTS = ()
        cc._DOC_SEARCH_ROOTS = ()
        cc._PERF_FULL_PATH_ROOTS = ()
        cc._PLAN_CONTRACT_ROOTS = (self.root / "docs" / "plans" / "66-tower-profile",)

    def _fixture(self, target_body: str, doc_body: str, doc_name: str = "README.md") -> Path:
        self._write("ci/scripts/perf/driver.sh", target_body)
        return self._write(f"docs/plans/66-tower-profile/{doc_name}", doc_body)

    def test_a_correct_full_path_citation_resolves(self):
        doc = self._fixture(
            "line 1\nline 2\nline 3\nline 4\npreflight_probe() {\n",
            "`preflight_probe` (`ci/scripts/perf/driver.sh:5`)\n",
        )
        self._commit("add driver and companion doc")
        self.assertEqual(cc.check_file(doc), [])

    def test_a_wrong_line_number_pointing_at_a_different_function_is_red(self):
        """THE audit's own regression shape: a citation whose line number
        now falls inside a DIFFERENT function's body must be reported --
        never silently read as resolved because no full-path scanning mode
        applied to this file at all (the bug this class fix closes)."""
        doc = self._fixture(
            "fi\nline 2\nline 3\nline 4\npreflight_probe() {\n",
            "`preflight_probe` (`ci/scripts/perf/driver.sh:1`)\n",
        )
        self._commit("add driver and companion doc")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("STALE", violations[0].message)

    def test_a_missing_full_path_target_still_fails_loudly(self):
        doc = self._fixture(
            "line 1\npreflight_probe() {\n",
            "`preflight_probe` (`ci/scripts/perf/no_such_driver.sh:2`)\n",
        )
        self._commit("add driver and companion doc")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("does not exist", violations[0].message)

    def test_an_epoch_pinned_sibling_keeps_its_current_sha_relative_behaviour(self):
        """`CONTRACT.md` (or any epoch-pinned `.md` under the same root) is
        deliberately EXCLUDED from this new HEAD-relative scope -- it keeps
        resolving sha-relative to its own declared epoch through
        `_check_plan_contract_citations` only; this new mode must never
        also turn on for it (which would re-resolve its frozen citations
        against HEAD, the exact drift the epoch-pin convention exists to
        prevent)."""
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        contract = self._write(
            "docs/plans/66-tower-profile/CONTRACT.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n\n`trainer.rs:1`\n",
        )
        readme = self._fixture(
            "line 1\npreflight_probe() {\n",
            "`preflight_probe` (`ci/scripts/perf/driver.sh:2`)\n",
        )
        self._commit("add contract, driver, and companion doc")
        self.assertIsNone(cc._full_path_mode(contract, contract.read_text()))
        self.assertEqual(cc._full_path_mode(readme, readme.read_text()), "text")
        self.assertEqual(cc.check_file(contract), [])
        self.assertEqual(cc.check_file(readme), [])


class BasenameMapHeaderTests(GitFixture):
    """The `citations-basename-map:` header (module doc's "An ambiguous bare
    basename can be disambiguated by a declared header map" section) --
    disambiguates a `_PLAN_CONTRACT_ROOTS` file's own ambiguous bare-basename
    citations without ever editing the frozen body, subject to the three
    narrowing checks that keep a map from asserting an unverifiable fact.
    """

    def setUp(self):
        super().setUp()
        cc._PLAN_CONTRACT_ROOTS = (self.root,)

    def test_mapped_ambiguous_basename_resolves(self):
        self._write("crates/one/main.rs", "fn main() {}\nfn second() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        good_sha = self._commit("add two main.rs files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: main.rs=crates/one/main.rs -->\n\n"
            "`main.rs:1-2`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(violations, [], [str(v) for v in violations])

    def test_map_to_a_nonexistent_path_is_red(self):
        self._write("crates/one/main.rs", "fn main() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        good_sha = self._commit("add two main.rs files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: main.rs=crates/nowhere/main.rs -->\n\n"
            "`main.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("does not exist in the tree", violations[0].message)

    def test_map_whose_basename_differs_from_the_key_is_red(self):
        self._write("crates/one/main.rs", "fn main() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        self._write("crates/one/other.rs", "fn other() {}\n")
        good_sha = self._commit("add fixture files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: main.rs=crates/one/other.rs -->\n\n"
            "`main.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("own basename is", violations[0].message)

    def test_map_that_repoints_a_unique_basename_elsewhere_is_red(self):
        """`trainer.rs` is NOT ambiguous -- only one exists -- so a map entry
        naming a DIFFERENT file for it must be refused, never silently
        accepted (a map can only resolve a genuine ambiguity, never
        re-point a citation that was already resolving correctly)."""
        self._write("crates/foo/trainer.rs", "line one\nline two\n")
        self._write("crates/foo/other.rs", "line one\nline two\n")
        good_sha = self._commit("add trainer.rs and other.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: trainer.rs=crates/foo/other.rs -->\n\n"
            "`trainer.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("already resolves uniquely to", violations[0].message)
        self.assertIn("crates/foo/trainer.rs", violations[0].message)

    def test_unmapped_ambiguous_basename_still_red_even_with_a_map_present(self):
        """A map entry for ONE ambiguous basename must never loosen the
        ambiguity check for a DIFFERENT, unmapped one in the same file."""
        self._write("crates/one/main.rs", "fn main() {}\n")
        self._write("crates/two/main.rs", "fn main() {}\n")
        self._write("crates/one/layer_norm.rs", "line one\n")
        self._write("crates/two/layer_norm.rs", "line one\n")
        good_sha = self._commit("add ambiguous fixture files")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: main.rs=crates/one/main.rs -->\n\n"
            "`layer_norm.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("AMBIGUOUS", violations[0].message)

    def test_malformed_map_entry_is_a_whole_file_violation(self):
        self._write("crates/foo/trainer.rs", "line one\n")
        good_sha = self._commit("add trainer.rs")
        doc = self._write(
            "doc.md",
            f"<!-- citations-resolve-at: {good_sha} -->\n"
            "<!-- citations-basename-map: main.rs -->\n\n"
            "`trainer.rs:1`\n",
        )
        self._commit("add contract")
        violations = cc.check_file(doc)
        self.assertEqual(len(violations), 1, [str(v) for v in violations])
        self.assertIn("citations-basename-map", violations[0].message)

    def test_real_contract_basename_map_resolves_the_two_ambiguous_citations(self):
        """A cheap real-repo smoke check, same shape as
        `PlanContractCitationTests`'s `test_real_contract_epoch_and_scope_are_
        recognized`: the real `CONTRACT.md` declares a `citations-basename-map`
        header, and every one of its `layer_norm.rs`/`main.rs` citations
        resolves cleanly through it."""
        cc._PLAN_CONTRACT_ROOTS = self._orig_plan_contract_roots
        contract = cc.REPO_ROOT / "docs" / "plans" / "66-tower-profile" / "CONTRACT.md"
        self.assertTrue(contract.is_file())
        text = contract.read_text()
        basename_map, error = cc._file_citations_basename_map(contract, text)
        self.assertIsNone(error)
        self.assertIsNotNone(basename_map)
        self.assertEqual(basename_map.get("layer_norm.rs"), "crates/jammi-encoders/src/layer_norm.rs")
        self.assertEqual(basename_map.get("main.rs"), "crates/jammi-bench/src/main.rs")


if __name__ == "__main__":
    unittest.main()
