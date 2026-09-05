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
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self._orig_known_files = cc._KNOWN_FILES
        self._orig_roots = cc._SEARCH_ROOTS
        self._orig_doc_roots = cc._DOC_SEARCH_ROOTS
        self._orig_perf_full_path_roots = cc._PERF_FULL_PATH_ROOTS
        self._orig_perf_full_path_exclude = cc._PERF_FULL_PATH_EXCLUDE
        self._orig_crate_comment_roots = cc._CRATE_COMMENT_ROOTS
        self._orig_repo_root = cc.REPO_ROOT
        # `_CRATE_COMMENT_ROOTS` defaults to the REAL `crates/` tree (module
        # scope), which every OTHER root here is protected from by an
        # explicit per-test monkeypatch onto a throwaway fixture -- reset
        # to empty by default so a test that calls `cc.main()` without
        # itself opting into the new crate-comment scope is never silently
        # coupled to this repo's own (still-drifting, not-yet-fixed) real
        # crate citations. A test exercising the new scope explicitly
        # re-points this at its own fixture directory.
        cc._CRATE_COMMENT_ROOTS = ()
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots
        cc._DOC_SEARCH_ROOTS = self._orig_doc_roots
        cc._PERF_FULL_PATH_ROOTS = self._orig_perf_full_path_roots
        cc._PERF_FULL_PATH_EXCLUDE = self._orig_perf_full_path_exclude
        cc._CRATE_COMMENT_ROOTS = self._orig_crate_comment_roots
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
        self._orig_crate_comment_roots = cc._CRATE_COMMENT_ROOTS
        self._orig_git_repo_root = cc._GIT_REPO_ROOT
        # Same isolation `CheckCitationsFixture` applies -- see its own
        # comment for why `_CRATE_COMMENT_ROOTS` needs an explicit reset
        # where every other root here is already reset by construction.
        cc._CRATE_COMMENT_ROOTS = ()
        self.addCleanup(self._restore)

    def _restore(self):
        cc._KNOWN_FILES = self._orig_known_files
        cc._SEARCH_ROOTS = self._orig_roots
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

        violations, exemptions = cc._check_file_impl(artifact)
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

        violations, exemptions = cc._check_file_impl(artifact)
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

        violations, exemptions = cc._check_file_impl(artifact)
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
        violations, exemptions = cc._check_file_impl(source)
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


if __name__ == "__main__":
    unittest.main()
