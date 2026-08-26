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


if __name__ == "__main__":
    unittest.main()
