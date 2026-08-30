#!/usr/bin/env python3
"""Hermetic git-SHAPE regression suite for `gpu_inference_ab_git.sh`
(round-1 adversarial audit B2). No GPU, no network, no jammi-bench binary —
builds a SCRATCH `origin` repo with real `git` subprocess calls (the SAME
`_scratch_git` idiom `test_check_ci_guard_wiring.py` already uses), then
drives the EXACT clone/checkout shapes `runpod_gpu_perf_ab.sh` uses (both
the FIXED shape and the OLD, empirically-broken one) against it, sourcing
`gpu_inference_ab_git.sh`'s own `gpu_inference_ab_ensure_history_for_merge_base`
function via a real `bash -c 'source ...; <call>'` subprocess — never a
re-implementation of that function's own logic in Python, and never a call
into `gpu_inference_ab.sh` itself (which needs a GPU + the CUDA toolchain
neither this suite's environment nor CI's hermetic runners carry).

## What this suite proves, not merely asserts

1. The OLD bug is REAL, not a hypothesized one: replicating `git clone
   --depth 1 -b <branch> <origin> <dest>` (the shape `runpod_gpu_perf_ab.sh`
   used before this fix) and then running `git merge-base origin/main HEAD`
   with NO repair step FAILS — the empirical "merge-base exits 128" the
   audit found, reproduced here mechanically rather than taken on faith.
2. The FIXED clone shape (a full, non-single-branch clone + a separate
   `checkout`, `runpod_gpu_perf_ab.sh`'s own new shape) already creates a
   working `origin/main` on its own, with or without the repair function.
3. The repair function is ALSO defense-in-depth: called against the SAME
   OLD, buggy single-branch/shallow clone shape from (1), it recovers a
   working `origin/main` and a resolvable merge-base — so a caller of
   `gpu_inference_ab.sh` whose OWN checkout is still shaped like (1) is not
   left stranded either.
4. `dry_run=1` short-circuits before touching git at all (returns 0 even
   against a directory with no git repository whatsoever).

Round-2 adversarial audit additions:
5. `GitLibExitArmTests` — `gpu_inference_ab_ensure_history_for_merge_base`'s
   OWN `2` (unshallow fetch failed) and `75` (advisory `origin/main`
   refresh-fetch failed) exit arms, isolated against a shallow-vs-full
   clone whose `origin` remote is repointed at a nonexistent path (F2;
   only the `0` arm was exercised before this addition).
6. `RunpodCloneCheckoutTests` — `runpod_clone_checkout.sh`'s own
   clone+checkout+wrong-tree-verification function (F1), the SAME code
   `runpod_gpu_perf_ab.sh` inlines verbatim into its remote heredoc,
   driven against a scratch repo with a real ref, a nonexistent ref, and a
   ref that resolves to the default branch's own tip.

Run: `python3 ci/scripts/perf/test_gpu_inference_ab_git_shape.py`
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_LIB = os.path.join(PERF_DIR, "gpu_inference_ab_git.sh")

# Mirrors test_check_ci_guard_wiring.py's own `_scratch_git` convention:
# never let a scratch repo's `git` invocation kick off a background
# maintenance daemon that outlives this test process.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _rev_parse(cwd, ref):
    return _git(["rev-parse", ref], cwd).stdout.strip()


def build_scratch_origin(root):
    """A real `origin` repo (not bare — `git clone` from a local worktree
    path works fine and is simpler to seed than a bare repo) with `main`
    (two commits) and a `feature` branch that diverges from `main`'s FIRST
    commit (one commit of its own) — so `merge-base(origin/main, feature)`
    has a real, distinct, checkable answer (`main`'s first commit), never
    trivially equal to either tip. Returns `(origin_dir, merge_base_sha,
    feature_tip_sha)`.
    """
    origin = os.path.join(root, "origin")
    os.makedirs(origin)
    _git(["init", "-q"], origin)
    _git(["config", "user.email", "test@example.com"], origin)
    _git(["config", "user.name", "test"], origin)
    _git(["symbolic-ref", "HEAD", "refs/heads/main"], origin)

    with open(os.path.join(origin, "a.txt"), "w", encoding="utf-8") as fh:
        fh.write("a\n")
    _git(["add", "a.txt"], origin)
    _git(["commit", "-q", "-m", "main: first commit"], origin)
    merge_base_sha = _rev_parse(origin, "HEAD")

    _git(["checkout", "-q", "-b", "feature"], origin)
    with open(os.path.join(origin, "b.txt"), "w", encoding="utf-8") as fh:
        fh.write("b\n")
    _git(["add", "b.txt"], origin)
    _git(["commit", "-q", "-m", "feature: diverges from main"], origin)
    feature_tip_sha = _rev_parse(origin, "HEAD")

    _git(["checkout", "-q", "main"], origin)
    with open(os.path.join(origin, "c.txt"), "w", encoding="utf-8") as fh:
        fh.write("c\n")
    _git(["add", "c.txt"], origin)
    _git(["commit", "-q", "-m", "main: second commit, after feature diverged"], origin)

    return origin, merge_base_sha, feature_tip_sha


def clone_fixed_shape(origin, dest):
    """`runpod_gpu_perf_ab.sh`'s own FIXED clone shape (round-1 adversarial
    audit B2): a full, non-single-branch clone, THEN a separate checkout —
    never `clone -b`.
    """
    _git(["clone", "--quiet", origin, dest], os.path.dirname(dest))
    _git(["checkout", "--quiet", "feature"], dest)


def clone_old_buggy_shape(origin, dest):
    """The OLD, empirically-broken shape this fix replaces: a SHALLOW,
    SINGLE-BRANCH clone straight onto the target ref via `-b`. `origin` is
    passed as a `file://` URL so `--depth` genuinely takes effect (see
    [`clone_shallow_with_broken_origin`]'s own doc for the local-clone
    quirk this avoids) — this fixture's own bug reproduction does not
    strictly NEED real shallowness (the single-branch restriction alone
    already starves `origin/main` of any tracking ref), but stating the
    shape honestly matters for a fixture whose own docstring claims it.
    """
    _git(["clone", "--quiet", "--depth", "1", "-b", "feature", f"file://{origin}", dest], os.path.dirname(dest))


def run_ensure_function(repo_root, dry_run="0"):
    """Sources `gpu_inference_ab_git.sh` and calls
    `gpu_inference_ab_ensure_history_for_merge_base` — a REAL bash
    subprocess, never a Python re-implementation of that function's logic.
    Returns the function's own exit code (0 / 2 / 75).
    """
    script = f'source "{GIT_LIB}"; gpu_inference_ab_ensure_history_for_merge_base "{repo_root}" "{dry_run}"'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def clone_shallow_with_broken_origin(origin, dest):
    """A SHALLOW clone whose `origin` remote is then repointed at a
    nonexistent path — forces
    `gpu_inference_ab_ensure_history_for_merge_base`'s OWN unshallow-fetch
    step to fail (round-2 adversarial audit F2's `2` arm: the unshallow
    fetch itself failed, a genuine infra problem).

    `origin` is passed as a `file://` URL, never a bare local path: git
    SILENTLY IGNORES `--depth` for a bare-local-path clone ("--depth is
    ignored in local clones; use file:// instead", discovered empirically
    while authoring this fixture) — the resulting clone would be a FULL
    (non-shallow) one despite `--depth 1`, which would silently skip the
    exact code path (`is-shallow-repository` → true → the unshallow-fetch
    branch) this fixture exists to isolate.
    """
    _git(["clone", "--quiet", "--depth", "1", f"file://{origin}", dest], os.path.dirname(dest))
    _git(["remote", "set-url", "origin", os.path.join(os.path.dirname(dest), "no-such-origin")], dest)


def clone_full_with_broken_origin(origin, dest):
    """A FULL (non-shallow) clone whose `origin` remote is then repointed
    at a nonexistent path — the unshallow-fetch branch is SKIPPED entirely
    (the clone is not shallow), isolating a failure of the
    explicit-refspec `origin/main` fetch ALONE (round-2 adversarial audit
    F2's `75` arm: advisory, never a hard refusal).
    """
    _git(["clone", "--quiet", origin, dest], os.path.dirname(dest))
    _git(["remote", "set-url", "origin", os.path.join(os.path.dirname(dest), "no-such-origin")], dest)


RUNPOD_SCRIPTS_DIR = os.path.dirname(PERF_DIR)
CLONE_CHECKOUT_LIB = os.path.join(RUNPOD_SCRIPTS_DIR, "runpod_clone_checkout.sh")


def run_runpod_clone_and_checkout(dest, repo_url, git_ref, default_branch):
    """Sources `runpod_clone_checkout.sh` and calls
    `runpod_perf_ab_clone_and_checkout` — a REAL bash subprocess driving the
    EXACT function `runpod_gpu_perf_ab.sh` inlines verbatim into its own
    remote heredoc (round-2 adversarial audit F1), never a Python
    re-implementation of its clone/checkout/wrong-tree-verification logic.
    """
    script = (
        f'source "{CLONE_CHECKOUT_LIB}"; '
        f'runpod_perf_ab_clone_and_checkout "{dest}" "{repo_url}" "{git_ref}" "{default_branch}"'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def merge_base_resolves(repo_root):
    """`True` iff `git merge-base origin/main HEAD` succeeds in
    `repo_root` — the SAME call `gpu_inference_ab.sh` itself makes right
    after its own `ensure_history_for_merge_base` step.
    """
    result = _git(["merge-base", "origin/main", "HEAD"], repo_root, check=False)
    return result.returncode == 0, result.stdout.strip(), result.returncode


class GitShapeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_old_single_branch_shallow_clone_is_the_proven_bug(self):
        """THE negative control (RC1: an assertion must be able to fail,
        proven the RED direction here): the OLD clone shape, with NO repair
        step, must FAIL `git merge-base origin/main HEAD` — this is the
        empirical bug the round-1 adversarial audit found (exit 128 in
        practice), reproduced mechanically, not merely asserted in prose.
        """
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")
        clone_old_buggy_shape(origin, dest)

        ok, _out, rc = merge_base_resolves(dest)
        self.assertFalse(
            ok,
            "the OLD single-branch shallow clone shape, with no repair step, was expected to FAIL "
            f"'git merge-base origin/main HEAD' (the proven bug) but it SUCCEEDED (rc={rc}) — either "
            "git's own defaults changed underneath this test, or the bug this suite exists to pin no "
            "longer reproduces the way the audit found it; re-derive, never just delete this test",
        )

    def test_fixed_clone_shape_gives_a_working_merge_base_with_no_repair_needed(self):
        """The FIXED shape (`runpod_gpu_perf_ab.sh`'s own new clone+checkout,
        [`clone_fixed_shape`]) already creates a working `origin/main` on
        its own — `git merge-base` resolves correctly to the TRUE common
        ancestor even before `gpu_inference_ab_ensure_history_for_merge_base`
        ever runs.
        """
        origin, merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")
        clone_fixed_shape(origin, dest)

        ok, resolved_sha, rc = merge_base_resolves(dest)
        self.assertTrue(ok, f"merge-base must resolve on the fixed clone shape (rc={rc})")
        self.assertEqual(
            resolved_sha,
            merge_base_sha,
            "the resolved merge-base must be the TRUE common ancestor commit, not merely 'some sha'",
        )

    def test_ensure_function_returns_0_and_repairs_the_old_buggy_shape_too(self):
        """The teeth (RC1, GREEN direction): `gpu_inference_ab_ensure_history_for_merge_base`,
        run against the SAME OLD, buggy single-branch/shallow clone
        [`test_old_single_branch_shallow_clone_is_the_proven_bug`] proved
        broken, returns 0 AND leaves `git merge-base origin/main HEAD`
        resolvable to the TRUE common ancestor afterward — the function is
        defense-in-depth robust even against the exact shape this fix
        replaces, not merely correct for the ONE new shape
        `runpod_gpu_perf_ab.sh` itself now produces.
        """
        origin, merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")
        clone_old_buggy_shape(origin, dest)

        # Sanity re-confirmation this fixture really is broken before
        # calling the function under test (belt-and-suspenders alongside
        # the dedicated negative-control test above).
        broken_ok, _out, _rc = merge_base_resolves(dest)
        self.assertFalse(broken_ok, "fixture setup drifted -- this clone must start out broken")

        rc, out, err = run_ensure_function(dest)
        self.assertEqual(rc, 0, f"expected the repair function to return 0, got {rc}\nstdout={out}\nstderr={err}")

        ok, resolved_sha, merge_rc = merge_base_resolves(dest)
        self.assertTrue(ok, f"merge-base must resolve AFTER the repair function ran (rc={merge_rc})")
        self.assertEqual(resolved_sha, merge_base_sha)

    def test_dry_run_short_circuits_before_touching_git_at_all(self):
        """`dry_run=1` must return 0 immediately, without even checking
        whether `repo_root` is a git repository — proven here against a
        directory that is NOT a git repo at all (a real git call would
        error loudly; the short-circuit must never reach one).
        """
        not_a_repo = os.path.join(self.root, "not-a-repo")
        os.makedirs(not_a_repo)

        rc, out, err = run_ensure_function(not_a_repo, dry_run="1")
        self.assertEqual(rc, 0, f"dry_run=1 must return 0 unconditionally, got {rc}\nstdout={out}\nstderr={err}")

    def test_sourcing_the_library_directly_as_a_script_refuses(self):
        """`gpu_inference_ab_git.sh` is sourceable-only (its own doc) — a
        DIRECT execution (never sourced) must refuse loudly (exit 2) rather
        than silently doing nothing useful.
        """
        result = subprocess.run(["bash", GIT_LIB], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("source it", result.stderr)


class GitLibExitArmTests(unittest.TestCase):
    """round-2 adversarial audit F2: `gpu_inference_ab_ensure_history_for_merge_base`'s
    OWN `2` (unshallow fetch failed) and `75` (advisory `origin/main`
    refresh-fetch failed) exit arms — only the `0` arm was exercised
    before this addition.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_unshallow_fetch_failure_returns_2(self):
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")
        clone_shallow_with_broken_origin(origin, dest)

        rc, out, err = run_ensure_function(dest)
        self.assertEqual(
            rc,
            2,
            f"a shallow clone whose origin cannot be reached must return 2 (the unshallow fetch "
            f"itself failed, a genuine infra problem)\nstdout={out}\nstderr={err}",
        )

    def test_origin_main_refresh_fetch_failure_returns_75(self):
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")
        clone_full_with_broken_origin(origin, dest)

        rc, out, err = run_ensure_function(dest)
        self.assertEqual(
            rc,
            75,
            f"a non-shallow clone whose origin cannot be reached must return 75 (advisory "
            f"origin/main refresh-fetch failure, never a hard refusal)\nstdout={out}\nstderr={err}",
        )


class RunpodCloneCheckoutTests(unittest.TestCase):
    """round-2 adversarial audit F1: `runpod_clone_checkout.sh`'s own
    clone+checkout+wrong-tree-verification function, driven against a
    scratch repo — EXECUTES the driver's own logic (never merely asserted
    from reading the source), including the wrong-tree refusal a bad or
    silently-defaulted GIT_REF earns.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_real_diverging_ref_checks_out_cleanly(self):
        origin, _merge_base_sha, feature_tip_sha = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")

        rc, out, err = run_runpod_clone_and_checkout(dest, origin, "feature", "main")
        self.assertEqual(rc, 0, f"stdout={out}\nstderr={err}")
        self.assertEqual(_rev_parse(dest, "HEAD"), feature_tip_sha)

    def test_a_nonexistent_ref_refuses_with_exit_2(self):
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        dest = os.path.join(self.root, "dest")

        rc, out, err = run_runpod_clone_and_checkout(dest, origin, "does-not-exist", "main")
        self.assertEqual(rc, 2, f"stdout={out}\nstderr={err}")

    def test_a_wrong_tree_ref_that_resolves_to_the_default_branch_head_refuses_with_exit_2(self):
        """The wrong-tree refusal itself (round-2 adversarial audit F1): a
        NON-default ref that happens to resolve to the exact SAME commit
        as origin/main's own CURRENT tip must refuse, even though the
        checkout itself succeeded cleanly.
        """
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        main_tip_sha = _rev_parse(origin, "HEAD")  # build_scratch_origin leaves origin checked out on main
        _git(["branch", "accidental-alias", "main"], origin)
        dest = os.path.join(self.root, "dest")

        rc, out, err = run_runpod_clone_and_checkout(dest, origin, "accidental-alias", "main")
        self.assertEqual(rc, 2, f"stdout={out}\nstderr={err}")
        self.assertIn("wrong-tree", err)
        self.assertIn(main_tip_sha, err)

    def test_deliberately_targeting_the_default_branch_is_exempt_from_the_wrong_tree_check(self):
        """`git_ref == default_branch` is the ONE legitimate case where
        landing on main's own tip is expected, not a bug — exempt.
        """
        origin, _merge_base_sha, _feature_tip = build_scratch_origin(self.root)
        main_tip_sha = _rev_parse(origin, "HEAD")
        dest = os.path.join(self.root, "dest")

        rc, out, err = run_runpod_clone_and_checkout(dest, origin, "main", "main")
        self.assertEqual(rc, 0, f"stdout={out}\nstderr={err}")
        self.assertEqual(_rev_parse(dest, "HEAD"), main_tip_sha)


if __name__ == "__main__":
    unittest.main()
