#!/usr/bin/env bash
# Sourceable-only library: the git-history-preparation step
# `gpu_inference_ab.sh`'s own producer needs before `git merge-base
# origin/main HEAD` can succeed. Extracted into its OWN file (never embedded
# inline in that script's procedural body) so a hermetic test
# (`test_gpu_inference_ab_git_shape.py`) can `source` this ONE function
# without executing `gpu_inference_ab.sh`'s own clone/build/measurement body
# (which needs a GPU + the CUDA toolchain neither this repo's CI runners nor
# this test's own environment carry). Defines exactly one function and runs
# nothing on source — safe to `source` from any shell, any working directory.
#
# ## Round-1 adversarial audit B2: the bug this file exists to fix
#
# `runpod_gpu_perf_ab.sh`'s own initial clone onto the rented pod used to be
# `git clone --depth 1 -b "$GIT_REF" "$GIT_REPO" jammi-ai` — a SHALLOW,
# SINGLE-BRANCH clone. A single-branch clone's remote config scopes its OWN
# default fetch refspec to that ONE branch alone
# (`+refs/heads/<GIT_REF>:refs/remotes/origin/<GIT_REF>`) — a later `git
# fetch origin main` against that remote config does NOT create a
# `refs/remotes/origin/main` tracking ref at all (git fetches the commit
# objects but has no configured DESTINATION ref to write them under), so
# `git merge-base origin/main HEAD` run afterward fails outright
# (`fatal: ... unknown revision`, empirically exit 128) — `origin/main`
# genuinely never existed as a local ref. `runpod_gpu_perf_ab.sh`'s own
# clone is fixed (full, non-single-branch, so `origin/main` exists from the
# initial clone onward) — this function is defense-in-depth for any OTHER
# caller of `gpu_inference_ab.sh` whose own checkout might still be
# single-branch/shallow, and the one place that FORCES the tracking ref to
# exist regardless: an EXPLICIT destination refspec
# (`+refs/heads/main:refs/remotes/origin/main`) creates
# `refs/remotes/origin/main` unconditionally, independent of whatever
# narrower refspec the remote's own config carries.
#
# Deliberately sets NO shell options of its own (`set -e`/`-u`/`pipefail`) —
# a sourced LIBRARY should not silently change its caller's shell behavior;
# `gpu_inference_ab.sh` (the real caller) already sets `set -uo pipefail` at
# its own top, and this file's own hermetic test controls its own subshell
# explicitly.

# gpu_inference_ab_ensure_history_for_merge_base <repo_root> [dry_run]
#
# Returns (bash function exit codes — this FILE's own `source` always
# returns 0; only calling the function itself can return non-zero):
#   0  -- `origin/main` is now a resolvable local ref, with enough history
#         for `git merge-base origin/main HEAD` to succeed against it.
#   2  -- the (only-if-actually-shallow) unshallow fetch failed — a genuine
#         infra/usage problem: this repo can never compute a real
#         merge-base without real history, and no fallback exists.
#   75 -- the explicit-refspec `origin/main` fetch failed — NEUTRAL (issue
#         #335 round-1 adversarial audit B2), not a hard refusal: if a
#         PREVIOUS clone step already populated `origin/main` (the normal
#         case after `runpod_gpu_perf_ab.sh`'s own full, non-single-branch
#         initial clone), this is a transient refresh failure against an
#         already-usable ref, not a reason to hard-refuse the whole run; if
#         `origin/main` genuinely does not exist yet either, this leaves the
#         caller unable to compute a merge-base at all, which the caller's
#         own subsequent `git merge-base` call surfaces as ITS OWN failure —
#         deliberately not double-refused here.
gpu_inference_ab_ensure_history_for_merge_base() {
  local repo_root="$1"
  local dry_run="${2:-0}"

  if [ "$dry_run" = "1" ]; then
    return 0
  fi

  if [ "$(git -C "$repo_root" rev-parse --is-shallow-repository 2>&1)" = "true" ]; then
    if ! git -C "$repo_root" fetch --unshallow --quiet origin; then
      echo "::error::'git fetch --unshallow' failed -- cannot compute a real merge-base off a shallow checkout." >&2
      return 2
    fi
  fi

  if ! git -C "$repo_root" fetch --quiet origin +refs/heads/main:refs/remotes/origin/main; then
    echo "::warning::'git fetch origin +refs/heads/main:refs/remotes/origin/main' failed -- neutral (nothing to compare safely this run); the caller's own 'git merge-base' call surfaces if origin/main is still unusable after this." >&2
    return 75
  fi

  return 0
}

# --- Allow this file to be SOURCED for its function definition alone (the
# normal, intended use — `gpu_inference_ab.sh` and this file's own hermetic
# test both `source` it) without ALSO being directly executed as a script
# (which would do nothing useful — the function above is never called by
# this file itself). The standard bash "am I sourced or executed" idiom. ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "gpu_inference_ab_git.sh defines gpu_inference_ab_ensure_history_for_merge_base for sourcing only -- source it, do not execute it directly." >&2
  exit 2
fi
