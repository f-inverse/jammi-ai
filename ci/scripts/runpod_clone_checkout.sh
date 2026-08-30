#!/usr/bin/env bash
# Sourceable-only: `runpod_gpu_perf_ab.sh`'s own clone+checkout+wrong-tree-
# verification block (round-2 adversarial audit F1), factored into its own
# file so a hermetic test can `source` and drive this EXACT function against
# a scratch repo -- and so the REMOTE pod runs the EXACT same code, never a
# second, independently-drifting copy: `runpod_gpu_perf_ab.sh` embeds this
# file's own content VERBATIM into its `rp_run_remote` heredoc via `$(cat
# ".../runpod_clone_checkout.sh")` (the function must exist on the pod
# BEFORE any clone happens there, so a runtime `source` of a not-yet-cloned
# file cannot work — textual inlining from the LOCAL copy is the only sound
# order of operations).
#
# The checkout step itself follows this repo's own bootstrap idiom
# (runpod_lib.sh:1515) verbatim: `git checkout --quiet "$ref" || { echo
# "::error::checkout $ref failed"; ...; }` — this function `return`s 2
# instead of `exit`ing (it is meant to be CALLED, not sourced-and-run), and
# the caller propagates that return code as its own exit code.
#
# Deliberately sets NO shell options of its own (`set -e`/`-u`/`pipefail`) —
# a sourced LIBRARY should not silently change its caller's shell behavior
# (mirrors `gpu_inference_ab_git.sh`'s own doctrine).

# runpod_perf_ab_clone_and_checkout <dest> <repo_url> <git_ref> <default_branch>
#
# Clones <repo_url> into <dest> as a blobless partial clone
# (`--filter=blob:none` — the SAME pod-clone idiom runpod_lib.sh:1505
# already uses: this workload only ever needs ONE checked-out tree's file
# contents plus the full commit graph for `git merge-base`, never every
# historical blob), then sets `uploadpack.allowFilter=true` ON THAT CLONE
# (round-3 adversarial audit B1, the auditor's own reproduction of a fatal
# clone-composition bug): `gpu_inference_ab.sh`'s own inner clones
# (`clone_and_checkout`) clone AGAIN, also `--filter=blob:none`, FROM
# <dest> once this driver hands off to it. A git repo's DEFAULT
# `uploadpack.allowFilter=false` means <dest>, acting as that inner
# clone's SOURCE, cannot honor a partial-clone request -- and, being
# partial itself, cannot silently degrade to serving a FULL one either (it
# does not have every blob to serve): the inner clone would FAIL HARD
# (fatal, exit 128), not gracefully fall back. This one `git config` call,
# made ONCE here, is what makes every later inner clone against <dest>
# sound; the inner clone call sites do not re-set it themselves.
# `test_gpu_inference_ab_git_shape.py`'s own
# `ClonePartialCompositionTests` drives the REAL two-stage composition
# (this function's outer clone, then `gpu_inference_ab.sh`'s own inner
# clone command verbatim) against a scratch repo, including the RED
# control (allowFilter left unset → the inner clone fails, empirically
# rc=128).
#
# Checks out <git_ref>, and leaves the caller's shell `cd`'d into <dest> on
# success (matching the caller's own prior inline `cd jammi-ai` step, so no
# caller-side `cd` is needed after calling this).
#
# WRONG-TREE refusal (round-2 adversarial audit F1): when <git_ref> is NOT
# literally <default_branch>, this REFUSES (return 2) if the checked-out
# HEAD resolves to the EXACT SAME commit as origin/<default_branch>'s own
# tip — a non-default ref that lands on main's own commit is almost
# certainly a silently-wrong ref (a typo, an upstream default gone stale),
# not a legitimate "this branch has zero commits ahead of main yet" case
# worth trusting blindly for an A/B comparator whose entire point is
# measuring TWO DIFFERENT trees. `<git_ref> == <default_branch>` is exempt
# (an operator who deliberately targets the default branch is not making
# this mistake).
#
# Returns 0 on success, 2 on any clone/checkout/verification failure
# (this script family's own usage/infra-error bucket).
runpod_perf_ab_clone_and_checkout() {
  local dest="$1" repo_url="$2" git_ref="$3" default_branch="$4"

  git clone --quiet --filter=blob:none "$repo_url" "$dest" \
    || { echo "::error::cloning $repo_url -> $dest failed"; return 2; }

  # round-3 adversarial audit B1 (fatal clone composition, the auditor's own
  # reproduction): this clone is now ITSELF a partial (blobless) clone --
  # `gpu_inference_ab.sh`'s own inner clones (`clone_and_checkout`) clone
  # AGAIN, `--filter=blob:none`, FROM this dir (`file://$REPO_ROOT`) once
  # this driver hands off to it. A git repo's default
  # `uploadpack.allowFilter=false` means IT, acting as the source for that
  # inner clone, cannot honor a partial-clone request -- and, being partial
  # itself, cannot silently degrade to serving a FULL one either (it does
  # not have every blob to serve): the inner clone FAILS HARD (fatal, exit
  # 128), not a graceful no-op fallback. Setting this here, once, on the
  # outer clone, is what makes every later inner clone against it sound --
  # the inner clone sites themselves do not re-set it.
  git -C "$dest" config uploadpack.allowFilter true \
    || { echo "::error::setting uploadpack.allowFilter on $dest failed -- inner clones against this dir would fail hard"; return 2; }

  cd "$dest" || { echo "::error::cd $dest failed"; return 2; }

  git checkout --quiet "$git_ref" || { echo "::error::checkout $git_ref failed"; return 2; }

  if [ "$git_ref" != "$default_branch" ]; then
    local head_sha default_sha
    head_sha="$(git rev-parse HEAD)" \
      || { echo "::error::git rev-parse HEAD failed after checking out $git_ref"; return 2; }
    default_sha="$(git rev-parse "origin/$default_branch" 2>&1)" \
      || { echo "::error::git rev-parse origin/$default_branch failed -- cannot verify wrong-tree for $git_ref"; return 2; }
    if [ "$head_sha" = "$default_sha" ]; then
      echo "::error::wrong-tree refusal -- git_ref='$git_ref' resolved to the SAME commit as origin/$default_branch's own tip ($head_sha); refusing rather than silently measuring the wrong tree." >&2
      return 2
    fi
  fi

  return 0
}

# --- Allow this file to be SOURCED for its function definition alone
# (the normal use — both `runpod_gpu_perf_ab.sh`'s heredoc-inlining AND
# this file's own hermetic test source it) without ALSO being directly
# executed as a script. Mirrors `gpu_inference_ab_git.sh`'s own idiom. ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "runpod_clone_checkout.sh defines runpod_perf_ab_clone_and_checkout for sourcing only -- source it, do not execute it directly." >&2
  exit 2
fi
