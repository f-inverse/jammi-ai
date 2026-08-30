#!/usr/bin/env bash
# gpu_inference_ab.sh -- issue #335's within-run GPU perf A/B producer:
# builds parent-HEAD and the PR change as two FULL, SIMULTANEOUSLY-RESIDENT
# clones, runs `jammi-bench gpu-inference-scale` on the SAME rented pod, back
# to back, in an order-balanced A,B,B,A interleaving, and merges the four
# legs through `gpu_inference_ab.py` (this directory) -- the SAME
# `generic_leg_identity_fields`/`generic_leg_premise_violations` refusal core
# `encode_ab.sh` already builds on.
#
# TWO MODES (issue #335's final unit): the multi-pod/both-device-model
# validation this producer's own exit criterion required before any
# enforcement is now DONE (the `--aa-null` empirical-null campaign committed
# under `ci/artifacts/gpu-perf-aa-null/`), and `gpu_inference_ab.py`'s own
# advisory band is now PRE-REGISTERED against that evidence (see that
# module's own doc). This does NOT make enforcement the default:
#   * non-enforcing (the DEFAULT — `GPU_INFERENCE_AB_ENFORCE` unset or `0`)
#     — the SAME recording-only posture this producer has always had: the
#     only hard (nonzero, non-75) refusal is a CORRECTNESS-of-measurement
#     problem (an identity/premise mismatch between legs, or a binary whose
#     own `provenance` does not match the clone it was built from). A
#     measured ratio, however far outside the pre-registered band, is
#     always recorded and printed, never gated.
#   * enforcing (`GPU_INFERENCE_AB_ENFORCE=1`) — opts THIS invocation into
#     also refusing (exit 1) a GREEN-premise run whose ratio falls OUTSIDE
#     the pre-registered band, on EITHER side (this is a NULL band, not a
#     "faster is always fine" one -- a ratio too far below 1.0 is refused
#     just as loudly as one too far above, with its own distinct,
#     direction-honest verdict; see `gpu_inference_ab.py`'s own
#     "Enforcement flip" doc for the full exit-code lattice, including the
#     `ENFORCE_INVALID_MODE` refusal below). Still OPT-IN, never this
#     script's own default: `gpu-perf-ab.yml`'s label-triggered PR path
#     never sets it, only an explicit `workflow_dispatch` with
#     `enforce: true` does. ALSO mutually exclusive with `--aa-null` (see
#     that section's own note below): this script refuses (exit 2) BEFORE
#     renting/building anything if both are requested together.
#
# ## WHY TWO FULL CLONES, NOT ONE CHECKOUT SWITCHING REFS
#
# `ModelInferenceSpec::embed_model_dir`/`::classifier_model_dir`
# (`crates/jammi-bench/src/model_inference.rs:165-175`) bake
# `env!("CARGO_MANIFEST_DIR")` -- a COMPILE-TIME constant -- into the fixture
# path each binary reads its committed `config.json`/`model.safetensors`/
# `tokenizer.json` from, joined against a RELATIVE `../../cookbook/
# fixtures/...` that resolves through whatever is CURRENTLY on disk at that
# absolute path when the binary actually RUNS, not when it was compiled.
# `gpu_inference.rs::run` now also HASHES those same files
# (`GpuInferenceTier::embed_checkpoint_*_sha256`/`infer_checkpoint_*_sha256`,
# issue #335's own D4 identity contract) at RUN time, off whatever bytes sit
# at that path at that moment.
#
# A single checkout that builds parent, `git checkout`s the PR ref IN PLACE,
# then builds the PR binary would therefore be UNSOUND the instant the
# parent binary is later RUN: its own baked fixture path now resolves
# through whatever the checkout was switched to LAST (the PR's tree), not
# the tree parent was built against -- silently measuring the wrong
# checkpoint bytes under the parent binary's own label, or (if the PR
# genuinely changed the fixture) producing a checksum on the parent leg that
# does not match what the parent binary was actually compiled to serve. Two
# clones, each holding its OWN ref checked out for its OWN binary's entire
# lifetime, is the only sound shape here -- never a single repo whose ref
# moves between builds.
#
# ## Parent = merge-base(origin/main, HEAD), NEVER HEAD^
#
# `HEAD^` is "the commit immediately before this one" -- for a PR with more
# than one commit (or a squash-merge target whose HEAD already IS the merge
# commit), that is not "main before this PR's changes landed", it is an
# arbitrary interior commit of the PR's own history (or, for a merge commit,
# an UNRELATED parent branch entirely). `git merge-base origin/main HEAD` is
# the actual common ancestor -- the real baseline this PR diverged from --
# regardless of how many commits the PR carries or whether HEAD is itself a
# merge commit.
#
# ## Order-balanced legs: A, B, B, A (never A, A, B, B)
#
# See `gpu_inference_ab.py`'s own module doc ("What actually cancels, and
# what does not", round-1 adversarial audit B4) for the full, corrected
# rationale — in short: the ORDER itself (never merely "pairing adjacent
# legs together") cancels a first-order MULTIPLICATIVE clock/thermal drift
# trend, by placing the two `b`-role legs symmetrically between the two
# `a`-role legs.
#
# ## `--aa-null`: the D6 empirical-null instrument
#
# `GPU_INFERENCE_AB_AA_NULL=1` builds the PARENT sha TWICE, from the SAME
# TWO independent clones (`clone-a`, `clone-b`) every invocation of this
# script already makes -- this mode changes ONLY which sha `clone-b` checks
# out (the parent sha, not the PR sha), never the clone COUNT (always
# exactly two clones per invocation, in either mode). Comparing a sha
# against itself, built and run independently, so the resulting ratio
# distribution is pure build+measurement+pod noise, never a real code
# difference. This is the instrument
# `gpu_inference_ab.py::PRE_REGISTERED_ADVISORY_BAND` (D6) was derived from —
# five runs, THREE primary (the other two, `pcie-p1`/`pcie-p2`, are
# committed but AUXILIARY: they ran CONCURRENTLY on one shared PCIe pod, a
# GPU-contention confound this instrument's own isolation assumption does
# not cover -- `ci/artifacts/gpu-perf-aa-null/README.md`'s own "Disclosure"
# section has the full evidence), committed under
# `ci/artifacts/gpu-perf-aa-null/` (that directory's own README.md has the
# full campaign protocol, per-run table, and characterization findings). A
# FUTURE campaign that widens this evidence base (more pods, more device
# models) re-derives the band from the WIDER committed set, never
# hand-tunes the two numbers directly -- `ci/scripts/check_aa_null_band.py`
# enforces this mechanically (see that module's own "ADVISORY
# classification" doc). `--aa-null` is MUTUALLY EXCLUSIVE with
# `GPU_INFERENCE_AB_ENFORCE=1` (checked below, before renting/building
# anything): `gpu_inference_ab.py::build_report`'s own `ENFORCE_INVALID_MODE`
# arm ALSO refuses this combination at the comparator level, but refusing
# here, at the producer's own edge, is strictly cheaper -- no PR exists to
# enforce against under `--aa-null`, and enforcing against ANOTHER null run
# would misfire the very instrument the band was derived from. Output is
# staged as `$OUT_DIR/aa_null_report.json` (a second
# copy of the merged report, INSIDE the pulled-back artifact directory --
# never under `ci/artifacts/gpu-perf-aa-null/` directly on the pod's own
# throwaway checkout, which `runpod_gpu_perf_ab.sh`'s own rsync step never
# reaches); a human still decides which run(s) get promoted to
# `ci/artifacts/gpu-perf-aa-null/` as the campaign's own committed evidence,
# the same convention `runpod_gpu_howwell.sh`'s own artifact pull follows.
#
# ## Exit codes (round-3 adversarial audit B2/B3's reconciled lattice —
# this table, the code sites below, `gpu_inference_ab.py`'s own exit-code
# doc, and gpu-perf-ab.yml's own step annotations must all agree; every
# exit site in this script cites which arm of this table it lands on)
#
#   0  -- a report was written and the merge's own status is GREEN (see
#         `gpu_inference_ab.py`'s own exit-code doc: recorded regardless of
#         the ratio's own value in NON-enforcing mode — under
#         `GPU_INFERENCE_AB_ENFORCE=1`, `0` additionally requires `mode ==
#         "ab"` AND the ratio landed INSIDE the pre-registered advisory
#         band; see arm `1`'s own three enforcement shapes below for the
#         cases that still exit `1` from this same GREEN status).
#   1  -- a REAL correctness-of-measurement refusal, ONLY ever raised once
#         `gpu_inference_ab.py` can CONFIRM the signal is real (round-3
#         adversarial audit B2/B3 correction: an earlier version of this
#         table claimed this arm was "always the PR's own problem, never
#         the parent's" -- that overclaimed a confirmation this script does
#         not always have): an identity mismatch between two otherwise-
#         comparable legs (status INVALID), a malformed measurement on an
#         otherwise identity-clean leg set (status INVALID_MEASUREMENT), the
#         four legs' RECORDED start order not verifying as A,B,B,A with
#         every timestamp actually PARSED (status INVALID --
#         `verify_recorded_order`; an UNPARSEABLE timestamp is a SEPARATE,
#         neutral 75 case, never this one), a `b`-role leg (`b1`/`b2`) that
#         RAN but did not produce an `OK` report AND the producer's own
#         `mode` marker (see below) confirms `ab` (a `b`-role leg is ALSO
#         parent-sha under `--aa-null` -- no PR exists to blame there), a
#         binary whose own `provenance` does not match the clone it was
#         supposedly built from, or the PR/comparison clone's build FAILING
#         (outside `--aa-null` mode — see that mode's own exception below,
#         where the SAME build-failure classification already correctly
#         attributes it to the parent-shaped bucket instead). A report is
#         still WRITTEN on every one of these, never an uncaught crash. PLUS
#         (issue #335's final unit), under `GPU_INFERENCE_AB_ENFORCE=1` ONLY:
#         THREE further, entirely different shapes — in every one, `status`
#         stays GREEN (the four legs' premises really did agree; NONE of
#         these are a correctness refusal), distinguishable from the four
#         correctness shapes above by `status` alone (stays `GREEN`, never
#         `INVALID`/`INVALID_MEASUREMENT`), never by the bare exit code:
#           - `combined_embed_p50_ratio` ABOVE the band's upper edge --
#             `enforce_verdict=PERF_REGRESSION` with its own
#             `perf_regression_reason`: a real, unambiguous slowdown signal.
#           - `combined_embed_p50_ratio` BELOW the band's lower edge --
#             `enforce_verdict=OUTSIDE_BAND_FAST` (a DELIBERATELY DIFFERENT
#             verdict, never folded into `PERF_REGRESSION` -- this band is a
#             NULL band, not a "faster is always fine" one) with its own
#             `outside_band_fast_reason`: AMBIGUOUS -- either a genuine
#             large improvement, or a `b`-role leg that silently
#             short-circuited/broke and finished suspiciously fast; this
#             script's own comparator never guesses which, only refuses and
#             names the ambiguity for a human to adjudicate.
#           - enforcement was requested but `mode != "ab"` (`"aa-null"` or
#             unconfirmed/`None`) -- `enforce_verdict=ENFORCE_INVALID_MODE`,
#             checked BEFORE the band is even consulted; this script's own
#             mutual-exclusion guard (below) already refuses the
#             `--aa-null`-plus-`GPU_INFERENCE_AB_ENFORCE=1` combination at
#             exit `2` before anything is even rented, so this arm fires
#             only for an UNCONFIRMED (absent/older-producer) `mode`.
#         NEVER raised when `GPU_INFERENCE_AB_ENFORCE` is unset/`0` (the
#         default) — see the "TWO MODES" section above.
#   2  -- a usage/infra error: bad arguments (INCLUDING
#         `GPU_INFERENCE_AB_AA_NULL=1` together with
#         `GPU_INFERENCE_AB_ENFORCE=1` — refused before any pod is
#         rented/built, see that section's own note), `nvidia-smi` itself
#         failing to run, a `git clone`/`checkout`/submodule-init failure,
#         an unshallow-fetch failure, HEAD or the computed merge-base not
#         resolving to a well-formed 40-hex sha.
#   75 -- neutral "nothing to compare safely right now", never a sign of a
#         code problem: the GPU was reported busy, the PARENT clone's build
#         failed (a broken baseline is not a code regression this A/B can
#         attribute to the PR), the `--aa-null` COMPARISON clone's build
#         failed (also parent-sha under that mode, same bucket as the
#         parent), the `origin/main`-tracking-ref refresh fetch failed
#         (`gpu_inference_ab_git.sh`'s own doc — ADVISORY, this script logs
#         it and continues to the real gate, the merge-base call, per
#         round-2 adversarial audit F2), HEAD already equals origin/main's
#         merge-base (no PR-side commits at all), both binaries report the
#         SAME `build_sha` outside `--aa-null` mode, a parent leg is missing
#         one or more declared identity fields entirely (predates issue
#         #335's own identity contract — status INCOMPLETE_IDENTITY), one or
#         more legs' RECORDED start timestamps could not be read or did not
#         parse (status INCOMPLETE_ORDER, round-3 adversarial audit B3 — a
#         missing/unparseable timestamp is NOT itself proof the order was
#         violated), or fewer than all four legs produced an `OK` report and
#         the CONFIRMED-real b-role-FAIL-in-`ab`-mode precondition above
#         does not hold (status INCOMPLETE — a `MISSING`/`DRY_RUN` leg of
#         EITHER role, a `b`-role FAIL under `--aa-null`, or an unconfirmed
#         `mode`: round-3 adversarial audit B2).
#
# ## RESIDUALS (round-3 adversarial audit freeze rule: this wave is EXACTLY
# three fixes, B1/B2/B3 above; every OTHER advisory the audit raised is
# acknowledged here, verbatim by name, and DELIBERATELY left untouched)
#
#   - wall-clock ties: `verify_recorded_order` treats an EQUAL recorded
#     timestamp between adjacent legs as non-decreasing (not a violation)
#     -- never investigated whether two legs finishing within the SAME
#     nanosecond-epoch tick is itself a signal worth its own check.
#   - df fail-open: the driver's own pre-flight disk-space check
#     (`runpod_gpu_perf_ab.sh`) logs a warning and CONTINUES when `df -BG /`
#     produces unparseable output, rather than refusing -- a broken `df`
#     parse silently skips the safety check instead of failing closed.
#   - RP_DISK_GB prose: sized from `docs/maintainer/dev-gpu.md`'s own
#     "3+ trees -> RP_DISK_GB=70+" guidance, an approximation for a
#     DIFFERENT (seed-and-clone) build substrate this producer does not
#     actually use, never a bespoke measurement of THIS workload's own
#     footprint.
#   - sourced-guard latency: the `[[ "${BASH_SOURCE[0]}" == "${0}" ]]`
#     "am I sourced or executed" idiom (`gpu_inference_ab_git.sh`,
#     `runpod_clone_checkout.sh`) was verified empirically for the ONE
#     invocation shape this repo actually uses (`bash -s` reading from
#     stdin) -- not exhaustively re-verified against every other way bash
#     can source vs. execute a file.
#   - caller's 75-continue branch: `gpu_inference_ab_ensure_history_for_merge_base`
#     returning 75 makes THIS script log a warning and continue to the real
#     gate (the `git merge-base` call, round-2 adversarial audit F2) rather
#     than exit immediately -- no further hardening of that continuation
#     path (e.g. re-verifying `origin/main`'s own freshness before the
#     merge-base call) was attempted beyond the F2 fix itself.
#
# Env vars:
#   GPU_INFERENCE_AB_WORK_DIR       where clone-a/clone-b + their own
#                                   CARGO_TARGET_DIRs live (default a
#                                   sibling of this checkout,
#                                   "../gpu-perf-ab-<UTC timestamp>").
#   GPU_INFERENCE_AB_OUT_DIR        where the merged report + raw legs land
#                                   (default "<repo>/.gpu-inference-ab-report/
#                                   <UTC timestamp>").
#   GPU_INFERENCE_AB_AA_NULL=1      the D6 instrument (see above).
#   GPU_INFERENCE_AB_ENFORCE=1      opt THIS invocation into the enforcement
#                                   flip (issue #335's final unit, see the
#                                   "TWO MODES" section above) -- REQUIRES
#                                   `mode == "ab"` (mutually exclusive with
#                                   `GPU_INFERENCE_AB_AA_NULL=1`, refused at
#                                   exit 2 before anything is rented, see
#                                   that env var's own note); a GREEN-premise
#                                   run whose ratio falls outside the
#                                   pre-registered advisory band on EITHER
#                                   side refuses (exit 1, status stays GREEN,
#                                   `enforce_verdict` is
#                                   `PERF_REGRESSION` above the upper edge or
#                                   `OUTSIDE_BAND_FAST` below the lower one
#                                   -- direction-honest, never one name for
#                                   both). Default unset (0): non-enforcing,
#                                   recording-only, this producer's
#                                   longstanding default posture.
#   GPU_INFERENCE_AB_SKIP_GPU_CHECK=1  skip the nvidia-smi idle check
#                                   (CPU/dry-run smoke test only).
#   GPU_INFERENCE_AB_DRY_RUN=1      print every command this script would
#                                   run instead of executing it; writes
#                                   `{"tool":"dry-run",...}` stub files per
#                                   leg so the merge stage still runs
#                                   end-to-end against real (if
#                                   fabricated-empty) files. Never clones,
#                                   never builds, never touches the GPU or
#                                   the network; never claims a real number.
#   RUNPOD_POD_ID                  (read, never set by this script) THIS
#                                   pod's own identity, when the calling
#                                   environment provides it (RunPod's own
#                                   convention) -- recorded verbatim into
#                                   every leg's own `provenance.pod_id`
#                                   (falls back to `$(hostname)` when unset,
#                                   e.g. a by-hand invocation off RunPod).
#                                   The structural fix for the
#                                   concurrent-invocations-sharing-one-pod
#                                   contamination class
#                                   `ci/artifacts/gpu-perf-aa-null/README.md`'s
#                                   own "Disclosure" section reconstructs by
#                                   hand today -- a future committed artifact
#                                   carries this directly.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../../.." && pwd)"

GPU_INFERENCE_AB_DRY_RUN="${GPU_INFERENCE_AB_DRY_RUN:-0}"
GPU_INFERENCE_AB_AA_NULL="${GPU_INFERENCE_AB_AA_NULL:-0}"
GPU_INFERENCE_AB_ENFORCE="${GPU_INFERENCE_AB_ENFORCE:-0}"
GPU_INFERENCE_AB_SKIP_GPU_CHECK="${GPU_INFERENCE_AB_SKIP_GPU_CHECK:-0}"

# round-4 delta-audit F4: --aa-null and enforcement are mutually exclusive
# -- refused HERE, before anything is rented/cloned/built (strictly cheaper
# than discovering this at the comparator, `gpu_inference_ab.py::build_report`'s
# own ENFORCE_INVALID_MODE arm, which ALSO refuses this combination as a
# defense in depth, never relied on as the ONLY refusal site). No PR exists
# under --aa-null (both b-role legs are ALSO parent-sha clones) -- enforcing
# a null-vs-null run against the band would misfire the very instrument the
# band was derived from.
if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ] && [ "$GPU_INFERENCE_AB_ENFORCE" = "1" ]; then
  echo "::error::GPU_INFERENCE_AB_AA_NULL=1 and GPU_INFERENCE_AB_ENFORCE=1 cannot both be set -- --aa-null measures a null (parent-vs-parent) run with no PR to enforce against; refusing before any pod is rented/built (exit 2, a usage error, never a correctness-of-measurement or capacity one)." >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
# Retention (round-1 adversarial audit advisory, documented rather than
# auto-cleaned): $WORK_DIR (two full clones + two `cargo build` target
# dirs, easily multiple GB) is DELIBERATELY left on disk when this script
# exits, success or failure alike -- a fresh, timestamped directory every
# invocation (never overwritten), so a failed leg's own binary/clone stays
# inspectable for post-mortem debugging without re-cloning/re-building. The
# actual cleanup mechanism is the RENTED POD'S OWN teardown: this script
# always runs inside `runpod_gpu_perf_ab.sh`'s ephemeral RunPod pod (torn
# down by `runpod_lib.sh`'s own EXIT trap the moment that OUTER driver
# exits), so $WORK_DIR's lifetime is bounded by the pod's, never by a
# growing accumulation across runs on a long-lived host. A direct,
# by-hand invocation of this script on a persistent box is the one case
# this reasoning does not cover -- an operator running it that way is
# expected to set GPU_INFERENCE_AB_WORK_DIR explicitly and clean up
# afterward, the same "operator/perf-lane tool" posture stacked_sweep.sh's
# own $CARGO_TARGET_DIR reuse already takes.
WORK_DIR="${GPU_INFERENCE_AB_WORK_DIR:-$(dirname "$REPO_ROOT")/gpu-perf-ab-$TS}"
OUT_DIR="${GPU_INFERENCE_AB_OUT_DIR:-$REPO_ROOT/.gpu-inference-ab-report/$TS}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"

# round-3 adversarial audit B2: the comparator has NO other way to know
# whether this run is the normal parent-vs-PR A/B or the --aa-null
# empirical-null instrument (both b-role legs are ALSO parent-sha clones
# under aa-null -- a b-role RUNTIME failure there carries no "the PR's own
# problem" signal at all, since there is no PR leg in play) -- write it
# ONE time, here, before any leg runs, so `gpu_inference_ab.py`'s own
# `build_report` can route a b-role leg failure correctly instead of
# blaming a nonexistent PR. DRY_RUN takes precedence over AA_NULL in this
# marker (a dry run's own aa_null flag changes no real measurement; every
# leg is a DRY_RUN stub either way, so the comparator's MISSING/DRY_RUN
# routing already handles it regardless of which value is recorded here).
if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
  printf 'dry-run' > "$RAW_DIR/mode"
elif [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
  printf 'aa-null' > "$RAW_DIR/mode"
else
  printf 'ab' > "$RAW_DIR/mode"
fi

# issue #335's final unit (the enforcement flip): the SAME file-based
# state-passing convention as `mode` above, written ONE time, before any leg
# runs, so `gpu_inference_ab.py::load_enforce` can read it -- never an env
# var threaded directly into that module (see this module's own doc).
if [ "$GPU_INFERENCE_AB_ENFORCE" = "1" ]; then
  printf '1' > "$RAW_DIR/enforce"
else
  printf '0' > "$RAW_DIR/enforce"
fi

# round-4 delta-audit F3(d): pod identity, the structural fix for the
# concurrent-invocations-sharing-one-pod contamination class
# `ci/artifacts/gpu-perf-aa-null/README.md`'s own "Disclosure" section
# reconstructs by hand today -- written ONE time, before any leg runs, the
# SAME file-based state-passing convention `mode`/`enforce` above already
# use, so `gpu_inference_ab.py::load_pod_id` can fold it into every OK
# leg's own `provenance.pod_id`. `RUNPOD_POD_ID` is the identity RunPod's
# own environment provides on a rented pod; `$(hostname)` is the fallback
# for a by-hand invocation off RunPod (still a real, if less specific,
# identity -- never a fabricated placeholder).
printf '%s' "${RUNPOD_POD_ID:-$(hostname)}" > "$RAW_DIR/pod_id"

CLONE_A="$WORK_DIR/clone-a"
CLONE_B="$WORK_DIR/clone-b"
TARGET_A="$WORK_DIR/target-a"
TARGET_B="$WORK_DIR/target-b"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

# --- GPU must be idle before the first build even starts (stacked_sweep.sh
# precedent) -- a busy GPU makes every timing this script would eventually
# produce meaningless before a single clone is even made.
#
# Exit-lattice split (round-1 adversarial audit B3): `nvidia-smi` itself
# FAILING to run (missing binary, driver problem) is an infra/usage error —
# exit 2, this box cannot even be asked the question. The query SUCCEEDING
# and reporting busy compute processes is NEUTRAL — exit 75, "nothing to
# compare safely right now", the SAME bucket a parent build failure or a
# merge-base==HEAD no-op falls into (never exit 1, which this script
# reserves for a REAL correctness-of-measurement refusal). ---
if [ "$GPU_INFERENCE_AB_SKIP_GPU_CHECK" != "1" ] && [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "::error::'nvidia-smi --query-compute-apps' failed (rc=$RC): $BUSY -- refusing to proceed without a confirmed-idle GPU. Set GPU_INFERENCE_AB_SKIP_GPU_CHECK=1 only for a CPU/dry-run smoke test." >&2
    exit 2
  fi
  if [ -n "$BUSY" ]; then
    echo "::warning::GPU is not idle -- nvidia-smi reports compute processes (neutral, nothing to compare safely right now):" >&2
    echo "$BUSY" >&2
    exit 75
  fi
fi

# --- ensure this checkout carries enough history for a real merge-base
# (round-1 adversarial audit B2) -- see gpu_inference_ab_git.sh's own doc
# for the exact bug this closes and this function's own exit-code
# contract (0 / 2 / 75).
#
# round-2 adversarial audit F2 (the LIBRARY's own design is correct, the
# CALLER was wrong): a `75` here is ADVISORY, not a gate -- it means "the
# origin/main REFRESH fetch failed", which is meaningless on its own if a
# PRIOR clone step already populated a usable origin/main (the normal case
# after runpod_gpu_perf_ab.sh's own full, non-single-branch initial
# clone). The REAL gate is the `git merge-base origin/main HEAD` call
# below: log the `75` and CONTINUE to it, rather than exiting on the
# history helper's own advisory result -- if origin/main is genuinely
# unusable, THAT call surfaces its own real error (exit 2, see below). A
# `2` here, in contrast, IS a hard stop: the unshallow fetch itself failed,
# meaning this repo can never compute a real merge-base at all, with no
# fallback for the merge-base call below to succeed through. ---
# shellcheck source=ci/scripts/perf/gpu_inference_ab_git.sh
source "$DIR/gpu_inference_ab_git.sh"
gpu_inference_ab_ensure_history_for_merge_base "$REPO_ROOT" "$GPU_INFERENCE_AB_DRY_RUN"
HISTORY_RC=$?
if [ "$HISTORY_RC" -eq 2 ]; then
  exit 2
elif [ "$HISTORY_RC" -eq 75 ]; then
  echo "::warning::gpu_inference_ab_ensure_history_for_merge_base returned 75 (advisory) -- continuing to the real gate, the merge-base call itself, which will surface its OWN error if origin/main is genuinely unusable." >&2
fi

SHA_RE='^[0-9a-fA-F]{40}$'
PR_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>&1)" || { echo "::error::'git rev-parse HEAD' failed: $PR_SHA" >&2; exit 2; }
if ! [[ "$PR_SHA" =~ $SHA_RE ]]; then
  echo "::error::HEAD did not resolve to a 40-hex commit ('$PR_SHA') -- refusing" >&2
  exit 2
fi
if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
  # round-1 adversarial audit B2: a well-formed 40-hex PLACEHOLDER -- dry-run
  # must walk the SAME validation PR_SHA/PARENT_SHA both go through below,
  # never step over it with a value that would fail the real check.
  PARENT_SHA="a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0"
else
  PARENT_SHA="$(git -C "$REPO_ROOT" merge-base origin/main HEAD 2>&1)" \
    || { echo "::error::'git merge-base origin/main HEAD' failed: $PARENT_SHA -- is origin/main fetched?" >&2; exit 2; }
fi
if ! [[ "$PARENT_SHA" =~ $SHA_RE ]]; then
  echo "::error::merge-base did not resolve to a 40-hex commit ('$PARENT_SHA') -- refusing" >&2
  exit 2
fi

if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
  # The D6 instrument: clone-b (the SAME second clone every invocation
  # makes, never an additional third one) checks out the SAME parent sha,
  # never the PR sha -- see this script's own header.
  B_SHA="$PARENT_SHA"
else
  B_SHA="$PR_SHA"
  if [ "$PARENT_SHA" = "$PR_SHA" ]; then
    echo "::warning::HEAD IS origin/main's merge-base (no PR-side commit) -- nothing to compare; neutral exit 75." >&2
    exit 75
  fi
fi

mkdir -p "$WORK_DIR"

# --- TWO SIMULTANEOUSLY-RESIDENT clones, checked out BEFORE any build ---
clone_and_checkout() {
  local clone="$1" sha="$2" label="$3"
  # round-2 adversarial audit F6 (round-3 adversarial audit B1 correction):
  # `--filter=blob:none` -- the SAME pod-clone idiom runpod_lib.sh:1505
  # already uses -- skips every HISTORICAL blob this workload never touches
  # (no git log -p, no diffing against history; only the ONE checked-out
  # tree's own files are ever read), a real disk-footprint saving for two
  # full source trees. `file://$REPO_ROOT` is REQUIRED for the filter to
  # even be ATTEMPTED (a bare local path silently ignores `--filter`
  # outright, discovered empirically). This clone's SOURCE, `$REPO_ROOT`,
  # is ITSELF a partial (blobless) clone by the time this runs
  # (`runpod_gpu_perf_ab.sh`'s own outer clone, made via
  # `runpod_clone_checkout.sh`) -- an earlier version of this comment
  # claimed an unsupported filter request "never a hard failure" (silent
  # fallback to a full clone); that claim was FALSE for exactly this
  # composition (round-3 adversarial audit B1, the auditor's own
  # reproduction): a source that is ITSELF partial cannot silently
  # fall back to serving a FULL clone (it does not have every blob to
  # serve), so an inner filtered clone against an unfiltered-serving
  # source FAILS hard (fatal, exit 128) rather than degrading gracefully.
  # The REAL contract this depends on: the source repo's own
  # `uploadpack.allowFilter` must be `true` so it can actually SERVE a
  # partial-clone request rather than attempt (and fail) a full one --
  # `runpod_clone_checkout.sh`'s own outer clone sets this immediately
  # after cloning, which is what makes THIS inner clone sound. This
  # function does not set it again (the outer lib's own responsibility,
  # asserted once, not re-asserted at every inner clone site).
  run_cmd git clone --no-hardlinks --quiet --filter=blob:none "file://$REPO_ROOT" "$clone" \
    || { echo "::error::cloning $label ($REPO_ROOT -> $clone) failed" >&2; return 1; }
  run_cmd git -C "$clone" checkout --quiet --detach "$sha" \
    || { echo "::error::checking out $label sha $sha in $clone failed" >&2; return 1; }
  # jammi-kernels/build.rs panics loudly the moment a flash-attn build
  # reaches it with no CUTLASS submodule checked out (runpod_gpu_howwell.sh's
  # own note) -- a local clone of a checkout that DID init the submodule
  # still carries no submodule content of its own until this runs.
  run_cmd git -C "$clone" submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
    || { echo "::error::submodule init for $label ($clone) failed" >&2; return 1; }
  return 0
}

clone_and_checkout "$CLONE_A" "$PARENT_SHA" "a (parent)" || exit 2
clone_and_checkout "$CLONE_B" "$B_SHA" "b ($( [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ] && echo 'aa-null parent' || echo pr ))" || exit 2

# --- both binaries built FULLY, up front, before any measurement ---
build_clone() {
  local clone="$1" target="$2"
  CARGO_TARGET_DIR="$target" run_cmd cargo build --release -p jammi-bench --features cuda --manifest-path "$clone/Cargo.toml"
}

if ! build_clone "$CLONE_A" "$TARGET_A"; then
  echo "::warning::parent clone build FAILED -- a broken baseline is not a code regression this A/B can attribute to the PR; neutral exit 75." >&2
  exit 75
fi
if ! build_clone "$CLONE_B" "$TARGET_B"; then
  if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ]; then
    # --aa-null: clone-b is ALSO a parent-sha clone (see this script's own
    # header) -- its build failing is the SAME "broken baseline" case as
    # clone-a's, never a "PR's own problem" (there is no PR leg in play
    # under this mode at all).
    echo "::warning::--aa-null comparison clone (a second independent parent-sha build) FAILED -- same bucket as a parent build failure; neutral exit 75." >&2
    exit 75
  fi
  # round-1 adversarial audit B3: a PR-side build failure is the PR's OWN
  # problem -- a real correctness-of-measurement refusal (exit 1), distinct
  # from the parent-broke-the-baseline case above (exit 75). The PR simply
  # not compiling is itself a genuine signal this producer surfaces rather
  # than swallowing into the same neutral bucket a pre-existing parent
  # breakage falls into.
  echo "::error::PR clone build FAILED -- this is the PR's own problem, not a neutral 'nothing to compare' state; exit 1." >&2
  exit 1
fi

BIN_A="$TARGET_A/release/jammi-bench"
BIN_B="$TARGET_B/release/jammi-bench"

# --- per-binary provenance cross-check (C5.1 shape, cf. encode_ab.sh) ---
check_provenance() {
  local bin="$1" clone="$2" label="$3"
  local expect_sha prov_json prov_sha
  expect_sha="$(git -C "$clone" rev-parse HEAD)"
  prov_json="$("$bin" provenance 2>&1)" || { echo "::error::'$bin provenance' ($label) failed: $prov_json" >&2; return 1; }
  prov_sha="$(printf '%s' "$prov_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["build_sha"])' 2>&1)" \
    || { echo "::error::could not parse build_sha from '$bin provenance' ($label): $prov_json" >&2; return 1; }
  if [ -z "$prov_sha" ] || [ "$prov_sha" != "$expect_sha" ]; then
    echo "::error::$label binary's provenance build_sha=$prov_sha but its own clone's HEAD=$expect_sha -- refusing before any leg." >&2
    return 1
  fi
  printf '%s' "$prov_sha"
}

if [ "$GPU_INFERENCE_AB_DRY_RUN" != "1" ]; then
  A_PROV_SHA="$(check_provenance "$BIN_A" "$CLONE_A" a)" || exit 1
  B_PROV_SHA="$(check_provenance "$BIN_B" "$CLONE_B" b)" || exit 1
  if [ "$GPU_INFERENCE_AB_AA_NULL" != "1" ] && [ "$A_PROV_SHA" = "$B_PROV_SHA" ]; then
    echo "::warning::both binaries report the SAME build_sha ($A_PROV_SHA) -- nothing to compare outside --aa-null mode; neutral exit 75." >&2
    exit 75
  fi
else
  A_PROV_SHA="$PARENT_SHA"
  B_PROV_SHA="$B_SHA"
fi

# --- one leg. NEVER aborts the sweep -- a leg failure is recorded as this
# leg's own outcome (its .exit file + stderr), same discipline
# stacked_sweep.sh/encode_ab.sh's own run_leg already follow. ---
run_leg() {
  local name="$1" bin="$2"
  local out_file="$RAW_DIR/${name}.json"
  local err_file="$RAW_DIR/${name}.stderr"
  local exit_file="$RAW_DIR/${name}.exit"
  local started_at_file="$RAW_DIR/${name}.started_at"

  # round-2 adversarial audit F3 (order binding): record a monotonic-for-
  # practical-purposes start timestamp (nanosecond epoch) BEFORE invoking
  # this leg's binary, in EVERY mode including --dry-run -- the ONE piece
  # of evidence `gpu_inference_ab.py`'s own comparator uses to MACHINE-
  # CHECK that the four legs actually ran in the A,B,B,A order the whole
  # drift-cancellation rationale depends on, rather than merely trusting
  # that this script's own source calls run_leg in that sequence.
  date +%s%N > "$started_at_file"

  printf -- '--- %s: ' "$name"
  printf '%q ' "$bin" gpu-inference-scale
  printf '\n'

  if [ "$GPU_INFERENCE_AB_DRY_RUN" = "1" ]; then
    printf '{"tool":"dry-run","ab_dry_run":true,"leg":"%s"}\n' "$name" > "$out_file"
    : > "$err_file"
    echo "0" > "$exit_file"
    return 0
  fi

  local rc=0
  "$bin" gpu-inference-scale > "$out_file" 2> "$err_file" || rc=$?
  echo "$rc" > "$exit_file"
  if [ "$rc" -ne 0 ]; then
    echo "::warning::${name} FAILED (exit ${rc}) -- recorded as this leg's own outcome; run continues." >&2
    tail -n 5 "$err_file" 2>/dev/null || true
  fi
  return 0
}

# --- Order-balanced A, B, B, A -- NEVER A, A, B, B (see this script's own
# header for why the order itself is load-bearing). ---
run_leg a1 "$BIN_A"
run_leg b1 "$BIN_B"
run_leg b2 "$BIN_B"
run_leg a2 "$BIN_A"

# --- merge: gpu_inference_ab.py's own leg-premise refusal + primary-endpoint
# ratio + advisory classification. ---
python3 "$DIR/gpu_inference_ab.py" "$RAW_DIR" "$OUT_DIR" "$A_PROV_SHA" "$B_PROV_SHA"
MERGE_RC=$?

# --- --aa-null: stage a second, clearly-named copy of the merged artifact
# for eventual commit under ci/artifacts/gpu-perf-aa-null/ (D6's own
# evidence path) -- a human still decides which run(s) get committed, the
# same convention runpod_gpu_howwell.sh's own artifact pull follows.
#
# Staged INSIDE $OUT_DIR (round-1 adversarial audit advisory), never under
# $REPO_ROOT/ci/artifacts/ directly: $REPO_ROOT here is the ON-POD checkout
# (e.g. /root/jammi-ai) that invoked this script, NOT the caller's own
# local repo -- a file written under $REPO_ROOT/ci/artifacts/ would sit
# inside the pod's own throwaway clone, OUTSIDE the ONE directory
# ($OUT_DIR, ".gpu-inference-ab-report/<ts>/") `runpod_gpu_perf_ab.sh`'s
# own rsync step actually pulls back before the pod is torn down -- it
# would silently never leave the pod at all. $OUT_DIR is exactly the tree
# that DOES get pulled, so staging here is what actually makes this
# artifact retrievable; an operator who wants to commit it under
# ci/artifacts/gpu-perf-aa-null/ for real (once enough runs exist to derive
# a real band, D6) copies it there from the pulled artifact directory. ---
if [ "$GPU_INFERENCE_AB_AA_NULL" = "1" ] && [ -f "$OUT_DIR/gpu_inference_ab_report.json" ]; then
  cp "$OUT_DIR/gpu_inference_ab_report.json" "$OUT_DIR/aa_null_report.json"
  echo "=== --aa-null artifact staged inside the pulled report dir: $OUT_DIR/aa_null_report.json (promote to ci/artifacts/gpu-perf-aa-null/ by hand once it is real evidence) ===" >&2
fi

echo
echo "=== clones: a=$CLONE_A ($PARENT_SHA) b=$CLONE_B ($B_SHA) ==="
echo "=== raw legs + merged report: ${OUT_DIR} ==="
exit "$MERGE_RC"
