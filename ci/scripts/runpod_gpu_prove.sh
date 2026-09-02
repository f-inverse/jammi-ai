#!/usr/bin/env bash
# GPU prove-lane: build jammi from source on a real GPU and run the gated GPU
# suites — the served client/server proof (grpc_embedding_gpu) and the
# engine-core correctness suite (gpu_capability). Doubles as the #277 regression
# gate: builds candle's kernels at THIS LEG'S NATIVE CUDA_COMPUTE_CAP (see
# NATIVE_COMPUTE_CAP below — overrides the image's baked cap, issue #434) and
# runs them on a real device. Shared deploy/run/teardown lives in runpod_lib.sh.
#
# GPU_PROVE_ARCH selects WHICH shipped CUDA arch (crates/jammi-kernels/build.rs
# GENCODE_ARCHES: sm_80/sm_86/sm_89/sm_90 today) this lane proves; default is
# today's A100 (sm_80) behavior. The sm_XX -> `rp_deploy_arch` key mapping
# below is the ONE place that translation lives; `rp_deploy_arch` itself
# (runpod_lib.sh) is the ONE place the arch key -> actual RunPod GPU-type-id
# candidate list lives — this script never hand-types a GPU-type-id string.
#
# esc-081 (proof surface == shipped surface): every CUDA-bearing cargo
# invocation below carries a LITERAL `--features` tuple -- never a shell
# variable -- immediately preceded by a `PROVE_TUPLE crate=<c> kind=<k>
# features=<literal>` echo, so `check_execution_surface_reachability.py`'s
# `is_gated`/`extract_tuples_from_line` (which operate on the SOURCE TEXT,
# never evaluate a variable) register the real tuple, and
# `check_flash_attn_closure.py` can assert SET EQUALITY between
# `ci/release-feature-manifest.json`'s `prove_lane.crates.<c>.kinds`
# declaration and these invocations (`ci/scripts/prove_surface.py`'s shared
# canonicalization computes the expected literal for each declared pair).
# The manifest is still READ at runtime (see the capability-surface-build
# group) as a TRIPWIRE ONLY -- comparing manifest-derived features against
# the literal below and failing loud on drift -- never used to BUILD the
# `--features` argument itself.
#
# esc-080/esc-082/esc-083: `rp_run_remote_watched` (runpod_lib.sh) layers an
# inactivity watchdog on top of the ssh budget; `PROVE_GROUPS` below names
# the six gating groups this driver's own pass/fail rule reads
# `PROVE_GROUP_RC` markers for (`device` and `bench` are NOT members -- see
# that array's own comment). The `jammi-kernels` clippy lane that used to run
# here (esc-051, esc-059) has moved entirely to `ci.yml`'s own hermetic
# `Clippy jammi-kernels --features flash-attn --all-targets` step (nvcc, no
# GPU needed) -- see `check_lint_surface_closure.py`'s own module doc.
#
# Exit 0 = suites passed (see the driver rule below for what "passed" means
# once the bench-cut exception is folded in); 75 = no capacity for the
# requested arch (neutral skip); 76 = watchdog inactivity kill with a
# gating group unresolved; 124 = budget cut with a gating group unresolved.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This lane's own pods never need 8h — the remote job is capped by RP_TIMEOUT
# (100m by default, see RP_TIMEOUT below), so a tighter deadline bounds the
# damage of an orphan.
RP_TTL_HOURS="${RP_TTL_HOURS:-3}"
# shellcheck source=ci/scripts/runpod_lib.sh
source "$DIR/runpod_lib.sh"

# esc-080: the prove lane's own budget, exported HERE ONLY (never in
# runpod_lib.sh, whose own `${RP_TIMEOUT:-3000}` default stays 3000s for
# every OTHER caller -- gpu-dev.sh, runpod_gpu_perf_ab.sh). Two-term backstop
# (esc-083, lead-amended control): `RP_TIMEOUT >= 1.5 * max healthy wall` AND
# `RP_TIMEOUT >= max healthy wall + 3 * RP_INACTIVITY` -- the inactivity
# watchdog is the hang detector, so the backstop only needs to outlast the
# slowest healthy leg plus a late-detected hang, not a from-scratch multiple
# of it. `check_gpu_prove_timings.py`'s R3 re-checks this rule against every
# committed healthy artifact on every run. Platform ceiling: with 3 retry
# attempts (`_gpu-prove-gate.yml`), `3 * 80m deploy-worst-case + 2 * 5m
# overhead + RP_TIMEOUT/60 <= 360m` bounds RP_TIMEOUT well above 6000s, so
# this value is not yet constrained by the job-timeout ceiling.
export RP_TIMEOUT="${RP_TIMEOUT:-6000}"

GIT_REPO="${GIT_REPO:-https://github.com/${GITHUB_REPOSITORY:-f-inverse/jammi-ai}.git}"
GIT_REF="${GIT_REF:-${GITHUB_SHA:-main}}"

# sm_XX (the GENCODE_ARCHES / check_gpu_parity_matrix.py silicon-axis naming)
# -> the `rp_deploy_arch` candidate-list key (runpod_lib.sh), and -> the bare
# numeric NATIVE_COMPUTE_CAP this leg's device actually is. NATIVE_COMPUTE_CAP
# overrides the CI image's baked CUDA_COMPUTE_CAP (.docker/ci-cuda.Dockerfile)
# in the remote build env below: candle-kernels 0.11 builds the quantized
# fast-path kernels as single-arch SASS (no PTX) from that env var, so every
# leg building at the image's one baked cap produces kernels that silently
# cannot launch on the other legs' devices (issue #434). Comment on each line
# names the SASS target the leg proves.
GPU_PROVE_ARCH="${GPU_PROVE_ARCH:-sm_80}"
case "$GPU_PROVE_ARCH" in
  sm_80) RP_DEPLOY_ARCH=a100 NATIVE_COMPUTE_CAP=80 ;; # Ampere floor — proves sm_80, #277.
  sm_86) RP_DEPLOY_ARCH=a40  NATIVE_COMPUTE_CAP=86 ;; # Ampere workstation class — proves sm_86.
  sm_89) RP_DEPLOY_ARCH=l4_l40s NATIVE_COMPUTE_CAP=89 ;; # Ada — proves sm_89, fp8 #308. L4 first (canonical commodity inference card, ~half L40S rental); L40S is a capacity-only fallback — same sm_89 SASS, identical correctness proof.
  sm_90) RP_DEPLOY_ARCH=h100 NATIVE_COMPUTE_CAP=90 ;; # Hopper — proves sm_90.
  *)
    echo "::error::unknown GPU_PROVE_ARCH '${GPU_PROVE_ARCH}' (want: sm_80|sm_86|sm_89|sm_90)"
    exit 2
    ;;
esac

# esc-081: the six PROOF groups this leg's driver rule below gates on --
# `::group::` names in the heredoc below, verbatim. `device` and `bench` are
# NOT members: `device` carries no proof (it is the compute_cap tripwire),
# and `bench` is deliberately NON-GATING (esc-082) -- a cut/hang inside
# `bench` with every group below still `rc=0` passes the leg (see
# `rp_prove_verdict` below). A group added to the script without a
# corresponding entry here (or vice versa) is caught by
# `test_gpu_prove_lane.sh`'s own closure fixture. Declared BEFORE the
# sourced-execution guard below so `test_gpu_prove_lane.sh` can `source`
# this file (RUNPOD_API_KEY/GPU_PROVE_ARCH pre-set, no network/ssh calls
# made merely by sourcing) and see it, exactly as it sees `rp_prove_verdict`.
PROVE_GROUPS=(capability-surface-build capability-surface-proof served-client-server-proof engine-core-sweep kernels-default kernels-cuda)

# esc-080/esc-082/esc-083 driver rule (D3/D4): decide a leg's real verdict
# from `rp_run_remote_watched`'s own return status (`$1`, taken after ITS OWN
# final drain to EOF) plus the `PROVE_GROUP_RC`/`PROVE_EXIT` markers actually
# landed in the log file (`$2`). A plain function (not inlined after the
# heredoc below) so `test_gpu_prove_lane.sh` can drive it directly, against
# hand-built log fixtures, without renting a pod or even sourcing this
# file's own top-level deploy flow.
#
# Precedence (never the reverse): a normal (in-suite) exit's status is
# returned VERBATIM, whatever it is; a ZERO status may be turned into a
# FAILURE by a `PROVE_GROUPS` member missing its marker or reporting
# rc != 0; a cut/hang with NO `PROVE_EXIT=` line (a genuine budget-cut or
# inactivity-kill, never an in-suite decision) applies the bench-cut
# exception: if every `PROVE_GROUPS` member already shows rc=0, the leg
# PASSES (0 + a `::warning::` naming the follow-up), else the cut/hang code
# (76/124) is returned. Returns the final verdict via `return` (0-255).
rp_prove_verdict() {
  local raw_rc="$1" log="$2"
  declare -A grc_map=()
  local line
  while IFS= read -r line; do
    case "$line" in
      *"PROVE_GROUP_RC "*"name="*"rc="*)
        local rest="${line#*PROVE_GROUP_RC }"
        local gname="${rest#name=}"; gname="${gname%% *}"
        local gval="${rest##*rc=}"; gval="${gval%% *}"
        grc_map["$gname"]="$gval"
        ;;
    esac
  done < "$log"

  local has_prove_exit=0
  if grep -q '^PROVE_EXIT=' "$log" 2>/dev/null; then
    has_prove_exit=1
  fi

  local all_proof_pass=1
  local missing_or_failed=()
  local g v
  for g in "${PROVE_GROUPS[@]}"; do
    v="${grc_map[$g]:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]] || [ "$v" -ne 0 ]; then
      all_proof_pass=0
      missing_or_failed+=("${g}=${v:-<missing>}")
    fi
  done

  local rc="$raw_rc"
  if [ "$raw_rc" -eq 0 ]; then
    # A zero status may be turned into a failure by the marker rule, never
    # the reverse — defends against a `PROVE_EXIT=0` that disagrees with its
    # own groups' markers (should never happen given the heredoc's own tail,
    # but a bare zero is never trusted blindly).
    if [ "$all_proof_pass" -ne 1 ]; then
      echo "::error::GPU prove: ssh exited 0 but PROVE_GROUPS member(s) missing or non-zero: ${missing_or_failed[*]:-<none>} — PROVE_EXIT disagrees with its own markers" >&2
      rc=1
    fi
  elif [ "$has_prove_exit" -eq 1 ]; then
    # In-suite exit (the remote reached its own final `echo PROVE_EXIT=…;
    # exit`) — returned VERBATIM, never relabeled, no matter the code.
    rc="$raw_rc"
  elif [ "$raw_rc" -eq 76 ] || [ "$raw_rc" -eq 124 ]; then
    # A genuine cut/hang (no PROVE_EXIT reached) — the bench-cut exception.
    if [ "$all_proof_pass" -eq 1 ]; then
      echo "::warning::GPU prove: bench cut/hung after every proof group already passed (raw rc=${raw_rc}) — produce the budget-cut artifact with ci/scripts/perf/gpu_prove_timings.py and add its R4 disposition" >&2
      rc=0
    else
      rc="$raw_rc"
    fi
  else
    rc="$raw_rc"
  fi
  return "$rc"
}

# Everything below runs only when this file is EXECUTED, never when it is
# `source`d (test_gpu_prove_lane.sh sources it to reach `rp_prove_verdict`/
# `rp_run_remote_watched`/`PROVE_GROUPS` without renting a pod or making any
# network call merely by sourcing).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then

# Sweep before renting anything. This workflow sets cancel-in-progress, so a
# superseded run is SIGKILLed and never runs its EXIT trap; the pod it had just
# rented is orphaned. Running the sweep here bounds any such orphan to the gap
# until the next prove run rather than "until the account empties" — which is
# what happened on 2026-07-24.
rp_sweep

rp_init
echo "=== provisioning a live ${RP_DEPLOY_ARCH} (${GPU_PROVE_ARCH}) ==="
rp_deploy_arch "$RP_DEPLOY_ARCH" || exit $?

echo "=== running GPU prove suites on ${RP_HOST}:${RP_PORT} ==="
LOG="$(mktemp)"
rp_run_remote_watched <<REMOTE | tee "$LOG"
export CARGO_TERM_COLOR=never
export CARGO_BUILD_RUSTC_WRAPPER=  # wrapper-off (ledger row 17: no cross-target-dir reuse, ~+33% wall on this image)
# Override the image's baked CUDA_COMPUTE_CAP with this leg's NATIVE arch.
# candle-kernels 0.11 builds the quantized fast-path as single-arch SASS (no
# PTX) from this var; leaving the baked cap in place would build every leg's
# fast kernels for the image's one arch, which silently cannot launch on the
# other legs' devices (issue #434). The load-time canary (crates/jammi-kernels)
# is the shipped-artifact guard for that failure mode; this makes each prove
# leg build and test its own native arch instead.
export CUDA_COMPUTE_CAP=${NATIVE_COMPUTE_CAP}
echo "::group::device"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv; echo "CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>}"; echo "::endgroup::"
# Hard assertion: nvidia-smi's reported compute_cap (e.g. "8.0") and the
# NATIVE_COMPUTE_CAP override above (e.g. "80") must name the SAME device --
# a mismatch here means the rented pod is not the arch this leg thinks it is,
# and every kernel built below would be silently wrong for it (issue #434's
# exact failure mode, one layer earlier). nvidia-smi's dotted form is
# normalized (dot stripped) before comparing against CUDA_COMPUTE_CAP's bare
# digit form.
compute_cap_raw="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')"
compute_cap_norm="\${compute_cap_raw//./}"
if [ "\${compute_cap_norm}" != "\${CUDA_COMPUTE_CAP:-}" ]; then
  echo "::error::compute_cap mismatch: nvidia-smi reports compute_cap=\${compute_cap_raw} (normalized \${compute_cap_norm}) but CUDA_COMPUTE_CAP=\${CUDA_COMPUTE_CAP:-<unset>} -- rented device does not match this leg's requested arch, refusing to build"
  exit 97
fi
cd /root && rm -rf jammi-ai
git clone --depth 1 -b "${GIT_REF}" "${GIT_REPO}" jammi-ai 2>&1 | tail -1
cd jammi-ai
echo "PROVE_SHA=\$(git rev-parse HEAD)"
rc=0
# The vendored FlashAttention-2 build (\`flash-attn\`, now in the manifest's
# cu12-tarball feature list read below) needs the CUTLASS submodule; a plain
# shallow \`git clone\` above does not fetch submodules.
git submodule update --init --depth 1 crates/jammi-kernels/third_party/cutlass \
  || { echo "::error::CUTLASS submodule init failed (network/remote unreachable?) — refusing to attempt the flash-attn build" >&2; exit 1; }

# capability-surface-build: builds the shipped cu12-tarball server binary and
# compile-checks the jammi-ai capability-surface test binary. The manifest
# read below is a TRIPWIRE ONLY (esc-081) -- it never feeds the literal
# \`--features\` arguments the two cargo invocations carry; a divergence
# between the manifest and the literals fails this group loud rather than
# silently building a narrower (or wider) surface than the manifest claims.
echo "::group::capability-surface-build"
grc=0
cu12_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
print(','.join(d['lanes']['cu12-tarball']['cargo_features']))
")"
if [ -z "\${cu12_features}" ]; then
  echo "::error::ci/release-feature-manifest.json produced an empty cu12-tarball cargo_features list" >&2
  grc=1
fi
ai_features="\$(python3 -c "
import json
d = json.load(open('ci/release-feature-manifest.json'))
lane = set(d['lanes']['cu12-tarball']['cargo_features'])
server_only = set(d['server_only_cargo_features']['features'])
print(','.join(sorted(lane - server_only)))
")"
if [ -z "\${ai_features}" ]; then
  echo "::error::deriving the jammi-ai-applicable feature subset from ci/release-feature-manifest.json produced an empty list" >&2
  grc=1
fi
echo "cu12-tarball cargo_features=\${cu12_features}"
echo "jammi-ai-applicable subset=\${ai_features}"
if [ "\${cu12_features}" != "cuda,flash-attn,jetstream-broker,storage-cloud" ]; then
  echo "::error::PROVE_SURFACE_DRIFT: manifest-derived cu12-tarball cargo_features (\${cu12_features}) no longer matches the literal jammi-server RELEASE tuple this leg builds (cuda,flash-attn,jetstream-broker,storage-cloud) -- update the literal (and its PROVE_TUPLE echo) in the SAME unit as the manifest edit" >&2
  grc=1
fi
if [ "\${ai_features}" != "cuda,flash-attn" ]; then
  echo "::error::PROVE_SURFACE_DRIFT: manifest-derived jammi-ai-applicable subset (\${ai_features}) no longer matches the literal jammi-ai TEST tuple's non-prove_only half (cuda,flash-attn)" >&2
  grc=1
fi
echo "PROVE_TUPLE crate=jammi-server kind=release features=cuda,flash-attn,jetstream-broker,storage-cloud"
cargo build --release -p jammi-server --bin jammi-server --features cuda,flash-attn,jetstream-broker,storage-cloud || grc=\$?
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability --no-run || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=capability-surface-build rc=\${grc}"
echo "::endgroup::"

# capability-surface-proof: \`capability_surface\` is delivered by ai-core in
# a LATER wave of THIS SAME PR (campaign #443) — this group must FAIL LOUD,
# never silently skip, if that test is absent from this ref: a name filter
# that matches zero tests exits 0 ("running 0 tests ... test result: ok"),
# which would otherwise read as a false-green capability proof.
echo "::group::capability-surface-proof"
grc=0
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cap_out="\$(JAMMI_KERNELS_STRICT=1 cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability capability_surface -- --nocapture 2>&1)"
cap_rc=\$?
echo "\${cap_out}"
if echo "\${cap_out}" | grep -q "running 0 tests"; then
  echo "::error::jammi-ai gpu_capability's capability_surface test matched ZERO tests on this ref — ai-core's capability-surface delivery (campaign #443) has not landed here; refusing to read a 0-test run as a pass" >&2
  grc=1
elif [ "\${cap_rc}" -ne 0 ]; then
  grc=\${cap_rc}
fi
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=capability-surface-proof rc=\${grc}"
echo "::endgroup::"

# served-client-server-proof: the shipped served attention surface (K4) --
# widened to the FULL jammi-server lane (jetstream-broker/storage-cloud are
# compile-time-only for this test binary; zero test hits under
# crates/jammi-server/tests/it*) plus live-gpu-tests, per
# prove_lane.crates.jammi-server in the manifest.
echo "::group::served-client-server-proof"
grc=0
echo "PROVE_TUPLE crate=jammi-server kind=test features=cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud"
cargo test -p jammi-server --features cuda,flash-attn,jetstream-broker,live-gpu-tests,storage-cloud --test it grpc_embedding_gpu -- --nocapture --test-threads=1 || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=served-client-server-proof rc=\${grc}"
echo "::endgroup::"

# engine-core-sweep: \`--skip capability_surface\` -- this generic sweep does
# not set JAMMI_KERNELS_STRICT=1 (other gpu_capability tests legitimately
# exercise the FALLBACK admission path, which strict mode would break), but
# capability_surface.rs's own module doc REQUIRES strict mode and asserts it
# at the top of the test — a real, fail-closed guard, correctly tripping
# "wrong mode" here. It is NOT skipped because it is unimportant: the
# dedicated capability-surface-proof group above already runs it, correctly,
# under JAMMI_KERNELS_STRICT=1. Never weaken capability_surface's own guard
# and never set strict mode on this generic sweep to work around it — do the
# opposite (name-exclude it from the one group that cannot satisfy its
# precondition). Widened to include flash-attn (prove_lane's jammi-ai `test`
# pair) — this leg now also exercises the shipped flash cascade through the
# engine-core suite, not only the dedicated capability-surface probe.
echo "::group::engine-core-sweep"
grc=0
echo "PROVE_TUPLE crate=jammi-ai kind=test features=cuda,flash-attn,live-gpu-tests"
cargo test -p jammi-ai --features cuda,flash-attn,live-gpu-tests --test gpu_capability -- --nocapture --test-threads=1 --skip capability_surface || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=engine-core-sweep rc=\${grc}"
echo "::endgroup::"

# kernels-default: jammi-kernels' own lib tests at DEFAULT features (records
# the x86_64 Linux run this pod is the only artifact for). prove_lane's own
# \`default\` kind for jammi-kernels -- canonicalizes to no \`--features\` flag
# at all, so it carries no gated tuple and needs no PROVE_TUPLE echo for
# `is_gated` purposes, but one is still emitted (features=<empty>) so the
# set-equality rule in check_flash_attn_closure.py has a uniform (crate,
# kind) -> literal pairing for EVERY declared prove_lane entry, gated or not.
echo "::group::kernels-default"
grc=0
echo "PROVE_TUPLE crate=jammi-kernels kind=default features="
cargo test -p jammi-kernels -- --nocapture --test-threads=1 || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=kernels-default rc=\${grc}"
echo "::endgroup::"

# kernels-cuda: widened to \`cuda,flash-attn\` (prove_lane's jammi-kernels
# \`test\` pair) -- this pod's GPU is the device the suite needs, and this is
# now the only lane that can compile \`cuda_parity\`'s flash-gated items and
# the \`flash_smoke\` target (\`required-features = ["flash-attn"]\`).
echo "::group::kernels-cuda"
grc=0
echo "PROVE_TUPLE crate=jammi-kernels kind=test features=cuda,flash-attn"
cargo test -p jammi-kernels --features cuda,flash-attn -- --nocapture --test-threads=1 || grc=\$?
[ "\$grc" -ne 0 ] && rc=\$grc
echo "PROVE_GROUP_RC name=kernels-cuda rc=\${grc}"
echo "::endgroup::"

# bench: recorded observability, deliberately NON-GATING (esc-082) -- \`bench_rc\`
# never touches \`rc\`, and this group runs LAST so a cut/hang here, with every
# group above already at rc=0, never blocks the leg (see the driver rule in
# runpod_gpu_prove.sh, after this heredoc). Widened to \`cuda,flash-attn\`
# (prove_lane's jammi-bench \`release\` pair).
echo "::group::bench"
bench_rc=0
echo "PROVE_TUPLE crate=jammi-bench kind=release features=cuda,flash-attn"
cargo run -p jammi-bench --release --features cuda,flash-attn -- gpu-inference-scale || bench_rc=\$?
echo "BENCH_EXIT=\${bench_rc}"
echo "PROVE_GROUP_RC name=bench rc=\${bench_rc}"
echo "::endgroup::"

echo "PROVE_EXIT=\${rc}"; exit \$rc
REMOTE
raw_rc="${PIPESTATUS[0]}"

rp_prove_verdict "$raw_rc" "$LOG"
rc=$?

rm -f "$LOG"
echo "=== GPU prove suites exit=${rc} (raw=${raw_rc}) ==="
exit "$rc"

fi # end sourced-execution guard
